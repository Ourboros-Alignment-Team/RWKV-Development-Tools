import os
import torch
import deepspeed
import gc
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math
import json
from RWKV.v7.triton.model import RWKV, L2WrapInfctx
from RWKV.v7.triton.state_manager import RWKVStateManager
from RWKV.v7.triton.state import BlockState, BlockStateList
from config import global_config
import torch.nn as nn

# 设置环境变量
os.environ["RWKV_HEAD_SIZE_A"] = "64"  # 根据您的模型需求调整

# 导入RWKV模型
from RWKV.functions import window_inference

# 初始化分词器 (这里您需要导入您使用的分词器)
# 简单示例：从transformers导入
# Initialize tokenizer
from config import global_config
tokenizer = global_config.tokenizer_eval


# 读取训练示例文本
with open("resources/docs/RWKV7结构解析.md", "r", encoding="utf-8") as f:
    eval_text = f.read()

# 进行分词，确保长度能被CHUNK_LEN整除
CHUNK_LEN = 24  # 您的模型使用的chunk长度
eval_tokens = tokenizer.encode(eval_text)
eval_tokens_len = (len(eval_tokens) // CHUNK_LEN) * CHUNK_LEN
eval_tokens = eval_tokens[:eval_tokens_len]


class RWKVDataset(Dataset):
    def __init__(self, size=50, seq_length=512):
        self.size = size
        # 确保序列长度是CHUNK_LEN的倍数
        self.seq_length = (seq_length // CHUNK_LEN) * CHUNK_LEN
        self.eval_tokens = tokenizer.encode(eval_text)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        
        # 随机选择起始位置
        if len(self.eval_tokens) >= self.seq_length + 1:
            start_idx = torch.randint(0, len(self.eval_tokens) - self.seq_length - 1, (1,)).item()
            inputs = torch.tensor(self.eval_tokens[start_idx:start_idx + self.seq_length])
            targets = torch.tensor(self.eval_tokens[start_idx + 1:start_idx + self.seq_length + 1])
        else:
            # 文本不够长则使用随机数据
            inputs = torch.randint(0, tokenizer.vocab_size, (self.seq_length,))
            shifted = torch.roll(inputs, shifts=-1)
            targets = shifted.clone()
            targets[-1] = torch.randint(0, tokenizer.vocab_size, (1,))
        return inputs, targets


def test_inference(model, tokenizer):
    """测试模型的推理能力"""
    print("\n===== 开始推理测试 =====")
    
    model.eval()
    
    prompt = "RWKV是一种创新的语言模型架构，它结合了"
    prompt_tokens = tokenizer.encode(prompt)
    
    print(f"提示: '{prompt}'")
    print("生成中...")
    
    with torch.no_grad():
        # 初始化状态
        B = 1
        states = None  # 模型会处理无状态情况
        
        generated_tokens = []
        max_new_tokens = 200
        
        # 处理初始提示
        input_ids = torch.tensor([prompt_tokens], device=model.device)
        
        # 生成新tokens
        for _ in range(max_new_tokens):
            # 使用模型的forward方法进行推理
            logits, states = model(input_ids, states)
            
            # 获取最后一个token的预测
            next_token_logits = logits[:, -1, :]
            
            # 简单采样 (可以替换为您喜欢的采样方法)
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            generated_tokens.append(next_token)
            
            # 早停条件
            if next_token == tokenizer.eos_token_id:
                break
                
            # 准备下一次输入 (只用最后预测的token)
            input_ids = torch.tensor([[next_token]], device=model.device)
    
    generated_text = tokenizer.decode(generated_tokens)
    print("\n===== 生成结果 =====")
    print(prompt + generated_text)
    print("=====================")
    
    model.train()
    return generated_text


def load_config(config_path="./configs/default.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def clear_memory():
    """
    清除PyTorch的梯度和缓存，释放GPU内存
    
    这个函数执行以下操作:
    1. 清除所有张量的梯度
    2. 清空PyTorch的CUDA缓存
    3. 调用Python的垃圾回收器
    """
    # 清除所有张量的梯度
    torch.cuda.empty_cache()
    
    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 调用Python的垃圾回收器
    gc.collect()
    
    print("已清除PyTorch梯度和CUDA缓存，释放GPU内存")


def train_with_state(config_path="./configs/default.json"):
    """
    使用state管理训练RWKV模型
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    train_config = config["train_service_config"]
    model_config = train_config["model"]
    training_params = train_config["train"]
    ds_config = train_config["deepspeed"]
    
    # 设置环境变量
    os.environ["RWKV_HEAD_SIZE_A"] = str(model_config["head_size"])
    
    # 模型参数
    load_model = model_config["load_model"]
    n_embd = model_config["n_embd"]
    n_layer = model_config["n_layer"]
    vocab_size = model_config["vocab_size"]
    ctx_len = model_config["ctx_len"]
    head_size = model_config["head_size"]
    head_size_divisor = model_config["head_size_divisor"]
    dtype = model_config["dtype"]
    chunk_len = model_config["chunk_len"]
    
    # 训练参数
    dropout = training_params["dropout"]
    lr_init = training_params["lr_init"]
    lr_final = training_params["lr_final"]
    beta1 = training_params["beta1"]
    beta2 = training_params["beta2"]
    adam_eps = training_params["adam_eps"]
    weight_decay = training_params["weight_decay"]
    grad_cp = training_params["grad_cp"]
    layerwise_lr = training_params["layerwise_lr"]
    warmup_steps = training_params["warmup_steps"]
    
    # 数据处理参数
    batch_size = config["pretrain_script_config"]["batch_size"]
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化RWKV模型
    model = RWKV(
        n_embd=n_embd,
        n_layer=n_layer,
        vocab_size=vocab_size,
        ctx_len=ctx_len,
        head_size=head_size,
        head_size_divisor=head_size_divisor,
        dropout=dropout,
        grad_cp=grad_cp,
        load_model=load_model,
        dtype=dtype,
        weight_decay=weight_decay,
        trainer=training_params  # 传递训练配置
    )
    
    # 为模型设置训练所需的args属性
    class Args:
        def __init__(self):
            self.n_embd = model.n_embd
            self.dim_att = model.dim_att
            self.head_size_a = model.head_size
            self.n_layer = model.n_layer
            self.chunk_ctx = chunk_len
            
    model.args = Args()
    
    # 创建优化器和学习率调度器
    optimizer, lr_scheduler = model.get_optim_groups()
    
    # DeepSpeed配置
    ds_config_dict = {
        "bfloat16": {
            "enabled": "auto" if dtype == "bf16" else False
        },
        "zero_optimization": {
            "stage": ds_config["ds_stage"],
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            } if ds_config["offload_optimizer"] else None,
            "overlap_comm": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": ds_config["gradient_accumulation_steps"],
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": batch_size
    }
    
    # 初始化数据集
    print("准备数据集...")
    # 使用已经准备好的eval_tokens数据，而不是随机生成的数据
    sequence_length = ctx_len
    dataset = RWKVDataset(size=config["pretrain_script_config"].get("dataset_size", 1000), seq_length=sequence_length)
    
    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_dict,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config["pretrain_script_config"]["num_workers_per_gpu"]
    )
    
    # 训练循环
    epochs = config["pretrain_script_config"]["epoches"]
    save_steps = config["pretrain_script_config"]["save_weight_steps"]
    
    # 添加更多灵活的保存配置选项
    save_epoch_interval = config["pretrain_script_config"].get("save_epoch_interval", 20)  # 默认每个epoch都保存
    save_batch_checkpoints = config["pretrain_script_config"].get("save_batch_checkpoints", False)  # 默认不保存中间批次
    
    print(f"开始训练: {epochs}个epochs, " + 
          (f"每{save_steps}步保存一次中间检查点, " if save_batch_checkpoints else "") +
          f"每{save_epoch_interval}个epoch保存一次最终检查点")
    
    for epoch in range(epochs):
        total_loss = 0
        
        # 初始化 BlockStateList 用于训练
        states = BlockStateList.create(
            model.n_layer, 
            batch_size, 
            model.n_embd, 
            model.dim_att // model.head_size, 
            model_engine.device, 
            model_engine.module.emb.weight.dtype
        )
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(model_engine.device), target.to(model_engine.device)
            
            # 使用模型的training_step进行前向传播、损失计算和反向传播
            model_engine.zero_grad()
            
            # 创建批次数据以传递给training_step
            batch = (data, target)
            
            # 使用模型的training_step方法
            loss = model_engine.training_step(batch, batch_idx)
            print(f"loss: {loss}")
            # 向后传播和优化
            model_engine.backward(loss)
            model_engine.step()
            
        
            # 记录损失
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            # 保存中间检查点 (仅当开启了中间检查点保存选项时)
            if save_batch_checkpoints and batch_idx > 0 and batch_idx % save_steps == 0:
                save_path = os.path.join(
                    config["ourborous_config"]["save_ckpt_dir"], 
                    f"rwkv_epoch{epoch}_batch{batch_idx}.pt"
                )
                model_engine.save_checkpoint(save_path)
                print(f"中间检查点已保存至 {save_path}")
        
        # 计算epoch的平均损失
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch: {epoch}, 平均损失: {avg_loss:.4f}')
        
        # 每隔save_epoch_interval个epoch保存一次
        if epoch % save_epoch_interval == 0 or epoch == epochs - 1:  # 确保最后一个epoch总是保存
            save_path = os.path.join(
                config["ourborous_config"]["save_ckpt_dir"], 
                f"rwkv_epoch{epoch}_final.pt"
            )
            model_engine.save_checkpoint(save_path)
            print(f"Epoch {epoch} 完成, 模型已保存至 {save_path}")





if __name__ == "__main__":
    train_with_state() 