import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import logging
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.cpp_extension import load
import types
from typing import Union, Optional, List, Tuple
from .states import LayerState, BlockStates

HEAD_SIZE = 64

# 配置日志记录器
logger = logging.getLogger(__name__)

# 内存监控阈值，当GPU内存使用率超过此值时触发清理
MEM_MONITOR_THRESHOLD = 0.8  # 默认为80%

def __nop(ob):
    """空操作函数，用于JIT编译开关"""
    return ob

MyModule = nn.Module
MyFunction = __nop

CHUNK_LEN = 24

# 修改路径设置，指向正确的CUDA文件位置
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda_dir = os.path.join(workspace_dir, "cuda")

flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', 
         f"-D_CHUNK_LEN_={CHUNK_LEN}", 
         "--use_fast_math", "-O3", 
         "-Xptxas -O3", "--extra-device-vectorization"]

load(name="wind_backstepping", 
     sources=[f'{cuda_dir}/wkv7_cuda.cu', f'{cuda_dir}/wkv7_op.cpp'], 
     is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    """RWKV v7 的核心计算函数，支持状态持久化"""
    @staticmethod
    def forward(ctx, w, q, k, v, z, b, prev_s=None, prev_sa=None):
        # B: batch size, T: sequence length, H: number of heads, C: head dimension (channels)
        B, T, H, C = w.shape 
        assert T % CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        
        # 如果提供了前一个状态，则使用它，否则创建新的空状态
        if prev_s is not None and prev_s.shape == (B, H, T//CHUNK_LEN, C, C):
            s = prev_s.clone()
        else:
            s = torch.zeros(B, H, T//CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
            if prev_s is not None:
                logger.warning(f"Invalid prev_s shape: expected {(B, H, T//CHUNK_LEN, C, C)}, got {prev_s.shape}. Creating new state.")
            
        # 处理上一个sa状态
        if prev_sa is not None:
            assert prev_sa.shape == (B, T, H, C), f"Expected shape: {(B, T, H, C)}, got: {prev_sa.shape}"
            sa = prev_sa.clone()
        else:
            sa = torch.zeros(B, T, H, C, dtype=torch.float32, device=w.device)
        
        # 调用CUDA操作
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        
        # 返回输出和最新状态
        return y, s, sa
        
    @staticmethod
    def backward(ctx, dy, ds=None, dsa=None):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db, None, None

def RUN_CUDA_RWKV7g(q, w, k, v, a, b, prev_s=None, prev_sa=None):
    """支持状态持久化的RWKV v7 CUDA核心计算函数包装器"""
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC//64, 64) for i in [q, w, k, v, a, b]]
    
    # 处理前一个状态
    if prev_s is not None:
        prev_s = prev_s.to(device=q.device, dtype=torch.float32)
    if prev_sa is not None:
        prev_sa = prev_sa.to(device=q.device, dtype=torch.float32)
        
    y, s, sa = WindBackstepping.apply(w, q, k, v, a, b, prev_s, prev_sa)
    return y.view(B, T, HC), s, sa


CHUNK_LEN_ONE = 2

flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', 
         f"-D_CHUNK_LEN_={CHUNK_LEN_ONE}", 
         "--use_fast_math", "-O3", 
         "-Xptxas -O3", "--extra-device-vectorization"]

load(name="wind_backstepping_one", 
     sources=[f'{cuda_dir}/wkv7_cuda_one.cu', f'{cuda_dir}/wkv7_op_one.cpp'], 
     is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBacksteppinOne(torch.autograd.Function):
    """RWKV v7 的核心计算函数，支持状态持久化"""
    @staticmethod
    def forward(ctx, w, q, k, v, z, b, prev_s=None, prev_sa=None):
        # B: batch size, T: sequence length, H: number of heads, C: head dimension (channels)
        B, T, H, C = w.shape 
        assert T % CHUNK_LEN_ONE == 0
        assert all(i.dtype==torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        
        # 如果提供了前一个状态，则使用它，否则创建新的空状态
        # todo 暂时先这么处理，需要要考虑合并形状不一致的state的办法
        if prev_s is not None and prev_s.shape == (B, H, T//CHUNK_LEN_ONE, C, C):
            s = prev_s.clone()
        else:
            s = torch.zeros(B, H, T//CHUNK_LEN_ONE, C, C, dtype=torch.float32, device=w.device)
            if prev_s is not None:
                logger.warning(f"Invalid prev_s shape: expected {(B, H, T//CHUNK_LEN_ONE, C, C)}, got {prev_s.shape}. Creating new state.")
            
        # 处理上一个sa状态
        if prev_sa is not None:
            assert prev_sa.shape == (B, T, H, C), f"Expected shape: {(B, T, H, C)}, got: {prev_sa.shape}"
            sa = prev_sa.clone()
        else:
            sa = torch.zeros(B, T, H, C, dtype=torch.float32, device=w.device)
        
        # 调用CUDA操作
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        
        # 返回输出和最新状态
        return y, s, sa
        
    @staticmethod
    def backward(ctx, dy, ds=None, dsa=None):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db, None, None

def RUN_CUDA_RWKV7g_One(q, w, k, v, a, b, prev_s=None, prev_sa=None):
    """支持状态持久化的RWKV v7 CUDA核心计算函数包装器"""
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC//64, 64) for i in [q, w, k, v, a, b]]
    
    # 处理前一个状态
    if prev_s is not None:
        prev_s = prev_s.to(device=q.device, dtype=torch.float32)
    if prev_sa is not None:
        prev_sa = prev_sa.to(device=q.device, dtype=torch.float32)
        
    y, s, sa = WindBacksteppinOne.apply(w, q, k, v, a, b, prev_s, prev_sa)
    return y.view(B, T, HC), s, sa


class RWKV_Tmix_x070(MyModule):
    """RWKV v7 的时间混合模块"""
    def __init__(self, dim_att, n_embd, n_layer, head_size_divisor, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = HEAD_SIZE
        self.n_head = dim_att // self.head_size
        assert dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            if self.layer_id!=0:
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)


            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(head_size_divisor**2)) # !!! notice eps value !!!

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first, layer_state=None):
        """前向传播函数，支持状态持久化"""
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        
        # 提取前一个状态（如果有）
        prev_s = None
        prev_sa = None
        if layer_state is not None:
            prev_s, prev_sa = layer_state.get()
        
        # 使用状态进行计算
        if True:
            x, s, sa = RUN_CUDA_RWKV7g_One(r, w, k, v, -kk, kk*a, prev_s, prev_sa)
        else:
            x, s, sa = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a, prev_s, prev_sa)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        
        # 创建新的LayerState
        new_layer_state = LayerState(s, sa)
        return x, v_first, new_layer_state
    

class RWKV_CMix_x070(MyModule):
    """RWKV v7 的通道混合模块"""
    def __init__(self, n_embd, n_layer, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.value = nn.Linear(n_embd * 4, n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(n_embd**0.5), 0.5/(n_embd**0.5))
        # self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        """前向传播函数"""
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)
    

class Block(nn.Module):
    """RWKV模型的基本块，包含时间混合和通道混合"""
    def __init__(self, n_embd, dim_att, n_layer, head_size_divisor, dropout, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_Tmix_x070(dim_att, n_embd, n_layer, head_size_divisor, layer_id)
        self.ffn = RWKV_CMix_x070(n_embd, n_layer, layer_id)

        self.dropout = dropout
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)

    def forward(self, x, v_first, layer_state=None):
        """前向传播函数，支持状态持久化"""
        if self.layer_id == 0:
            x = self.ln0(x)

        # 时间混合层（支持状态）
        x_attn, v_first, att_layer_state = self.att(self.ln1(x), v_first, layer_state)
        x = x + x_attn

        # 通道混合层（不需要状态）
        x = x + self.ffn(self.ln2(x))
        
        # 返回当前block的LayerState
        return x, v_first, att_layer_state
    
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class RWKV(nn.Module):
    """RWKV v7 模型主类，支持状态持久化"""
    def __init__(
        self,
        # Model architecture parameters
        n_embd=-1,
        n_layer=-1,
        vocab_size=-1,
        head_size=64,
        head_size_divisor=8,
        ctx_len=1024,
        
        # Model loading parameters
        load_model=None,
        dtype="bf16",

        # Training parameters
        dropout=0.0,
        grad_cp=1):

        super().__init__()
        
        # 检查CUDA是否可用
        if not RWKV.check_cuda_available():
            logger.warning("CUDA不可用，模型将在CPU上运行，性能可能受到影响")
            
        # Unified dtype handling
        dtype_map = {
            "fp32": torch.float,
            "fp16": torch.half,
            "bf16": torch.bfloat16
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)
        self.rnn_model = None
        # Ensure load_model is specified
        assert load_model is not None, "load_model must be specified"
        
        # Load weights
        model_weights = torch.load(load_model, map_location='cpu')
        
        # Calculate init layer if not specified
        if n_layer < 0:
            max_block_id = max((int(x.split('.')[1]) for x in model_weights.keys() if 'blocks.' in x), default=0)
            n_layer = max_block_id + 1

        # Calculate embedding size if not specified
        if n_embd < 0:
            n_embd = model_weights['head.weight'].shape[1]

        # Calculate vocab size if not specified
        if vocab_size < 0:
            vocab_size = model_weights['head.weight'].shape[0]

        dim_att = n_embd
        self.n_head = dim_att // head_size
        self.head_size = head_size
        self.dim_ffn = int((n_embd * 3.5) // 32 * 32)  # Not used but kept for reference

        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, dim_att, n_layer, head_size_divisor, dropout, i) 
            for i in range(n_layer)
        ])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Init dropout
        self.dropout = dropout
        if dropout > 0:
            self.drop0 = nn.Dropout(p=dropout)

        # Store necessary parameters
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.grad_cp = grad_cp

        # Load state dict
        self.load_state_dict(model_weights)

        # Convert parameters to the specified dtype
        for p in self.parameters():
            p.data = p.data.to(device=self.get_device(),dtype=self.dtype)

        # Clean up
        del model_weights
        self.clear_gpu_memory(force=True)

        # 初始化为None，稍后会改为BlockStates对象
        self.rnn_states = None

    @staticmethod
    def check_cuda_available():
        """检查CUDA是否可用"""
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU运行模型")
            return False
        return True
    
    @staticmethod
    def get_gpu_memory_usage():
        """获取当前GPU内存使用情况"""
        if not torch.cuda.is_available():
            return 0, 0, 0
        
        # 获取当前设备
        device = torch.cuda.current_device()
        
        # 获取分配的内存
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
        
        # 获取缓存的内存
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
        
        # 获取总内存
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
        
        return allocated, reserved, total
    
    @staticmethod
    def clear_gpu_memory(force=False):
        """清理GPU内存"""
        if not torch.cuda.is_available():
            return
        
        allocated, reserved, total = RWKV.get_gpu_memory_usage()
        usage_ratio = allocated / total if total > 0 else 0
        
        # 只有当内存使用率超过阈值或强制清理时才执行
        if force or usage_ratio > MEM_MONITOR_THRESHOLD:
            logger.info(f"清理GPU内存 (使用率: {usage_ratio:.2f})")
            gc.collect()
            torch.cuda.empty_cache()
            
            # 清理后再次检查
            new_allocated, _, _ = RWKV.get_gpu_memory_usage()
            logger.info(f"清理后GPU内存: {new_allocated:.2f}GB (减少了 {allocated - new_allocated:.2f}GB)")
    
    def optimize_tensor(self, tensor, dtype=None, device=None, inplace=False):
        """优化张量的内存使用"""
        if tensor is None:
            return None
            
        # 如果不指定dtype，使用模型的默认dtype
        if dtype is None:
            dtype = self.dtype
            
        # 如果不指定device，使用模型的device
        if device is None:
            device = self.get_device()
            
        # 如果不是inplace操作，创建新张量
        if not inplace:
            return tensor.to(dtype=dtype, device=device)
            
        # inplace操作
        tensor.data = tensor.data.to(dtype=dtype, device=device)
        return tensor
    
    def get_optim_groups(self, layerwise_lr=0.0, weight_decay=0.01, my_pile_stage=0):
        """
        将模型参数分组以应用不同的学习率和权重衰减
        
        参数:
            layerwise_lr: 是否使用分层学习率（0表示禁用）
            weight_decay: 权重衰减系数
            my_pile_stage: Pile数据集的训练阶段
            
        返回:
            参数分组列表，用于优化器初始化
        """
        # 参数分组
        lr_decay = set()  # 需要权重衰减的参数
        lr_1x = set()     # 基础学习率参数
        lr_2x = set()     # 2倍学习率参数
        lr_3x = set()     # 3倍学习率参数
        
        # 根据参数名称进行分类
        for n, p in self.named_parameters():
            # 权重矩阵参数
            if (("_w1" in n) or ("_w2" in n)) and (layerwise_lr > 0):
                lr_1x.add(n)
            # 时间状态参数（需要权重衰减）
            elif (("time_sta" in n) and (weight_decay > 0)):
                lr_decay.add(n)
            # 时间混合参数
            elif (("time_mix" in n) or ("time_maa" in n)) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            # 时间衰减参数
            elif (("time_decay" in n) or ("time_daaaa" in n) or ("att.w0" in n)) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            # 其他时间相关参数
            elif ("time_faaaa" in n) and (layerwise_lr > 0):
                if my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (layerwise_lr > 0):
                lr_3x.add(n)
            # 二维及以上权重矩阵
            elif (len(p.squeeze().shape) >= 2) and (weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            # 其他所有参数
            else:
                lr_1x.add(n)

        # 创建参数字典
        param_dict = {n: p for n, p in self.named_parameters()}
        
        # 构建优化器分组
        optim_groups = []
        if layerwise_lr > 0:
            if my_pile_stage == 2:
                # Pile阶段2使用特殊的学习率缩放
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},
                ]
            else:
                # 标准分层学习率
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            # 不使用分层学习率
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        # 添加需要权重衰减的参数组
        if weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": weight_decay, "my_lr_scale": 1.0}]
            
        return optim_groups
    
    
    def create_optimizer(self, lr_init=5e-4, lr_final=1e-5, beta1=0.9, beta2=0.99, 
                           adam_eps=1e-8, weight_decay=0.01, layerwise_lr=0.0, 
                           my_pile_stage=0, adamw_mode=True, warmup_steps=1000):
        """
        创建RWKV模型的优化器和学习率调度器
        
        参数:
            model: RWKV模型实例
            lr_init: 初始学习率
            lr_final: 最终学习率
            beta1, beta2: Adam优化器的beta参数
            adam_eps: Adam优化器的epsilon值
            weight_decay: 权重衰减系数
            layerwise_lr: 是否使用分层学习率
            my_pile_stage: Pile数据集的训练阶段
            adamw_mode: 是否使用AdamW模式（否则使用Adam）
            warmup_steps: 预热步数
            
        返回:
            optimizer: 配置好的优化器
            lr_scheduler: 配置好的学习率调度器
        """
        # 在创建优化器前清理内存
        if hasattr(self, 'clear_gpu_memory'):
            self.clear_gpu_memory(force=True)
        else:
            gc.collect()
            torch.cuda.empty_cache()
            
        # 获取参数分组
        optim_groups = self.get_optim_groups(layerwise_lr, weight_decay, my_pile_stage)
        
        # 创建DeepSpeed CPU Adam优化器
        optimizer = DeepSpeedCPUAdam(
            optim_groups,
            lr=lr_init,
            betas=(beta1, beta2),
            eps=adam_eps,
            adamw_mode=adamw_mode,
            weight_decay=weight_decay,
            amsgrad=False,
            bias_correction=True,
        )
        
        # 创建预热学习率调度器
        lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
            optimizer,
            warmup_min_lr=lr_final,
            warmup_max_lr=lr_init,
            warmup_num_steps=warmup_steps,
            warmup_type="linear",
        )
        
        # 创建优化器后再次清理内存
        if hasattr(self, 'clear_gpu_memory'):
            self.clear_gpu_memory()
        else:
            gc.collect()
            torch.cuda.empty_cache()
            
        return optimizer, lr_scheduler


    def reset_states(self):
        """重置所有层的状态"""
        self.rnn_states = BlockStates()

    def forward(self, idx, v_first=None, use_states=True):
        """支持状态持久化的前向传播"""
        B, T = idx.size() if isinstance(idx, torch.Tensor) else (len(idx), len(idx[0]))
        C = self.n_embd
        H = C // HEAD_SIZE

        assert T <= self.ctx_len, f"Cannot forward sequence of length {T}, model ctx_len is {self.ctx_len}."
        
        # 检查当前内存使用情况
        if torch.cuda.is_available():
            allocated, _, total = self.get_gpu_memory_usage()
            if allocated / total > MEM_MONITOR_THRESHOLD * 0.9:
                logger.info(f"前向传播前清理GPU内存，当前使用率: {allocated/total:.2f}")
                self.clear_gpu_memory()
        
        # 转换输入为tensor
        if not isinstance(idx, torch.Tensor):
            x = torch.tensor(idx, device=self.get_device(), dtype=torch.long)
        else:
            x = idx.to(self.get_device())
            
        x = self.emb(x)
        x = self.optimize_tensor(x)

        if self.dropout > 0:
            x = self.drop0(x)

        # 初始化v_first
        if v_first is None:
            v_first = torch.zeros_like(x, device=x.device)
        else:
            v_first = self.optimize_tensor(v_first)

        # 确保有BlockStates对象
        if use_states and self.rnn_states is None:
            self.rnn_states = BlockStates()
        
        # 初始化新状态
        new_states = BlockStates()
        
        # 通过blocks处理
        for i, block in enumerate(self.blocks):
            # 获取当前层的前一个状态
            layer_state = None
            if use_states and self.rnn_states is not None:
                layer_state = self.rnn_states[i]
                
            # 处理当前块
            if self.grad_cp == 1 and not use_states:  # 状态持久化模式不使用checkpointing
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first, state = block(x, v_first, layer_state)
                new_states[i] = state
                
            # 内存管理
            if (i + 1) % 8 == 0 and torch.cuda.is_available() and self.n_layer > 16:
                allocated, _, total = self.get_gpu_memory_usage()
                if allocated / total > MEM_MONITOR_THRESHOLD * 0.8:
                    logger.debug(f"处理第{i+1}层后清理GPU内存")
                    self.clear_gpu_memory()

        x = self.ln_out(x)
        logits = self.head(x)

        # 更新模型状态（仅当启用状态持久化时）
        if use_states:
            # 将状态移到CPU并分离计算图
            self.rnn_states = new_states.detach().cpu()

        return logits, v_first


    def get_device(self):
        """Helper to get the device of the model"""
        return next(self.parameters()).device
        
    def get_states(self):
        """获取当前模型状态
        
        Returns:
            BlockStates: 模型各层的状态，可用于保存或分析
        """
        return self.rnn_states
    
    def set_states(self, states):
        """设置模型状态
        
        Args:
            states (BlockStates): 要设置的模型状态
            
        Returns:
            self: 返回模型自身，支持链式调用
        """
        if isinstance(states, BlockStates):
            self.rnn_states = states
        else:
            # 兼容旧格式的状态
            self.rnn_states = BlockStates()
            if isinstance(states, list):
                for i, state_tuple in enumerate(states):
                    if state_tuple is not None:
                        self.rnn_states[i] = state_tuple
        return self
        

if __name__ == "__main__":
    
    model = RWKV(load_model="/home/neromous/MachineLr/datadisk/rwkv-checkpoints/basemodel/rwkv-x070-2b9-world-v3-53%trained-20250121-ctx4k.pth")
    
    optimizer, lr_scheduler = model.create_optimizer()

    ds_config = {
        "bfloat16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": 4
    }

    # 获取模型所在设备
    device = model.get_device()
    print(f"模型设备: {device}")
    
    # 确保输入张量在同一设备上
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], 
                            device=device)
    target_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], 
                            device="cuda")
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        #args=args, 
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config
    )
    # 执行前向传播
    logits, v_first = model_engine(input_ids)
    print(f"输出形状: {logits.shape}")

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.reshape(-1))
    print(f"损失: {loss}")

    loss = L2Wrap.apply(loss, logits)
    print(f"损失: {loss}")

    model_engine.backward(loss)
    model_engine.step()



        # 执行前向传播
    logits, v_first = model_engine(input_ids, use_states=True)
    print(f"输出形状: {logits.shape}")

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.reshape(-1))
    print(f"损失: {loss}")

    loss = L2Wrap.apply(loss, logits)
    print(f"损失: {loss}")

    model_engine.backward(loss)
    model_engine.step()


    
    # 检查logits中是否存在nan值
    if torch.isnan(logits).any():
        print("警告: logits中存在NaN值")
        # 获取nan的具体位置
        nan_indices = torch.where(torch.isnan(logits))
        print(f"NaN值位置: {nan_indices}")
    