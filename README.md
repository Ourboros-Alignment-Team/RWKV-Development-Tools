<div align="center">

<h1>RWKV-Development-Tools</h1>

[![Code License](https://img.shields.io/badge/LICENSE-Apache2.0-green.svg?style=for-the-badge)](https://github.com/Ourboros-Alignment-Team/RWKV-Development-Tools/tree/main/LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-yellow.svg?style=for-the-badge)](https://img.shields.io/badge/python-3.11-yellow.svg?style=for-the-badge)
[![QQ Group](https://img.shields.io/badge/qq%20group-873610818-blue?style=for-the-badge)](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=zcGtQcxps3ZEtGwV0-qdHFF2RULZOnQ4)

</div>

`RWKV-Development-Tools` 是一个面向RWKV模型的研究平台。

✅ 提供多种可视化工具，满足语言建模等任务下的研究、使用需求。

✅ 完善的类RNN模型历史管理工具。

✅ 提供OpenAI API协议的对话接口的前端用于测试各种模型（需提供API Key）。

✅ 一套工作流允许用户快速将对话历史送入模型sft，或是打标签后强化学习。

✅  提供多智能体框架，支持在线学习。

✅  支持 RWKV V7

✅  GRPO强化学习的支持

✅  一个多智能体的框架， Ourborous，以及它的前端界面。

✅  翻新后端服务，并支持批量推理。


---
## TODO

⏳ 项目文档。

⏳ 接入mcp和blackfog

⏳ 端到端多模态训练与推理。

## 使用方法

创建虚拟环境：
```bash
conda create [你的后端环境名] python=3.11.3
conda create [你的前端环境名] python=3.11.3
```

安装后端依赖：
```bash
conda activate [你的后端环境名]
pip install -r requirements_backend.txt
```

安装前端与智能体工具依赖：
```bash
conda activate [你的前端与智能体工具环境名]
pip install -r requirements_frontend_and_tools.txt
conda install -c conda-forge opencc
```

**请勿将前后端依赖装在一起，它们会互相冲突！**

调整.configs/config2.0.json中的路径依赖

开启推理服务：
```bash
conda activate [你的后端环境名]
python inference_service.py
```

开启训练服务（可选）：
```bash
conda activate [你的后端环境名]
deepspeed --num_gpus 1 train_api.py
```

启动webui：
```bash
conda activate [你的前端环境名]
python playground_webui.py
```
建议使用http://127.0.0.1:7860/?__theme=dark 而不是 http://127.0.0.1:7860/ 启动webui。



## 快速开始

- 前往huggingface下载最新的[rwkv模型](https://huggingface.co/BlinkDL/rwkv-6-world)，
或者使用本项目提供的示例微调模型[rwkv-ruri-3b](https://huggingface.co/ssg-qwq/rwkv-ruri-3b)。
- 聊天
    - 将模型文件下载至某目录下，并将`.configs/config2_0.json`中的`ckpt_dir`改为该目录。
    - 将`.configs/config2_0.json`中`infer_service_config`字段的`load_model`改为模型路径。
    - 将`.configs/config2_0.json`中`infer_service_config`字段的`device`改为你需要的GPU编号。
    - 开启推理服务和webui，即可聊天。

- 标注
    - 聊天时，点击`保存生成历史(强化学习)`，然后切换至`标注`，即可在下拉菜单中选择保存的历史记录。可以为每次重新生成的选项记录`score`和`safety`标签，或者确认最佳选项。

- 训练
    - 聊天时，点击`保存数据集`即可将当前聊天历史保存为数据集。
    - 将`.configs/config2_0.json`中`train_service_config`字段的`device`改为你需要的GPU编号。
    - 将数据集整理成数据文件夹后，切换至`训练`，键入该文件夹地址后，点击`开始训练`即可尝试微调。
    - 训练前，需要开启训练服务。


## 训练

### 配置需求

目前依赖cuda算子运行。

| 训练模式 | 模型规模 | 内存消耗 | 显存消耗 | batch_size |
|------|------|------|------|------|
| **全量训练** | 3B (ds stage2 offload) | 64GB | 12GB | 1 |
|  | 7B (ds stage2 offload) | 256GB | 24GB | 1 |
|  | 14B | >320GB | >48GB | 1 |
| **低秩训练** | 3B (ds stage1) | 64GB | 12GB | 4 |
|  | 7B (ds stage1) | 64GB | 24GB | 4 |
|  | 14B | >128GB | >24GB | 4 |
---
### 训练数据准备
`RWKV-Development-Tools`会以桶索引的方式处理并读取数据集。
目前支持四种数据集类型。

➡️ **txt文本数据集**<br>
txt文件数据集是一个没有任何格式需求的txt文件，在tokenize时，系统会自动在txt的末尾加上eod (0号token)。<br>
目前请不要使用太大的文件，会对创建索引的速度产生影响。

➡️ **RWKV风格数据集**<br>
RWKV风格数据集是一个jsonl文件，其每一行格式如：<br>
```
{"text": 数据内容}
```
- 系统会自动在每一行的末尾加上eod。<br>
- 为避免单个文件的行数太多。建议将行数过多 (>10000)或单行字数过多的数据保存为多个文件。<br>
数据集例:<br> 
```
{"text": "这是一段文本\n\nuser: 文本中说了什么？\n\nassistant: 文本中说了xxx\n\n"}
```

➡️ **ourborous V1风格数据集**<br>
ourborous风格数据集可以十分简洁地管理special tokens，每一行格式如: <br>
```
{"data": [{sp_token_role: content}, {sp_token_role: content}]}
```
- 其中：<br>
`sp_token_role`: 字符串，用来决定special tokens的角色标签，常用的标签有 `conversations`（非bot发言）, `response`bot发言）, `think`思考）等。<br>
`content` : 字符串，完整对话内容，例如"**assistant:** 是的。"<br>
- **请不要与一般的OPENAI协议对话数据集混淆！** `content`中应包含说话人的前缀字符，例如 `assistant: ` <br>
数据集例:<br>
```
{"data": [{"conversation": "user: 你好"}, {"response": "assistant: 你好。"}]}
```

➡️ **ourborous V2风格数据集**<br>
ourborous-V2风格数据集用于管理更加复杂的数据形式，每一行格式如:<br>
```
{"data_dict_list": [{"role":sp_token_role, "content": content}, {"role": sp_token_role, "content": content}]}
```
- 数据集例:<br>
```
{"data_dict_list": [{"role":"conversation", "content":"user: 你好"}, {"role": "response", "content":"assistant: <-voice->", "voice":"/home/你好.wav"}]}
```

---
请将以上数据集准备好，放入一个**数据文件夹**内<br>
数据文件夹结构如：<br>
```
-- 数据文件夹名字
    -- 数据子文件夹名字1
        -- 数据文件1.txt
        -- 数据文件114514.jsonl
        -- 任意数据文件名.jsonl
    -- 数据子文件夹名字2
        -- 数据文件.jsonl
    -- ...
```
- 数据文件夹内的所有子文件会被遍历读取。在第一次读取时，系统将自动在最外层文件夹内生成索引，包括: `in_line_indices`文件夹、`dataset_metadata.json`和`idx.index`。<br>
- 目前暂不支持动态检验并更改这些索引，如果调整了数据集内部结构，请删除这些文件。<br>
- 另外，如果数据集很大，第一次读取数据文件夹可能较慢。

<!-- ---
### 强化学习数据准备 -->

---
### 预训练

webui仅支持单gpu训练，如果想要使用多gpu训练，请使用预训练工具。

#### 调整config.json

预训练的配置位于`pretrain_script_config`字段。请调整需要的配置，尤其是正确的`dataset_folder`。

#### 执行训练

```bash
deepspeed --num_gpus [需要gpu数量] pre_train.py
```
或者
```bash
deepspeed --include localhost: [gpu编号列表] pre_train.py
```

