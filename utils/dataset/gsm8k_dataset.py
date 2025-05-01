import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow.parquet as pq
import re
import json
import os
from utils.message_manager import Conversation, cList
from config import global_config
import re
import math
from asteval import Interpreter
import random
from utils.rl.functions import repeat_penalty


def validate_math_expression(expr, tolerance=1e-9):
    """
    验证带等号的数学表达式，检查所有等号左边和右边是否相等。

    参数:
        expr (str): 带等号的数学表达式，例如 "2+3=5=6-1"
        tolerance (float): 允许的浮点数误差阈值，默认为 1e-9

    返回:
        bool: 如果所有等号两边相等返回 True，否则返回 False
    """
    # 去除所有空格
    expr = "".join(expr.split())

    # 检查是否包含等号
    if "=" not in expr:
        return False

    # 将表达式按等号分割为多个部分
    parts = expr.split("=")
    if len(parts) < 2:
        return False

    # 初始化 asteval 解释器
    aeval = Interpreter()

    try:
        # 计算第一个部分的值
        previous_value = aeval(parts[0])
        if not isinstance(previous_value, (int, float)):
            return False

        # 依次比较相邻部分的值
        for part in parts[1:]:
            current_value = aeval(part)
            if not isinstance(current_value, (int, float)):
                return False

            # 比较当前部分与前一部分的值是否在容差范围内相等
            if abs(previous_value - current_value) >= tolerance:
                return False

            previous_value = current_value

        # 所有部分都相等
        return True

    except (ZeroDivisionError, ValueError, SyntaxError):
        # 处理数学错误（如除以零）或语法错误
        return False

def extract_and_format_math_expressions(text):
    # 定义数学表达式的正则模式
    # 匹配数字、运算符和括号的组合，包括嵌套括号
    pattern = r"\b\d+(?:\.\d+)?(?:\s*[-+*/=]\s*\d+(?:\.\d+)?)*\b|\([^()]*?(?:\([^()]*\)[^()]*)*?\)"
    detect_text = text.replace("x", "*").replace("X", "*").replace("×", "*").replace("÷", "/").replace(" ", "")
    # 查找所有匹配的表达式
    expressions = re.findall(pattern, detect_text)

    # 整理格式
    formatted_expressions = []
    for expr in expressions:
        # 去除所有空格
        expr = "".join(expr.split())
        formatted_expressions.append(expr)
    formatted_expressions = [x for x in formatted_expressions if "=" in x]

    return formatted_expressions


def find_last_number(s):
    # 使用正则表达式查找所有数字
    numbers = re.findall(r"-?\d+", s)
    if numbers:
        # 返回最后一个数字
        return numbers[-1]
    return None


class GSM8KDataset(Dataset):
    def __init__(self, parquet_file_path):
        self.table = pq.read_table(parquet_file_path)
        self.df = self.table.to_pandas()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row["question"]
        answer = row["answer"]

        # 使用正则表达式捕获推理过程
        reasoning_steps = re.findall(r"<<(.*?)>>", answer)

        # 使用正则表达式捕获最终结果
        final_result_match = re.search(r"#### (\d+)", answer)
        ground_truth = final_result_match.group(1) if final_result_match else None

        # 去除 <<>> 和 #### 及其后面的字符
        cleaned_answer = re.sub(r"<<.*?>>", "", answer)  # 去除 <<>> 及其内容
        cleaned_answer = re.sub(
            r"####.*", "", cleaned_answer
        )  # 去除 #### 及其后面的字符
        cleaned_answer += f"The answer is {ground_truth}."
        cleaned_answer = cleaned_answer.strip()  # 去除多余的空格

        return {
            "question": question,
            "reasoning_steps": reasoning_steps,
            "ground_truth": ground_truth,
            "cleaned_answer": cleaned_answer,
        }

    def save_rwkv_dataset(
        self, save_dir, req_role="conversation", resp_role="response"
    ):
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/train.jsonl", "w", encoding="utf-8") as f:
            for d in self:
                line = {
                    "data": [
                        {req_role: "Q: " + d["question"]},
                        {resp_role: "A: " + d["cleaned_answer"]},
                    ],
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


class GSM8KRLDataset(GSM8KDataset):
    def __init__(
        self,
        parquet_file_path,
        req_role="conversation",
        req_prefix="Q",
        resp_role="response",
        resp_prefix="A",
        tokenizer=None,
        ref_answer_prob=0.05,
    ):
        super().__init__(parquet_file_path)
        self.tokenizer = tokenizer
        self.req_role = req_role
        self.req_prefix = req_prefix
        self.resp_role = resp_role
        self.resp_prefix = resp_prefix
        self.ref_answer_prob = ref_answer_prob

    def reward_func(
        self,
        req_text_batch,
        speak_text_batch,
        upload_tokens_batch,
        speak_tokens_batch,
        reward_func_ground_truth_batch,
        **kwargs,
    ):
        B = len(speak_text_batch)
        rewards = torch.zeros((B), dtype=torch.float32)

        base_rwds = torch.tensor([0.5] * B, dtype=torch.float32)

        natural_speak_rwds = torch.tensor([0.5] * B, dtype=torch.float32)
        for b, (speak_text, reward_func_ground_truth, reasoning_steps) in enumerate(
            zip(
                speak_text_batch,
                reward_func_ground_truth_batch,
                kwargs["reasoning_steps_batch"],
            )
        ):

            speak_text = speak_text.strip()
            last_number = find_last_number(speak_text)
            reward_func_ground_truth = str(reward_func_ground_truth)
            eqs_in_speak_text = extract_and_format_math_expressions(speak_text)
            if speak_text.endswith(reward_func_ground_truth) or speak_text.endswith(
                f"{reward_func_ground_truth}."
                or speak_text.endswith(f"{reward_func_ground_truth}。")
                or f"The answer is {reward_func_ground_truth}" in speak_text
            ):
                rewards[b] = 1
            elif (
                last_number is not None and last_number == reward_func_ground_truth
            ):
                rewards[b] = 0.88
            else:
                rewards[b] = 0

            if "<|start|>" in speak_text:
                rewards[b] -= 0.5

            lines_in_speak_text = speak_text.split("\n")
            natural_speak_rwds[b] = repeat_penalty(
                natural_speak_rwds[b], lines_in_speak_text, penalty_factor=0.05
            )

            if not eqs_in_speak_text:
                base_rwds[b] -= 0.1
            # 数学表达式惩罚
            for eq in eqs_in_speak_text:
                is_equal = validate_math_expression(eq)
                print("=========>", eq, is_equal)
                if not is_equal:
                    base_rwds[b] *= 0.9
            rewards[b] = rewards[b] * 2 + base_rwds[b] + natural_speak_rwds[b]
            print(
                f"""batch {b}:
question: {req_text_batch[b]}
text = {speak_text}
ground_truth = {reward_func_ground_truth}
reward = {rewards[b].item()}
base_reward = {base_rwds[b].item()}
natural_speak_reward = {natural_speak_rwds[b].item()}"""
            )
        print(
            f"""
batch_size: {B}
batch_rewards: {rewards.tolist()}
=======================================              
"""
        )
        return rewards.unsqueeze(1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row["question"]
        answer = row["answer"]

        # 使用正则表达式捕获推理过程
        reasoning_steps = re.findall(r"<<(.*?)>>", answer)

        # 使用正则表达式捕获最终结果
        final_result_match = re.search(r"#### (\d+)", answer)
        ground_truth = final_result_match.group(1) if final_result_match else None

        # 去除 <<>> 和 #### 及其后面的字符
        cleaned_answer = re.sub(r"<<.*?>>", "", answer)  # 去除 <<>> 及其内容
        cleaned_answer = re.sub(
            r"####.*", "", cleaned_answer
        )  # 去除 #### 及其后面的字符
        cleaned_answer += f"The answer is {ground_truth}."
        cleaned_answer = cleaned_answer.strip()  # 去除多余的空格

        input_conversations = cList(
            [
                (
                    Conversation(
                        self.req_role,
                        f"{self.req_prefix}: {question}",
                    )
                    if random.random() > self.ref_answer_prob
                    else Conversation(
                        self.req_role,
                        f"{self.req_prefix}: {question}"
                        + f"(reference answer: {ground_truth})",
                    )
                ),
            ]
        )
        resp_start_with_tokens = global_config.role[self.resp_role][
            "prefix"
        ] + self.tokenizer.encode(f"{self.resp_prefix}:")

        return (
            input_conversations,
            resp_start_with_tokens,
            ground_truth,
            cleaned_answer,
            None,
            {"reasoning_steps": reasoning_steps},
        )
