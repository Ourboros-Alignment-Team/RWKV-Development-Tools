import torch
from typing import List
from utils.message_manager import cList
import re


def remove_regix(text, regix_pattern=r"^\d+\."):
    return re.sub(regix_pattern, "", text).strip()


def partial_equal_line(str1, str2, min_n_repeat=20, tolerance=5):
    if min_n_repeat <= 0:
        return str1 == str2
    if len(str1) < min_n_repeat or len(str2) < min_n_repeat:
        return False
    n_repeat_str = 0
    n_diff_str = 0
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            n_repeat_str += 1
        else:
            n_diff_str += 1
            if n_diff_str > tolerance:
                break
    return n_repeat_str >= min_n_repeat


def repeat_penalty(
    rewards: torch.Tensor,
    line_texts: List[str],
    penalty_factor: float = 0.05,
    min_n_repeat: int = -1,
    ignore_regix: str = r"^\d+\.",
    use_presence_penalty: bool = False,
    n_repeat_tolerance: int = 5,
):
    count = 0  # 连续重复计数
    prev_line = None  # 上一行内容

    for L, line in enumerate(line_texts):
        if not line:
            continue
        line = remove_regix(line, ignore_regix)
        # 计算该行和上一行重复的字符个数
        if use_presence_penalty:
            n_repeat = sum(
                1
                for x in line_texts[:L]
                if partial_equal_line(x, line, min_n_repeat, n_repeat_tolerance)
            )
            is_repeat = n_repeat > 0
        else:
            is_repeat = partial_equal_line(
                line, prev_line, min_n_repeat, n_repeat_tolerance
            )
        if use_presence_penalty and is_repeat:
            rewards -= penalty_factor * n_repeat
        else:
            if is_repeat:
                count += 1  # 增加连续重复计数
                rewards -= penalty_factor * count  # 根据重复次数减少奖励值
            else:
                count = 0  # 重置连续重复计数
        prev_line = line  # 更新上一行内容

    # 确保奖励值不小于 0
    natural_speak_rwds = torch.max(
        rewards, torch.tensor([0.0], dtype=rewards.dtype, device=rewards.device)
    )
    return natural_speak_rwds


def repeat_penalty_cList(
    rewards: torch.Tensor,
    conversations: cList,
    penalty_factor_text: float = 0.05,
    penalty_factor_conversation: float = 0.25,
    min_n_repeat_text: int = -1,
    min_n_repeat_conversation: int = -1,
    bot_role: str = "response",
    ignore_regix_text: str = r"^\d+\.",
    use_presence_penalty_text: bool = False,
    n_repeat_tolerance_text: int = 5,
    n_repeat_tolerance_conversation: int = 5,
):
    count_conversation = 0  # 连续重复计数

    bot_conversations = [x for x in conversations if x.role == bot_role]
    last_conversation = None  # 上一行内容
    for conversation in bot_conversations:
        # 跨对话重复
        speak_text = conversation()
        if min_n_repeat_conversation > 0:
            is_repeat = partial_equal_line(
                speak_text,
                last_conversation(),
                min_n_repeat_conversation,
                n_repeat_tolerance_conversation,
            )
        else:
            is_repeat = speak_text == last_conversation  # 判断是否重复
        if is_repeat:
            count_conversation += 1  # 增加连续重复计数
            rewards -= (
                penalty_factor_conversation * count_conversation
            )  # 根据重复次数减少奖励值
        else:
            count_conversation = 0  # 重置连续重复计数

        # 对话内重复
        rewards = repeat_penalty(
            rewards,
            speak_text,
            penalty_factor_text,
            min_n_repeat_text,
            ignore_regix_text,
            use_presence_penalty=use_presence_penalty_text,
            n_repeat_tolerance=n_repeat_tolerance_text,
        )

        last_conversation = conversation  # 更新上一行内容

    natural_speak_rwds = torch.max(
        rewards, torch.tensor([0.0], dtype=rewards.dtype, device=rewards.device)
    )
    return natural_speak_rwds
