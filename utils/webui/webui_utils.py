import numpy as np
import matplotlib.pyplot as plt
import os
import json
import gradio as gr



def loss_curve_plot(data, caption: str = "Loss Curve"):
    plt.close()
    # 如果数据为空，返回一个空图
    if not data:
        return plt.figure()

    fig, ax = plt.subplots(facecolor="#2F2F2F")
    ax.set_facecolor("#2F2F2F")

    # 绘制损失曲线
    ax.plot(data, label="Loss Curve", color="#00FF7F", linewidth=2)  # 更亮的绿色

    # 添加标题和标签
    ax.set_title(caption, color="white")
    ax.set_xlabel("Steps", color="white")
    ax.set_ylabel("Loss", color="white")

    ax.spines["top"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # 添加图例
    legend = ax.legend()
    legend.get_frame().set_facecolor("#2F2F2F")  # 图例背景设为深色
    legend.get_frame().set_edgecolor("white")  # 图例边框设为白色
    for text in legend.get_texts():
        text.set_color("white")  # 图例文字设为白色
    # 调整图形边距
    plt.tight_layout()

    # 返回图形对象
    return fig


# 定义偏好键的列表
PREFERENCE_KEYS = [
    "sender_name",
    "replier_name",
    "temp",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "decay_penalty",
    "usr_sp_token",
    "bot_sp_token",
    "bmk_temp",
    "bmk_top_p",
    "bmk_presence_penalty",
    "bmk_frequency_penalty",
    "bmk_penalty_decay",
    "bmk_use_init_state",
    "bmk_use_init_state_dir",
    "api_base",
    "api_key",
    "api_model",
    "api_sender_name",
    "api_bot_name",
    "api_temp",
    "api_top_p",
    "api_presence_penalty",
    "api_frequency_penalty",
    "ollr_train_data_folder_dir",
    "ollr_train_init_state",
    "ollr_train_load_init_state_dir",
    "ollr_train_epoch",
    "ollr_train_batch_size",
    "ollr_train_n_save_ckpt",
    "ollr_train_ctx_len",
    "ollr_train_multi_scale_alpha",
    "ollr_train_keep_states_mode",
    "fllr_dataset_list",
    "fllr_train_epoch",
    "fllr_train_batch_size",
    "fllr_train_n_save_ckpt",
    "fllr_train_save_ckpt_step",
    "fllr_train_n_step_save_ckpt",
    "gsm8k_parquet_file_path_input",
    "gsm8k_n_rollout_questions_input",
    "gsm8k_req_sp_token",
    "gsm8k_req_prefix",
    "gsm8k_resp_sp_token",
    "gsm8k_resp_prefix",
    "gsm8k_n_save_ckpt",
    "gsm8k_max_resp_ctx_len",
    "gsm8k_tiny_batch_size",
    "gsm8k_num_rollouts",
    "gsm8k_train_batch_size",
    "gsm8k_n_save_episode",
    "gsm8k_temperature",
    "gsm8k_top_p",
    "gsm8k_presence_penalty",
    "gsm8k_frequency_penalty",
    "gsm8k_penalty_decay",
    "grpo_ourborous_dataset",
    "grpo_ourborous_n_save_episode_ckpt",
    "grpo_ourborous_train_batch_size",
    "grpo_dpo_pair_dataset",
    "grpo_dpo_pair_n_samples_episode",
    "grpo_dpo_pair_n_episodes",
    "grpo_dpo_pair_system_sp_token",
    "grpo_dpo_pair_system_prefix",
    "grpo_dpo_pair_req_sp_token",
    "grpo_dpo_pair_req_prefix",
    "grpo_dpo_pair_resp_sp_token",
    "grpo_dpo_pair_resp_prefix",
    "grpo_dpo_pair_n_save_episode_ckpt",
    "grpo_dpo_pair_train_batch_size",
    "grpo_lr_init",
    "grpo_lr_final",
    "grpo_lr_warmup",
    "grpo_accumulate_grad",
]


def save_user_preference(save_path, *args):
    """保存用户偏好到指定路径

    Args:
        save_path: 保存路径
        *args: 按顺序包含所有需要保存的偏好值
    """
    # 尝试加载已有的偏好文件
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            preferences = json.load(f)
    except FileNotFoundError:
        preferences = {}

    # 更新偏好字典
    for key, value in zip(PREFERENCE_KEYS, args):
        if value is not None and not isinstance(value, dict):  # 跳过gr.update()返回的值
            preferences[key] = value

    # 保存到文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(preferences, f, ensure_ascii=False, indent=4)


def load_user_preference(load_path, agent):
    """从指定路径加载用户偏好"""
    try:
        with open(load_path, "r") as f:
            preferences = json.load(f)

        # 提取常用的偏好值
        common_preferences = [preferences.get(key) for key in PREFERENCE_KEYS[:9]]

        # 更新聊天机器人参数
        agent.update_chatbot_params(*common_preferences)

        # 返回所有偏好值，未找到的偏好值使用gr.update()代替
        return [preferences.get(key, gr.update()) for key in PREFERENCE_KEYS] + [
            "已加载历史填写。"
        ]
    except FileNotFoundError:
        return [gr.update()] * len(PREFERENCE_KEYS) + ["无历史操作记录"]
    except Exception as e:
        return [gr.update()] * len(PREFERENCE_KEYS) + [f"加载失败: {str(e)}"]
