from config import global_config
import time
import json
import os
import torch
import gc
import requests
from utils.inference_chatbot import Chatbot
from utils.containers import Conversation, cList
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def update_plot(data):
    if not data:
        return None
    # 处理数据，确保只显示最后100个点
    if len(data) > 100:
        x = np.arange(len(data) - 100, len(data))
        y = np.array(data[-100:])
    else:
        x = np.arange(len(data))
        y = np.array(data)

    # 创建平滑渐变的颜色映射
    colors = [(0, 0.8, 0), (0.8, 0.8, 0), (0.8, 0, 0)]  # 绿色  # 黄色  # 红色
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    # 构造分段线条
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 100))
    lc.set_array(y)
    lc.set_linewidth(2)  # 增加线条宽度

    # 创建图形并设置深灰色背景
    fig, ax = plt.subplots(facecolor="#2F2F2F")
    ax.set_facecolor("#2F2F2F")

    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 120)

    # 去掉所有标识
    ax.axis("off")

    # 调整图形边距
    plt.tight_layout()

    return fig


class InferAgent:
    def __init__(
        self,
        infer_server: str = "http://0.0.0.0:4514",
        init_sender: str = "user",
        init_replier: str = "assistant",
    ):
        gc.collect()
        torch.cuda.empty_cache()
        self.infer_server = infer_server
        self.sender = init_sender
        self.replier = init_replier
        self.chatbot = Chatbot(infer_server)
        self.time_on = False

        # 以下用于记录
        self.accumulate_logs = {}
        self.last_conversations = None
        self.turn_logs = cList()
        self.turn = 1
        self.last_turn = self.turn

    def calc_token_ban(self, input_str):
        if not input_str.strip():
            return []
        try:
            # 将字符串按逗号分隔，并转换为整数
            int_list = [int(i) for i in input_str.split(",")]
            return int_list
        except ValueError as e:
            raise e

    def send_message(
        self,
        message,
        max_resp_len=512,
        format=False,
        format_str=None,
        token_ban_str="",
        auto_think=False,
        force_think=False,
    ):
        """处理用户输入的消息，并返回AI的回复"""
        token_ban = self.calc_token_ban(token_ban_str)
        text = message.strip()
        if text:

            self.last_turn = self.turn
            self.turn += 1
            replier_prefix = f"{self.replier}: "
            current_time = time.localtime()
            time_string = time.strftime(r"(%Y-%m-%d %H:%M:%S)", current_time)

            sender_text = (
                f"{self.sender}: {text}"
                if not self.time_on
                else f"{self.sender}{time_string}: {text}"
            )
            send_conversations = cList(
                [Conversation(self.chatbot.usr_sp_token_role, sender_text)]
            )

            api_protocol_conversations = [
                {
                    "role": (
                        "assistant"
                        if conversation.role in global_config.ego_types
                        else "user"
                    ),
                    "content": conversation.content,
                    "metadata": (
                        {"title": "思考过程", "id": 0, "status": "pending"}
                        if conversation.role == "think"
                        else {}
                    ),
                }
                for conversation in (
                    self.chatbot.history.conversations_history + send_conversations
                )
            ]

            # 打印用户的发言
            # chat_output = f"{history}\n{sender_text}\n{replier_prefix}"
            think_conversations = []
            already_think = False
            if auto_think or force_think:
                role_prefix_pairs = [
                    ("think", f"({replier_prefix}"),
                    ("think", f"{replier_prefix}"),
                ]
                hit = self.chatbot.estimate_desires(
                    role_prefix_pairs=role_prefix_pairs,
                    start_with_conversations=send_conversations,
                    ignore_tolerance=2,
                )
                if hit >= 0 or force_think:
                    already_think = True
                    print(f"\n触发思考：{hit}")
                    think_prefix = role_prefix_pairs[hit][1] if not force_think else role_prefix_pairs[0][1]
                    for (
                        next_text,
                        now_full_str,
                        in_ppls,
                        out_ppls,
                    ) in self.chatbot.stream_chat(
                        send_conversations,
                        replier_prefix,
                        rpy_role="think",
                        max_resp_len=max_resp_len,
                        format_constrain_str=None,
                        token_ban=token_ban,
                        need_ppl=True,
                    ):
                        think_conversations = [
                            {
                                "role": ("assistant"),
                                "content": f"{think_prefix}{now_full_str}",
                                "metadata": {
                                    "title": "思考过程",
                                    "id": 0,
                                    "status": "pending",
                                },
                            }
                        ]
                        yield (
                            api_protocol_conversations + think_conversations,
                            "",
                            update_plot(in_ppls),
                            update_plot(out_ppls),
                        )

            format = format_str if format else None
            # 使用stream_chat，逐步更新AI的回复
            for next_text, now_full_str, in_ppls, out_ppls in self.chatbot.stream_chat(
                send_conversations if not already_think else None,
                replier_prefix,
                max_resp_len=max_resp_len,
                format_constrain_str=format,
                token_ban=token_ban,
                need_ppl=True,
                mark_last_conversation=not already_think,
            ):
                now_api_conversations = [
                    {
                        "role": ("assistant"),
                        "content": f"{replier_prefix}{now_full_str}",
                    }
                ]
                print(api_protocol_conversations
                    + think_conversations
                    + now_api_conversations)
                yield (
                    api_protocol_conversations
                    + think_conversations
                    + now_api_conversations,
                    "",
                    update_plot(in_ppls),
                    update_plot(out_ppls),
                )

            # 更新对话历史
            # self.last_conversations = self.turn_logs + send_conversations.to_dict_list()
            self.last_conversations = self.turn_logs + send_conversations
            self.turn_logs += send_conversations + cList.from_single_conversation(
                Conversation(
                    self.chatbot.bot_sp_token_role, f"{replier_prefix}{now_full_str}"
                )
            )
            self.add_accumulate_log(
                self.turn, self.turn_logs.to_dict_list(), None, None
            )
            self.turn_logs = cList()

    def estimate_chat_desire(self, check_special_tokens=[]):
        hit = -1
        return hit

    def add_custom_message(self, role, content):
        """添加自定义消息"""
        if content:
            add_msg = cList.from_single_conversation(Conversation(role, content))
            self.chatbot.add_messages(add_msg)
            self.turn_logs += add_msg
            api_protocol_conversations = [
                {
                    "role": (
                        "assistant"
                        if conversation.role in global_config.ego_types
                        else "user"
                    ),
                    "content": conversation.content,
                    "metadata": (
                        {"title": "思考过程", "id": 0, "status": "pending"}
                        if conversation.role == "think"
                        else {}
                    ),
                }
                for conversation in (self.chatbot.history.conversations_history)
            ]
            return api_protocol_conversations
        return "无效的消息内容。"

    # def get_history(self):
    #     """获取历史消息"""
    #     history = self.chatbot.history.conversations_history()
    #     return history

    def regenerate(
        self, max_resp_len=512, format=False, format_str=None, token_ban_str=""
    ):
        """重新生成上次的回复"""
        token_ban = self.calc_token_ban(token_ban_str)
        replier_prefix = f"{self.replier}: "
        format = format_str if format else None

        api_protocol_conversations = [
            {
                "role": (
                    "assistant"
                    if conversation.role in global_config.ego_types
                    else "user"
                ),
                "content": conversation.content,
                "metadata": (
                    {"title": "思考过程", "id": 0, "status": "pending"}
                    if conversation.role == "think"
                    else {}
                ),
            }
            for conversation in (
                self.chatbot.history.conversations_history[
                    : self.chatbot.history.last_idx_history
                ]
                + [self.last_conversations[0]]
            )
        ]

        for (
            next_text,
            now_full_str,
            in_ppls,
            out_ppls,
        ) in self.chatbot.stream_regenerate(
            replier_prefix,
            max_resp_len=max_resp_len,
            format_str=format,
            token_ban=token_ban,
            need_ppl=True,
        ):
            now_api_conversations = [
                {
                    "role": ("assistant"),
                    "content": f"{replier_prefix}{now_full_str}",
                }
            ]
            yield api_protocol_conversations + now_api_conversations, update_plot(
                in_ppls
            ), update_plot(out_ppls)

        # 更新对话历史

        self.turn_logs += cList(
            [
                self.last_conversations[0],
                Conversation(
                    self.chatbot.bot_sp_token_role, f"{replier_prefix}{now_full_str}"
                ),
            ]
        )
        self.add_accumulate_log(self.turn, self.turn_logs.to_dict_list(), None, None)
        self.turn_logs = cList()

    def reset(self):
        """重置会话"""
        self.chatbot.reboot()
        self.accumulate_logs = {}
        self.last_conversations = None
        self.turn_logs = cList()
        self.turn = 1
        self.last_turn = self.turn
        return "会话已重置。", []

    def listen(self, max_resp_len=512, format=False, format_str=None, token_ban_str=""):
        """等待AI回复"""
        token_ban = self.calc_token_ban(token_ban_str)
        replier_prefix = f"{self.replier}: "
        format = format_str if format else None
        api_protocol_conversations = [
            {
                "role": (
                    "assistant"
                    if conversation.role in global_config.ego_types
                    else "user"
                ),
                "content": conversation.content,
                "metadata": (
                    {"title": "思考过程", "id": 0, "status": "pending"}
                    if conversation.role == "think"
                    else {}
                ),
            }
            for conversation in (self.chatbot.history.conversations_history)
        ]
        for next_text, now_full_str, in_ppls, out_ppls in self.chatbot.stream_chat(
            None,
            replier_prefix,
            max_resp_len=max_resp_len,
            format_constrain_str=format,
            token_ban=token_ban,
            need_ppl=True,
        ):
            now_api_conversations = [
                {
                    "role": ("assistant"),
                    "content": f"{replier_prefix}{now_full_str}",
                }
            ]
            yield api_protocol_conversations + now_api_conversations, update_plot(
                in_ppls
            ), update_plot(out_ppls)

        # 更新对话历史
        self.turn_logs += cList.from_single_conversation(
            Conversation(
                self.chatbot.bot_sp_token_role, f"{replier_prefix}{now_full_str}"
            )
        )
        return (
            f"{replier_prefix}{now_full_str}",
            update_plot(in_ppls),
            update_plot(out_ppls),
        )

    def think(self, max_resp_len=512):
        """思考"""
        replier_prefix = f"({self.replier}: "
        api_protocol_conversations = [
            {
                "role": (
                    "assistant"
                    if conversation.role in global_config.ego_types
                    else "user"
                ),
                "content": conversation.content,
                "metadata": (
                    {"title": "思考过程", "id": 0, "status": "pending"}
                    if conversation.role == "think"
                    else {}
                ),
            }
            for conversation in (self.chatbot.history.conversations_history)
        ]
        for next_text, now_full_str, in_ppls, out_ppls in self.chatbot.stream_chat(
            None,
            replier_prefix,
            rpy_role="think",
            max_resp_len=max_resp_len,
            format_constrain_str=None,
            token_ban=[],
            need_ppl=True,
        ):
            now_api_conversations = [
                {
                    "role": ("assistant"),
                    "content": f"{replier_prefix}{now_full_str}",
                    "metadata": {"title": "思考过程", "id": 0, "status": "pending"},
                }
            ]
            yield api_protocol_conversations + now_api_conversations, update_plot(
                in_ppls
            ), update_plot(out_ppls)

    def back_to_last(self):
        """删除最近一轮对话"""
        self.chatbot.back_to_last()
        if self.turn > 1:
            self.turn_logs = cList()
            self.turn = self.last_turn
        api_protocol_conversations = [
            {
                "role": (
                    "assistant"
                    if conversation.role in global_config.ego_types
                    else "user"
                ),
                "content": conversation.content,
                "metadata": (
                    {"title": "思考过程", "id": 0, "status": "pending"}
                    if conversation.role == "think"
                    else {}
                ),
            }
            for conversation in (self.chatbot.history.conversations_history)
        ]
        return "已删除最近一轮对话。", api_protocol_conversations

    def save_as_dataset(self):
        """保存聊天记录为数据集"""
        current_time = time.localtime()
        time_string = time.strftime(r"%Y-%m-%d-%H-%M-%S", current_time)
        self.chatbot.save_as_dataset(global_config.save_dataset_dir, time_string)
        return "聊天记录已保存为数据集。"

    def save_as_rl_pairs(self):
        """保存生成历史为强化学习组"""
        current_time = time.localtime()
        time_string = time.strftime(r"%Y-%m-%d-%H-%M-%S", current_time)
        os.makedirs(global_config.save_rl_dataset_dir, exist_ok=True)
        jsonl = os.path.join(
            global_config.save_rl_dataset_dir, f"{time_string}_pairs.jsonl"
        )
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.accumulate_logs, ensure_ascii=False))
        return "生成历史已保存为强化学习组。"

    def load_infer_weight(self, load_dir):
        """加载模型权重"""
        requests.post(self.infer_server + "/load_weight", json={"load_dir": load_dir})
        return f"模型已加载：{load_dir}"

    def reboot_inference(self, load_state_dir=None):
        """重启推理并加载历史"""
        self.chatbot.reboot(load_state_dir)
        self.accumulate_logs = {}
        self.last_conversations = None
        self.turn_logs = cList()
        self.turn = 1
        self.last_turn = self.turn
        return f"历史已加载并重置：{load_state_dir}"

    def load_model(self, model_path):
        """加载模型的逻辑"""
        requests.post(self.infer_server + "/load_weight", json={"load_dir": model_path})
        self.chatbot.reboot()
        self.accumulate_logs = {}
        self.last_conversations = None
        self.turn_logs = cList()
        self.turn = 1
        self.last_turn = self.turn
        return f"模型已加载：{model_path}"

    def load_history_and_reset(self, history_path):
        """加载历史并重置的逻辑"""
        self.chatbot.reboot(history_path)
        self.accumulate_logs = {}
        self.last_conversations = None
        self.turn_logs = cList()
        self.turn = 1
        self.last_turn = self.turn
        return f"历史已加载并会话重置：{history_path}", ""

    def update_sender(self, sender_name):
        """更新用户名字"""
        self.sender = sender_name
        return f"用户名字已更新为: {sender_name}"

    def update_replier(self, replier_name):
        """更新AI名字"""
        self.replier = replier_name
        return f"AI名字已更新为: {replier_name}"

    def toggle_time(self, time_on):
        """切换时间戳"""
        self.time_on = time_on
        return f"时间戳已{'开启' if time_on else '关闭'}。"

    def add_accumulate_log(self, turn, choice, score, safety):
        """记录对话日志"""
        str_turn = f"{turn}"
        if str_turn not in self.accumulate_logs:
            self.accumulate_logs[str_turn] = [
                {"choice": choice, "score": score, "safety": safety, "is_best": False}
            ]
        else:
            self.accumulate_logs[str_turn] += [
                {"choice": choice, "score": score, "safety": safety, "is_best": False}
            ]

    def update_chatbot_params(
        self,
        sender,
        replier,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
        decay_penalty,
        usr_sp_token,
        bot_sp_token,
    ):
        """更新chatbot的参数和名字"""
        # 更新名字
        self.sender = sender
        self.replier = replier

        # 更新参数
        self.chatbot.temp = temp
        self.chatbot.top_p = top_p
        self.chatbot.presence_penalty = presence_penalty
        self.chatbot.frequency_penalty = frequency_penalty
        self.chatbot.decay_penalty = decay_penalty

        # 更新 special tokens
        self.chatbot.change_sp_tokens(usr_sp_token, bot_sp_token)

        return "名字和参数已更新。"

    def save_checkpoint(self, ckpt_folder):
        """保存当前会话的存档点"""
        current_time = time.localtime()
        ckpt_name = f"checkpoint_{time.strftime(r'%Y-%m-%d-%H-%M-%S', current_time)}"
        self.chatbot.save_ckpt(ckpt_folder, ckpt_name)
        return f"存档点已保存为：{ckpt_name}"

    def load_checkpoint(self, ckpt_dir):
        """加载存档点并重置会话"""
        self.chatbot.load_ckpt(ckpt_dir)
        self.accumulate_logs = {}
        self.last_conversations = None
        self.turn_logs = cList()
        self.turn = 1
        self.last_turn = self.turn
        return f"存档点已加载并会话重置：{ckpt_dir}"
