import copy
import os
import torch
from .message_manager import Conversation, cList
import json
from collections import OrderedDict
import shutil


class LocalMemoryStatePool:
    def __init__(self, pool_size=16, cache_dir="") -> None:
        self.pool_size = pool_size
        self.states = OrderedDict()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def __getitem__(self, key):
        if key in self.states:
            state_dir = self.states[key]
            state = torch.load(state_dir)
        elif os.path.exists(os.path.join(self.cache_dir, f"{key}.state")):
            self.states[key] = os.path.join(self.cache_dir, f"{key}.state")
            state = torch.load(self.states[key])
        else:
            state = None
        return state

    def get(self, key, device):
        if key in self.states:
            state_dir = self.states[key]
            state = torch.load(state_dir, map_location=device)
        elif os.path.exists(os.path.join(self.cache_dir, f"{key}.state")):
            self.states[key] = os.path.join(self.cache_dir, f"{key}.state")
            state = torch.load(self.states[key], map_location=device)
        else:
            state = None
        return state

    def __setitem__(self, key, value):
        if value is None:
            return
        state_dir = os.path.join(self.cache_dir, f"{key}.state")
        torch.save(value, state_dir)
        self.states[key] = state_dir
        if len(self.states) > self.pool_size:
            _, state_dir = self.states.popitem(last=False)
            if os.path.exists(state_dir):
                os.remove(state_dir)

    def __contains__(self, key):
        return key in self.states

    def clear(self):
        for key in self.states:
            if os.path.exists(self.states[key]):
                os.remove(self.states[key])
        self.states.clear()

    def save(self, key, dir):
        if key in self.states:
            state_dir = self.states[key]
        elif os.path.exists(os.path.join(self.cache_dir, f"{key}.state")):
            state_dir = os.path.join(self.cache_dir, f"{key}.state")
        else:
            return  # 如果状态不存在，直接返回
        shutil.copy(state_dir, os.path.join(dir))

    def __delitem__(self, key):
        if key in self.states:
            state_dir = self.states[key]
            if os.path.exists(state_dir):
                os.remove(state_dir)
            del self.states[key]

    def keys(self):
        return self.states.keys()

    def values(self):
        return [self.states[key] for key in self.states]

    def items(self):
        return self.states.items()


class ChatbotMessagesContainer:
    def __init__(self) -> None:
        self.conversations_history = cList()  # 永久记录
        self.conversations_to_train = cList()  # 只记录上次Train后的部分
        self.last_conversation = None

        self.acting = False
        self.act_conversations = cList()
        self.act_display_conversations = cList()

        self.last_token = None  # 尚未推理的token
        self.last_token_last_conversation = None

    @staticmethod
    def index_of(clist: cList, conversation: Conversation):
        try:
            return clist.index(conversation)
        except ValueError:
            return 0

    @property
    def last_idx_history(self):
        return self.index_of(self.conversations_history, self.last_conversation)

    @property
    def last_idx_train(self):
        return self.index_of(self.conversations_to_train, self.last_conversation)

    def back_to_last(self):
        if self.last_conversation:
            self.conversations_history = self.conversations_history[
                : self.last_idx_history
            ]
            self.conversations_to_train = self.conversations_to_train[
                : self.last_idx_train
            ]
        else:
            self.restart()
        self.last_token = (
            self.last_token_last_conversation
            if self.last_token_last_conversation
            else None
        )
        return self.last_conversation

    def add_message(self, as_last: bool, conversation: Conversation, last_token=11):
        self.conversations_history.append(conversation)
        self.conversations_to_train.append(conversation)
        if as_last:
            self.last_conversation = conversation
            self.last_token_last_conversation = self.last_token
        self.last_token = last_token

    def add_messages(self, as_last: bool, conversations: cList, last_token=11):
        self.conversations_history += conversations
        self.conversations_to_train += conversations
        if as_last:
            self.last_conversation = conversations[0] if conversations else None
            self.last_token_last_conversation = self.last_token
        self.last_token = last_token

    def insert_message(self, conversation: Conversation, index, which="history"):
        """
        插入一句对话
        """
        assert which in ["train", "history"]
        assert index <= len(self.conversations_to_train)
        if which == "history":
            self.conversations_history.insert(index, conversation)
        elif which == "train":
            self.conversations_to_train.insert(index, conversation)

    def delete_message(self, index, which="history"):
        """
        删除一句对话
        """
        assert which in ["train", "history"]
        assert index < len(self.conversations_to_train)
        if which == "history":
            assert index < len(self.conversations_history)
            self.conversations_to_load.pop(index)
        elif which == "train":
            assert index < len(self.conversations_to_train)
            self.conversations_to_train.pop(index)

    def edit_message(self, conversation: Conversation, index, which="history"):
        """
        编辑一句对话
        """
        assert which in ["train", "load"]
        if which == "load":
            assert index < len(self.conversations_to_load)
            self.conversations_to_load[index] = conversation
        elif which == "train":
            assert index < len(self.conversations_to_train)
            self.conversations_to_train[index] = conversation

    def overwrite_messages(
        self,
        conversations: cList,
        which="history",
        last_conversation: Conversation = None,
    ):
        assert which in ["train", "load"]
        if which == "load":
            self.conversations_to_load = conversations
        elif which == "train":
            self.conversations_to_train = conversations
        if last_conversation:
            self.last_conversation = last_conversation

    def restart(self):
        for conversation in self.conversations_to_train:
            try:
                self.conversations_history.remove(conversation)
            except:
                continue
        self.conversations_to_train.clear()
        self.acting = False
        self.act_conversations.clear()
        self.act_display_conversations.clear()
        self.online_lr_state = None
        self.last_token = None
        self.last_token_last_conversation = None

    def on_train(self):
        self.conversations_to_train.clear()

    def end_acting_dataproc(self):
        self.conversations_history += self.act_display_conversations
        self.conversations_to_train += self.act_display_conversations
        self.acting = False

    def act_message(self, display: bool, conversation: Conversation, last_token=11):
        self.act_conversations.append(conversation)
        if display:
            self.act_display_conversations.append(conversation)
        self.last_token = last_token

    def start_acting(self):
        self.acting = True
        self.act_conversations.clear()
        self.act_display_conversations.clear()

    def save_as_checkpoint(self, dir, name):
        folder = os.path.join(dir, name)
        os.makedirs(folder, exist_ok=True)
        with open(
            os.path.join(folder, "conversations_train.json"), "w", encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    self.conversations_to_train.to_dict_list(), ensure_ascii=False
                )
            )
        with open(
            os.path.join(folder, "conversations_hist.json"), "w", encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    self.conversations_history.to_dict_list(), ensure_ascii=False
                )
            )

    def load_checkpoint(self, ckpt_folder):
        if os.path.isdir(ckpt_folder):
            # 加载训练对话记录
            train_file = os.path.join(ckpt_folder, "conversations_train.json")
            hist_file = os.path.join(ckpt_folder, "conversations_hist.json")
            if os.path.exists(train_file):
                with open(train_file, "r", encoding="utf-8") as f:
                    train_data = json.load(f)
                    self.conversations_to_train = cList.from_dicts(train_data)
                print(f"load hist to train from: {train_file}.")
            # 加载历史对话记录
            if os.path.exists(hist_file):
                with open(hist_file, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
                    self.conversations_history = cList.from_dicts(hist_data)
                print(f"load hist conversation from: {hist_file}.")

    def save_as_dataset(self, fp):
        conversations = self.conversations_history
        dic, txt = conversations.to_dataset()
        with open(fp, "w", encoding="utf-8") as f:
            f.write(json.dumps(dic, ensure_ascii=False))
