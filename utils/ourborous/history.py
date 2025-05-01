from typing import List, Tuple
from utils.message_manager import cList
import os
import json
import shutil
import uuid


class OurborousAgentHistoryContainer:
    def __init__(self, save_state_dir: str):
        self.save_state_dir = save_state_dir
        os.makedirs(save_state_dir, exist_ok=True)

        self.untrained_histories = (
            []
        )  # List[List[(cList, float)]]: turns[regenerates[conversations, reawrds]]

        self.trained_histories = (
            []
        )  # List[List[(cList, float)]]: turns[regenerates[conversations, reawrds]]

        self.state_after_train_index = None
        self.now_state_index = None
        self.last_state_index = None

    def clear_all_states(self):
        """
        清空缓存
        """
        self.state_after_train_index = None
        self.now_state_index = None
        self.last_state_index = None
        for file in os.listdir(self.save_state_dir):
            os.remove(os.path.join(self.save_state_dir, file))

    def prob_collapse(self, index: int):
        """
        选择某一生成作为确定历史
        """
        item = self.untrained_histories[-1].pop(index)
        self.untrained_histories[-1].append(item)

    def add_turns(self, turns: List[List[cList]]):
        """
        添加一轮生成的消息
        """
        self.untrained_histories += turns

    def add_turn_messages(self, turn_generates: List[cList]):
        """
        添加一轮生成的消息
        """
        self.untrained_histories.append(turn_generates)
        index_range = (0, len(turn_generates))
        return index_range

    def supplement_turn_messages(
        self, turn_generates: List[cList], turn_index: int = -1
    ):
        """
        在某一轮生成的消息中补充消息
        """
        self.untrained_histories[turn_index] += turn_generates
        index_range = (
            len(self.untrained_histories[turn_index]) - len(turn_generates),
            len(self.untrained_histories[turn_index]),
        )
        return index_range

    def back_to_last(self):
        """
        回退到上一轮生成的消息
        """
        self.untrained_histories.pop(-1)

    def on_train(self):
        """
        训练结束，将未训练的历史转移到已训练的历史中
        """
        self.trained_histories += self.untrained_histories
        self.untrained_histories = []

    def to_sft_dataset(
        self,
        save=True,
        save_path: str = "",
        save_version: str = "v1",
        select="untrained",
    ):
        """
        保存历史到sft数据集
        """
        assert select in ["untrained", "trained", "full"]
        select_history = (
            self.trained_histories + self.untrained_histories
            if select == "full"
            else (
                self.trained_histories
                if select == "trained"
                else self.untrained_histories
            )
        )
        chosen_conversations = [turn[-1] for turn in select_history]
        conversations = sum(chosen_conversations[1:], chosen_conversations[0])
        conversations: cList
        if save:
            if save_version == "v1":
                dataset,txt = conversations.to_dataset()
                with open(save_path, "w", encoding="utf-8") as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif save_version == "v2":
                dataset = conversations.to_dict_list()
                with open(save_path, "w", encoding="utf-8") as f:
                    for item in dataset:
                        line = {"data_dict_list": item}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")

        return conversations

    @staticmethod
    def serialize_history(history: List[List[cList]]):
        """
        序列化历史
        """
        new_history = []
        for turn in history:
            new_turns = []
            for conversations in turn:
                conversations: cList
                new_turns.append((conversations.to_dict_list()))
            new_history.append(new_turns)
        return new_history

    @staticmethod
    def dict_list_to_history(data_list: List[List[List[dict]]]):
        """
        字典列表转历史
        """
        history = []
        for i, turn in enumerate(data_list):
            history.append([])
            for j, conversations in enumerate(turn):
                history[i].append((cList.from_dicts(conversations)))
        return history


    def savepoint(self, save_folder_path: str):
        """
        存档
        """
        os.makedirs(save_folder_path, exist_ok=True)
        data_fdir = os.path.join(save_folder_path, "data.json")
        save_data = {
            "save_state_dir": self.save_state_dir,
            "untrained_histories": OurborousAgentHistoryContainer.serialize_history(
                self.untrained_histories
            ),
            "trained_histories": OurborousAgentHistoryContainer.serialize_history(
                self.trained_histories
            ),
        }
        with open(data_fdir, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False,indent=4)

        state_after_train_fdir = os.path.join(
            save_folder_path, "state_after_train.state"
        )
        now_state_fdir = os.path.join(save_folder_path, "now_state.state")
        last_state_fdir = os.path.join(save_folder_path, "last_state.state")
        self.save_state(self.state_after_train_index, state_after_train_fdir)
        self.save_state(self.now_state_index, now_state_fdir)
        self.save_state(self.last_state_index, last_state_fdir)

    @staticmethod
    def load_from_savepoint(save_folder_path: str):
        """
        从存档中加载
        """
        data_fdir = os.path.join(save_folder_path, "data.json")
        with open(data_fdir, "r", encoding="utf-8") as f:
            save_data = json.load(f)
        save_state_dir = save_data["save_state_dir"]
        agent = OurborousAgentHistoryContainer(save_state_dir)
        agent.untrained_histories = OurborousAgentHistoryContainer.dict_list_to_history(
            save_data["untrained_histories"]
        )
        agent.trained_histories = OurborousAgentHistoryContainer.dict_list_to_history(
            save_data["trained_histories"]
        )

        state_after_train_fdir = os.path.join(
            save_folder_path, "state_after_train.state"
        )
        now_state_fdir = os.path.join(save_folder_path, "now_state.state")
        last_state_fdir = os.path.join(save_folder_path, "last_state.state")
        if os.path.exists(state_after_train_fdir):
            new_name = "state_after_train"
            shutil.copy(state_after_train_fdir, f"{save_state_dir}/{new_name}.state")
            agent.state_after_train_index = new_name
        if os.path.exists(now_state_fdir):
            new_name = "now_state"
            shutil.copy(now_state_fdir, f"{save_state_dir}/{new_name}.state")
            agent.now_state_index = new_name
        if os.path.exists(last_state_fdir):
            new_name = "last_state"
            shutil.copy(last_state_fdir, f"{save_state_dir}/{new_name}.state")
            agent.last_state_index = new_name
        return agent

    def save_state(self, state_idx: str, to_dir: str):
        from_dir = f"{self.save_state_dir}/{state_idx}.state"
        try:
            shutil.copy(from_dir, to_dir)
            return to_dir
        except Exception as e:
            print(f"Warning: {from_dir} not found.\n{e}")
            return None

    def reset(self):
        """
        重置历史
        """
        self.untrained_histories = []
        self.trained_histories = []

        return self.untrained_histories, self.trained_histories

    def restart(self):
        """
        从训练点重新开始
        """
        self.untrained_histories = []
        self.now_state_index = None
        self.last_state_index = None

        return self.untrained_histories, self.trained_histories

    def untrained_ctx_len(self, encoding_func: callable):
        """
        计算未训练的历史长度
        """
        history = [turn[-1] for turn in self.untrained_histories]
        conversations = cList([turn for turn in history])
        return conversations.calc_ctx_len(encoding_func)
