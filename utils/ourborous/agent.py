from utils.ourborous.history import OurborousAgentHistoryContainer
from utils.message_manager import cList, Conversation
from utils.ourborous.chain import Ourboroustools
from utils.ourborous.online_trainner import OurborousOnlineTrainner
import requests
import os
from config import global_config
import json
from typing import List
from utils.dataset.dataset import RLGroupDataset
from bottle import Bottle, request, run
import threading
import time
from functools import partial
import copy
import pickle

import threading


def test_server(server):
    try:
        response = requests.get(server + "/test")
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        # 任何网络异常都会捕获并返回 False
        print(f"Exception occurred: {e}")
        return False


class OurborousAgent:
    def __init__(
        self,
        inference_server: str = "http://0.0.0.0:4514",
        train_server: str = "http://0.0.0.0:3005",
        save_state_dir="/home/li/MachineLr/ramdisk",
        load_savepoint=None,
        auto_sft: bool = False,
        auto_sft_lr: float = 1e-5,
        auto_rl: bool = False,
        auto_rl_lr: float = 5e-6,
        auto_rl_bsz: int = 1,
        auto_rl_times: int = 1,
        accumulate_grad: bool = True,
        accumulate_grad_bsz: int = 1,
        auto_train_reward_model: bool = False,
    ):
        self.inference_server = inference_server
        self.train_server = train_server
        self.rw_model_server = f"http://{global_config.server_config.reward_model.host}:{global_config.server_config.reward_model.port}"
        self.reward_model_on = global_config.ourborous_config.reward_model_on
        self.history_container = OurborousAgentHistoryContainer(
            save_state_dir=save_state_dir,
        )
        self.tools = Ourboroustools()
        self.temp_tools = []
        if load_savepoint is not None:
            self.load_savepoint(load_savepoint)

        # train
        self.auto_train_on_ctx = 3072
        self.auto_sft = auto_sft
        self.auto_rl = auto_rl
        self.auto_train_reward_model = auto_train_reward_model
        self.trainner = OurborousOnlineTrainner(
            agent=self,
            train_server=train_server,
            auto_sft_lr=auto_sft_lr,
            auto_rl_lr=auto_rl_lr,
            auto_rl_bsz=auto_rl_bsz,
            auto_rl_times=auto_rl_times,
            accumulate_grad=accumulate_grad,
            accumulate_grad_bsz=accumulate_grad_bsz,
            reward_model_server=self.rw_model_server,
            save_rm_log_dir=f"{global_config.save_dataset_dir}/rm_logs",
        )
        self.sync_train_event = threading.Event()

        # inference
        self.lookat_idx = 0

        self.last_req_conversations = None
        self.last_rpy_role = None
        self.last_replier_name = None

        self.broadcast_sync = global_config.ourborous_config.broadcast_sync

    @property
    def untrained_ctx_len(self):
        return self.history_container.untrained_ctx_len(
            global_config.tokenizer_eval.encode
        )

    def restart(self):
        if not test_server(self.inference_server):
            return None
        if self.history_container.now_state_index is not None:
            requests.post(
                self.inference_server + "/remove_state_id",
                json={"access_token": f"{self.history_container.now_state_index}"},
            )
        if self.history_container.last_state_index is not None:
            requests.post(
                self.inference_server + "/remove_state_id",
                json={"access_token": f"{self.history_container.last_state_index}"},
            )
        if self.history_container.state_after_train_index is not None:
            state_dir = os.path.join(
                self.history_container.save_state_dir,
                f"{self.history_container.state_after_train_index}.state",
            )
            if os.path.exists(state_dir):
                package = {
                    "load_dir": state_dir,
                }
                self.history_container.now_state_index = requests.post(
                    self.inference_server + "/regist_state_id", json=package
                ).json()["access_token"]
                self.history_container.last_state_index = requests.post(
                    self.inference_server + "/regist_state_id", json=package
                )
        return self.history_container.restart()

    def reset(self):
        if not test_server(self.inference_server):
            return None
        self.lookat_idx = 0
        self.last_req_conversations = None
        self.last_rpy_role = None
        self.last_replier_name = None
        if self.history_container.now_state_index is not None:
            requests.post(
                self.inference_server + "/remove_state_id",
                json={"access_token": f"{self.history_container.now_state_index}"},
            )
        if self.history_container.last_state_index is not None:
            requests.post(
                self.inference_server + "/remove_state_id",
                json={"access_token": f"{self.history_container.last_state_index}"},
            )
        if self.history_container.state_after_train_index is not None:
            requests.post(
                self.inference_server + "/remove_state_id",
                json={
                    "access_token": f"{self.history_container.state_after_train_index}"
                },
            )
        return self.history_container.reset()

    def gather_hist_conversation(self, histories, last_idx=-1):
        if not histories:
            return cList()
        hists = [hist[-1] for hist in histories[:-1]] + [histories[-1][last_idx]]
        if hists:
            return sum(hists[1:], hists[0])
        else:
            return cList()

    def infer_history(
        self,
        infer_conversations: cList,
    ):
        package = {
            "conversations": (
                infer_conversations.to_dict_list() if infer_conversations else [11]
            ),
            "state_idx": self.history_container.now_state_index,
            "save_to_now_state_idx": self.history_container.now_state_index,
            "save_logits": False,
        }
        requests.post(
            self.inference_server + "/infer",
            json=package,
        ).json()

    def choose_conversation_as_hist(self, choose_idx):
        if self.history_container.untrained_histories:
            package = {
                "from_access_token": self.history_container.now_state_index,
                "to_access_token": self.history_container.last_state_index,
            }
            requests.post(self.inference_server + "/copy_state", json=package)
            self.history_container.prob_collapse(choose_idx)
            infer_hist = self.history_container.untrained_histories[-1][-1]
            self.infer_history(infer_hist)

    def ourborous_conversations_to_api_protocol_conversations(
        self, conversations: cList
    ):

        return [
            (
                {
                    "role": (
                        "assistant"
                        if conversation.role in global_config.ego_types
                        else "user"
                    ),
                    "content": (
                        conversation.content.replace(":", ":\n", 1)
                        if ":" in conversation.content.split("\n")[0]
                        else conversation.content
                    ),
                    "metadata": ({"title": "思考过程", "id": 0, "status": "pending"}),
                }
                if conversation.role == "think"
                else {
                    "role": (
                        "assistant"
                        if conversation.role in global_config.ego_types
                        else "user"
                    ),
                    "content": (
                        conversation.content.replace(":", ":\n", 1)
                        if ":" in conversation.content.split("\n")[0]
                        else conversation.content
                    ),
                }
            )
            for conversation in conversations
        ]

    def back_to_last(self, back_history: bool = True):
        if (
            self.history_container.untrained_histories
            and self.last_req_conversations is not None
        ):
            if back_history:
                self.history_container.back_to_last()
            last_req_conversations = self.last_req_conversations
            self.last_req_conversations = None
            package = {
                "from_access_token": self.history_container.last_state_index,
                "to_access_token": self.history_container.now_state_index,
            }
            requests.post(self.inference_server + "/copy_state", json=package)
            return last_req_conversations
        else:
            return None

    @staticmethod
    def begin_tool_chain(
        agent,
        batch_resp_conversations: List[cList],
        batch_full_conversations: List[cList],
        **kwargs,
    ):
        B = len(batch_resp_conversations)
        lock = threading.Lock()
        batch_return_full_conversations = [cList() for _ in range(B)]
        batch_events = [threading.Event() for _ in range(B)]

        tharads = []

        def batch_end():
            for thread in tharads:
                if thread.is_alive():
                    return False
            return True

        def inside_callback(i, full_conversations):
            with lock:
                batch_return_full_conversations[i] = full_conversations
                batch_events[i].set()

        def chain_begin(i):
            for return_full_conversations in agent.temp_tools[i].entry.chain(
                agent,
                batch_resp_conversations[i][-1],
                batch_full_conversations[i],
                **kwargs,
                callback=partial(inside_callback, i),
            ):
                pass

        for i in range(B):
            t = threading.Thread(target=chain_begin, args=(i,))
            tharads.append(t)
            t.start()

        while True:
            time.sleep(0.02)
            for i in range(B):
                if batch_events[i].is_set():
                    new_turn_msgs = [
                        conversations for conversations in batch_full_conversations
                    ]

                    yield new_turn_msgs
                    batch_events[i].clear()
            if batch_end():
                break

    def lookat_pre(self):
        self.lookat_idx = self.lookat_idx - 1 if self.lookat_idx > 0 else 0
        api_protocol_conversations = (
            self.ourborous_conversations_to_api_protocol_conversations(
                self.gather_hist_conversation(
                    self.history_container.trained_histories
                    + self.history_container.untrained_histories,
                    last_idx=self.lookat_idx,
                )
            )
        )
        n_resps = (
            len(self.history_container.untrained_histories[-1])
            if self.history_container.untrained_histories
            else 0
        )
        lookat_str = f"第{self.lookat_idx+1}个回复/共{n_resps}个回复/(ctx {self.untrained_ctx_len})"
        return api_protocol_conversations, lookat_str

    def lookat_next(self):
        lookat_range = len(self.history_container.untrained_histories[-1])
        self.lookat_idx = (
            self.lookat_idx + 1
            if self.lookat_idx < lookat_range - 1
            else lookat_range - 1
        )
        api_protocol_conversations = (
            self.ourborous_conversations_to_api_protocol_conversations(
                self.gather_hist_conversation(
                    self.history_container.trained_histories
                    + self.history_container.untrained_histories,
                    last_idx=self.lookat_idx,
                )
            )
        )
        n_resps = (
            len(self.history_container.untrained_histories[-1])
            if self.history_container.untrained_histories
            else 0
        )
        lookat_str = f"第{self.lookat_idx+1}个回复/共{n_resps}个回复/(ctx {self.untrained_ctx_len})"
        return api_protocol_conversations, lookat_str

    def refresh_history(self):
        return self.ourborous_conversations_to_api_protocol_conversations(
            self.gather_hist_conversation(
                self.history_container.trained_histories
                + self.history_container.untrained_histories[:-1]
            )
            + self.history_container.untrained_histories[-1][self.lookat_idx]
        )

    def chat(
        self,
        send_conversations: cList,
        rpy_role: str = "response",
        replier_name: str = "assistant",
        num_rollouts: int = 1,
        temperature: float = 1,
        top_p: float = 0.8,
        alpha_frequency: float = 0.2,
        alpha_presence: float = 0.2,
        alpha_decay: float = 0.9961,
        max_resp_len: float = 2048,
        format_constraint_str: float = None,
        token_ban: List[int] = [0],
        allow_think: bool = False,
        force_think: bool = False,
        regenerate: bool = False,
    ):
        if not regenerate and self.temp_tools:
            self.tools = copy.deepcopy(self.temp_tools[self.lookat_idx])
        if self.history_container.now_state_index is None:
            self.history_container.now_state_index = requests.post(
                self.inference_server + "/regist_state_id", json={}
            ).json()["access_token"]
        self.choose_conversation_as_hist(self.lookat_idx if not regenerate else -1)

        # # 自动训练(后续改为并行训练，训练完自动加载模型，更新当前历史)
        if (
            self.history_container.untrained_ctx_len(
                global_config.tokenizer_eval.encode
            )
            > self.auto_train_on_ctx
        ):
            if len(self.history_container.untrained_histories) > 1:

                if self.auto_train_reward_model:
                    thread = threading.Thread(
                        target=self.auto_train_reward_model_check_and_begin,
                    )
                    thread.start()
                if self.auto_sft:
                    thread = threading.Thread(target=self.auto_sft_check_and_begin)
                    thread.start()
                if self.auto_rl:
                    thread = threading.Thread(
                        target=self.auto_rl_check_and_begin,
                    )
                    thread.start()
                self.save_data_hist(
                    "untrained", save_folder_dir=global_config.save_dataset_dir
                )

                self.history_container.trained_histories += (
                    self.history_container.untrained_histories[:-1]
                )
                self.history_container.untrained_histories = [
                    self.history_container.untrained_histories[-1],
                ]
                if self.sync_train_event.is_set():
                    print("sync train model...")

                    if self.broadcast_sync:
                        send_conversations = (
                            cList.from_dicts(
                                Conversation(
                                    role="system", content="同步在线训练模型。"
                                )
                            )
                            + send_conversations
                        )
                        api_protocol_conversations = (
                            self.ourborous_conversations_to_api_protocol_conversations(
                                self.gather_hist_conversation(
                                    self.history_container.trained_histories
                                    + self.history_container.untrained_histories
                                )
                                + send_conversations
                            )
                        )
                    else:
                        api_protocol_conversations = (
                            self.ourborous_conversations_to_api_protocol_conversations(
                                self.gather_hist_conversation(
                                    self.history_container.trained_histories
                                    + self.history_container.untrained_histories
                                )
                                + cList.from_dicts(
                                    Conversation(
                                        role="system", content="同步在线训练模型。"
                                    )
                                )
                                + send_conversations
                            )
                        )
                    yield api_protocol_conversations

                    self.sync_train_model()
                    self.sync_train_event.clear()
                print("saving dataset...")
                print("saving dataset done...")
                print("saving savepoint...")
                time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                self.savepoint(f"{global_config.save_dataset_dir}/{time_str}")
                print("saving savepoint done...")
            else:
                print(
                    "Warning: 构成ctx_len的生成数不够，本轮不同步模型与训练，请检查最大生成长度，以保证稳定性。"
                )

        self.infer_history(send_conversations)
        self.lookat_idx = 0
        api_protocol_conversations = (
            self.ourborous_conversations_to_api_protocol_conversations(
                self.gather_hist_conversation(
                    self.history_container.trained_histories
                    + self.history_container.untrained_histories
                )
                + send_conversations
            )
        )
        yield api_protocol_conversations

        self.last_req_conversations = send_conversations
        self.last_rpy_role = rpy_role
        self.last_replier_name = replier_name
        replier_prefix = f"{replier_name}:" if replier_name else ""

        turn_msgs = [cList()] * num_rollouts
        if not regenerate:
            idx_range = self.history_container.add_turn_messages(turn_msgs)
        else:
            idx_range = self.history_container.supplement_turn_messages(turn_msgs)

        hit = -1
        if allow_think and not force_think:
            hit = self.estimate_desires(
                role_prefix_pairs=[
                    ("think", replier_prefix),
                ],
                start_with_conversations=send_conversations,
            )
        already_think = False
        if force_think or hit >= 0:
            already_think = True
            for think_full_text_batch in self.generate(
                send_conversations=send_conversations,
                rpy_role="think",
                replier_name=f"({replier_name}",
                num_rollouts=num_rollouts,
                temperature=temperature,
                top_p=top_p,
                alpha_frequency=alpha_frequency,
                alpha_presence=alpha_presence,
                alpha_decay=alpha_decay,
                max_resp_len=max_resp_len,
                format_constraint_str=None,
                token_ban=token_ban,
                use_now_state_idx=self.history_container.now_state_index,
                save_to_now_state_idx=None,
            ):
                out_think_full_text_batch = [
                    f"({replier_prefix}" + out_full_text
                    for out_full_text in think_full_text_batch
                ]

                # 更新当前历史
                think_conversation_batch = (
                    [
                        cList.from_single_conversation(
                            Conversation(role="think", content=out_full_text)
                        )
                        for out_full_text in out_think_full_text_batch
                    ]
                    if already_think
                    else []
                )
                full_conversations_batch = [
                    cList(send_conversations + think_conversation_batch[i])
                    for i in range(num_rollouts)
                ]

                for i, idx in enumerate(range(idx_range[0], idx_range[1])):
                    self.history_container.untrained_histories[-1][idx] = (
                        full_conversations_batch[i]
                    )

                api_protocol_conversations = (
                    self.ourborous_conversations_to_api_protocol_conversations(
                        self.gather_hist_conversation(
                            self.history_container.trained_histories
                            + self.history_container.untrained_histories,
                            last_idx=self.lookat_idx,
                        )
                    )
                )

                yield api_protocol_conversations

        start_with_tokens_batch = (
            [
                (
                    send_conversations.to_tokens(global_config.tokenizer_eval.encode)[0]
                    if send_conversations
                    else []
                )  # 发送消息
                + clist.to_tokens(global_config.tokenizer_eval.encode)[0]  # 思考消息
                + global_config.role[rpy_role]["prefix"]  # 角色special token
                + (
                    global_config.tokenizer_eval.encode(replier_prefix)
                    if replier_prefix
                    else []
                )  # 回复前缀
                for clist in think_conversation_batch
            ]
            if already_think
            else [
                (
                    send_conversations.to_tokens(global_config.tokenizer_eval.encode)[0]
                    if send_conversations
                    else []
                )  # 发送消息
                + global_config.role[rpy_role]["prefix"]  # 角色special token
                + (
                    global_config.tokenizer_eval.encode(replier_prefix)
                    if replier_prefix
                    else []
                )  # 回复前缀
            ]
            * num_rollouts
        )

        for speak_full_text_batch in self.batching(
            start_with_tokens_batch=start_with_tokens_batch,
            rpy_role=rpy_role,
            temperature=temperature,
            top_p=top_p,
            alpha_frequency=alpha_frequency,
            alpha_presence=alpha_presence,
            alpha_decay=alpha_decay,
            max_resp_len=max_resp_len,
            format_constraint_str=format_constraint_str,
            token_ban=token_ban,
            use_now_state_idx_list=[self.history_container.now_state_index]
            * num_rollouts,
            save_to_now_state_idx_list=None,
        ):
            out_speak_full_text_batch = [
                replier_prefix + out_full_text
                for out_full_text in speak_full_text_batch
            ]

            # 更新当前历史
            resp_conversations_batch = [
                cList.from_single_conversation(
                    Conversation(role=rpy_role, content=out_full_text)
                )
                for out_full_text in out_speak_full_text_batch
            ]
            full_conversations_batch = [
                cList(
                    send_conversations
                    + (think_conversation_batch[i] if already_think else [])
                    + resp_conversations_batch[i]
                )
                for i in range(num_rollouts)
            ]
            turn_msgs = [conversations for conversations in full_conversations_batch]

            for i, idx in enumerate(range(idx_range[0], idx_range[1])):
                self.history_container.untrained_histories[-1][idx] = turn_msgs[i]

            api_protocol_conversations = (
                self.ourborous_conversations_to_api_protocol_conversations(
                    self.gather_hist_conversation(
                        self.history_container.trained_histories
                        + self.history_container.untrained_histories,
                        last_idx=self.lookat_idx,
                    )
                )
            )
            yield api_protocol_conversations

        def batch_conversations_to_api_protocol_conversations_cat_history(
            agent,
            batch_full_conversations: List[cList],
        ):
            new_turn_msgs = [
                conversations for conversations in batch_full_conversations
            ]
            for i, idx in enumerate(range(idx_range[0], idx_range[1])):
                self.history_container.untrained_histories[-1][idx] = new_turn_msgs[i]
            return self.ourborous_conversations_to_api_protocol_conversations(
                self.gather_hist_conversation(
                    self.history_container.trained_histories
                    + self.history_container.untrained_histories
                )
                + batch_full_conversations[agent.lookat_idx]
            )

        self.temp_tools = (
            [copy.deepcopy(self.tools) for _ in range(num_rollouts)]
            if not regenerate
            else (
                self.temp_tools
                + [copy.deepcopy(self.tools) for _ in range(num_rollouts)]
            )
        )

        for new_turn_msgs in OurborousAgent.begin_tool_chain(
            self,
            resp_conversations_batch,
            full_conversations_batch,
            rpy_role=rpy_role,
            replier_name=replier_name,
            temperature=temperature,
            top_p=top_p,
            alpha_frequency=alpha_frequency,
            alpha_presence=alpha_presence,
            alpha_decay=alpha_decay,
            max_resp_len=max_resp_len,
            token_ban=token_ban,
            use_now_state_idx=self.history_container.now_state_index,
        ):
            yield batch_conversations_to_api_protocol_conversations_cat_history(
                self, new_turn_msgs
            )

        if self.reward_model_on:
            outer_history = ""
            for turn in self.history_container.untrained_histories[:-1]:
                chosen_rollout = turn[-1]
                outer_history += "\n".join([c().strip() for c in chosen_rollout]) + "\n"
            for i, rollout in enumerate(self.history_container.untrained_histories[-1]):
                inner_history = ""
                for j, conversation in enumerate(rollout):
                    if conversation.role in global_config.ego_types:
                        history = outer_history + inner_history
                        response = conversation().strip()
                        reward = self.trainner.rm_infer(history, response)
                        self.history_container.untrained_histories[-1][i][
                            j
                        ].score = reward
                    inner_history += f"{conversation().strip()}\n"

    def generate(
        self,
        send_conversations: cList,
        rpy_role: str = "response",
        replier_name: str = "assistant",
        num_rollouts: int = 1,
        temperature: float = 1,
        top_p: float = 0.8,
        alpha_frequency: float = 0.2,
        alpha_presence: float = 0.2,
        alpha_decay: float = 0.9961,
        max_resp_len: int = 2048,
        format_constraint_str: str = None,
        token_ban: List[int] = [],
        use_now_state_idx: str = None,
        save_to_now_state_idx: str = None,
        **kwargs,
    ):
        lock = threading.Lock()
        batch_events = [threading.Event() for _ in range(num_rollouts)]
        full_text_batch = [""] * num_rollouts

        def add_str_to_batch(i, next_str):
            with lock:
                full_text_batch[i] += next_str
                batch_events[i].set()

        tharads = []

        def batch_end():
            for thread in tharads:
                if thread.is_alive():
                    return False
            return True

        batch_callbacks = [partial(add_str_to_batch, i) for i in range(num_rollouts)]

        rpy_prefix = f"{replier_name}:"
        for i in range(num_rollouts):
            thread = threading.Thread(
                target=self._run_chat_task,
                kwargs={
                    "conversations": send_conversations,
                    "resp_start_with_role": rpy_role,
                    "resp_start_with_str": rpy_prefix,
                    "stop_with_tokens": global_config.role[rpy_role]["postfix"][:1],
                    "stop_supp_tokens": global_config.role[rpy_role]["postfix"][1:],
                    "temp": temperature,
                    "top_p": top_p,
                    "presence_penalty": alpha_presence,
                    "frequency_penalty": alpha_frequency,
                    "decay_penalty": alpha_decay,
                    "use_now_state_idx": use_now_state_idx,
                    "save_to_now_state_idx": save_to_now_state_idx,
                    "max_resp_len": max_resp_len,
                    "format_constraint_str": format_constraint_str,
                    "token_ban": token_ban,
                    "callback": batch_callbacks[i],
                },
            )
            tharads.append(thread)
            thread.start()

        while True:
            time.sleep(0.02)
            for i in range(num_rollouts):
                if batch_events[i].isSet():
                    yield full_text_batch
                    batch_events[i].clear()
            if batch_end():
                break

    def batching(
        self,
        start_with_tokens_batch: List[List[int]],
        rpy_role: str = "response",
        temperature: float = 1,
        top_p: float = 0.8,
        alpha_frequency: float = 0.2,
        alpha_presence: float = 0.2,
        alpha_decay: float = 0.9961,
        max_resp_len: int = 2048,
        format_constraint_str: str = None,
        token_ban: List[int] = [],
        use_now_state_idx_list: List[str] = None,
        save_to_now_state_idx_list: List[str] = None,
    ):
        lock = threading.Lock()
        batch_events = [threading.Event() for _ in range(len(start_with_tokens_batch))]
        full_text_batch = [""] * len(start_with_tokens_batch)

        def add_str_to_batch(i, next_str):
            with lock:
                full_text_batch[i] += next_str
                batch_events[i].set()

        tharads = []

        def batch_end():
            for thread in tharads:
                if thread.is_alive():
                    return False
            return True

        batch_callbacks = [
            partial(add_str_to_batch, i) for i in range(len(start_with_tokens_batch))
        ]

        for i in range(len(start_with_tokens_batch)):
            thread = threading.Thread(
                target=self._run_generate_task,
                kwargs={
                    "start_with_tokens": start_with_tokens_batch[i],
                    "stop_with_tokens": global_config.role[rpy_role]["postfix"][:1],
                    "stop_supp_tokens": global_config.role[rpy_role]["postfix"][1:],
                    "temp": temperature,
                    "top_p": top_p,
                    "presence_penalty": alpha_presence,
                    "frequency_penalty": alpha_frequency,
                    "decay_penalty": alpha_decay,
                    "use_now_state_idx": (
                        use_now_state_idx_list[i] if use_now_state_idx_list else None
                    ),
                    "save_to_now_state_idx": (
                        save_to_now_state_idx_list[i]
                        if save_to_now_state_idx_list
                        else None
                    ),
                    "max_resp_len": max_resp_len,
                    "format_constraint_str": format_constraint_str,
                    "token_ban": token_ban,
                    "callback": batch_callbacks[i],
                },
            )
            tharads.append(thread)
            thread.start()
        while True:
            time.sleep(0.02)
            for i in range(len(start_with_tokens_batch)):
                if batch_events[i].isSet():
                    yield full_text_batch
                    batch_events[i].clear()
            if batch_end():
                break

    def regenerate(
        self,
        num_rollouts: int = 1,
        temperature: float = 1,
        top_p: float = 0.8,
        alpha_frequency: float = 0.2,
        alpha_presence: float = 0.2,
        alpha_decay: float = 0.9961,
        max_resp_len: int = 2048,
        format_constraint_str: str = None,
        token_ban: List[int] = [],
        allow_think: bool = False,
        force_think: bool = False,
    ):
        last_conversations = self.back_to_last(back_history=False)
        if not last_conversations:
            return None
        return self.chat(
            send_conversations=last_conversations,
            rpy_role=self.last_rpy_role,
            replier_name=self.last_replier_name,
            num_rollouts=num_rollouts,
            temperature=temperature,
            top_p=top_p,
            alpha_frequency=alpha_frequency,
            alpha_presence=alpha_presence,
            alpha_decay=alpha_decay,
            max_resp_len=max_resp_len,
            format_constraint_str=format_constraint_str,
            token_ban=token_ban,
            allow_think=allow_think,
            force_think=force_think,
            regenerate=True,
        )

    def _run_chat_task(
        self,
        conversations: cList,
        resp_start_with_role: str = "response",
        resp_start_with_str: str = "assistant:",
        stop_with_tokens: List[int] = [65535],
        stop_supp_tokens: List[int] = [11],
        temp: float = 1,
        top_p: float = 0.8,
        presence_penalty: float = 0.2,
        frequency_penalty: float = 0.2,
        decay_penalty: float = 0.9961,
        use_now_state_idx: str = None,
        save_to_now_state_idx: str = None,
        max_resp_len: int = 2048,
        format_constraint_str: str = None,
        token_ban: List[int] = [],
        callback: callable = None,
    ):
        package = {
            "conversations": conversations.to_dict_list(),
            "resp_start_with_role": resp_start_with_role,
            "resp_start_with_str": resp_start_with_str,
            "stop_with_tokens": stop_with_tokens,
            "stop_supp_tokens": stop_supp_tokens,
            "temp": temp,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "decay_penalty": decay_penalty,
            "use_now_state_idx": use_now_state_idx,
            "save_to_now_state_idx": save_to_now_state_idx,
            "max_resp_len": max_resp_len,
            "format_constraint_str": format_constraint_str,
            "token_ban": token_ban,
        }
        with requests.post(
            self.inference_server + "/chat_task",
            json=package,
            stream=True,
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
            for chunk in response.iter_lines():
                result = json.loads(chunk)
                res_text = result["next"]
                if callback:
                    callback(res_text)

    def _run_generate_task(
        self,
        start_with_tokens: List[int],
        stop_with_tokens: List[int] = [65535],
        stop_supp_tokens: List[int] = [11],
        temp: float = 1,
        top_p: float = 0.8,
        presence_penalty: float = 0.2,
        frequency_penalty: float = 0.2,
        decay_penalty: float = 0.9961,
        use_now_state_idx: str = None,
        save_to_now_state_idx: str = None,
        max_resp_len: int = 2048,
        format_constraint_str: str = None,
        token_ban: List[int] = [],
        callback: callable = None,
    ):
        package = {
            "start_with_tokens": start_with_tokens,
            "stop_with_tokens": stop_with_tokens,
            "stop_supp_tokens": stop_supp_tokens,
            "temp": temp,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "decay_penalty": decay_penalty,
            "use_now_state_idx": use_now_state_idx,
            "save_to_now_state_idx": save_to_now_state_idx,
            "max_resp_len": max_resp_len,
            "format_constraint_str": format_constraint_str,
            "token_ban": token_ban,
        }
        with requests.post(
            self.inference_server + "/generate_task",
            json=package,
            stream=True,
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
            for chunk in response.iter_lines():
                result = json.loads(chunk)
                res_text = result["next"]
                if callback:
                    callback(res_text)

    def estimate_desires(
        self,
        role_prefix_pairs: list = [],
        start_with_conversations: cList = [],
        ignore_tokens: list = [11, 33, 261, 263, 41, 42],
        ignore_tolerance: int = 2,
    ):
        start_with_conversations_dict_list = start_with_conversations.to_dict_list()
        package = {
            "role_prefix_pairs": role_prefix_pairs,
            "start_with_conversations": start_with_conversations_dict_list,
            "use_now_state_idx": self.history_container.now_state_index,
            "ignore_tokens": ignore_tokens,
            "ignore_tolerance": ignore_tolerance,
        }
        hit = requests.post(
            self.inference_server + "/estimate_desires", json=package
        ).json()["hit"]
        return hit

    def add_message(self, add_conversations: cList, batch_idx: int = None):
        batch_idx = batch_idx if batch_idx is not None else self.lookat_idx
        self.history_container.untrained_histories[-1][batch_idx] += add_conversations
        api_protocol_conversations = (
            self.ourborous_conversations_to_api_protocol_conversations(
                self.gather_hist_conversation(
                    self.history_container.trained_histories
                    + self.history_container.untrained_histories,
                    last_idx=self.lookat_idx,
                )
            )
        )
        return api_protocol_conversations

    def savepoint(self, save_folder_path: str):
        os.makedirs(save_folder_path, exist_ok=True)
        self.history_container.savepoint(save_folder_path)
        with open(os.path.join(save_folder_path, "tools.pkl"), "wb") as f:
            pickle.dump(self.tools, f)
        with open(os.path.join(save_folder_path, "temp_tools.pkl"), "wb"):
            pickle.dump(self.temp_tools, f)
        messages = {
            "last_req_conversations": (
                self.last_req_conversations.to_dict_list()
                if self.last_req_conversations
                else None
            ),
            "last_rpy_role": self.last_rpy_role,
            "last_replier_name": self.last_replier_name,
        }
        with open(
            os.path.join(save_folder_path, "agent_metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(messages, f, ensure_ascii=False)

    def load_savepoint(self, save_folder_path: str):
        if os.path.exists(save_folder_path):
            self.history_container.load_from_savepoint(save_folder_path)
            if os.path.exists(os.path.join(save_folder_path, "tools.pkl")):
                with open(os.path.join(save_folder_path, "tools.pkl"), "rb"):
                    self.tools = pickle.load(f)
            if os.path.exists(os.path.join(save_folder_path, "temp_tools.pkl")):
                with open(os.path.join(save_folder_path, "temp_tools.pkl"), "rb"):
                    self.temp_tools = pickle.load(f)
            if not os.path.exists(
                os.path.join(save_folder_path, "agent_metadata.json")
            ):
                print(
                    f"Warning: 存档点文件夹 {save_folder_path} 中没有 agent_metadata.json 文件，无法读取上次对话信息。"
                )
            with open(
                os.path.join(save_folder_path, "agent_metadata.json"),
                "r",
                encoding="utf-8",
            ) as f:
                messages = json.load(f)
            self.last_req_conversations = (
                cList.from_dicts(messages["last_req_conversations"])
                if messages["last_req_conversations"]
                else None
            )
            self.last_rpy_role = messages["last_rpy_role"]
            self.last_replier_name = messages["last_replier_name"]
        else:
            print(f"Warning: 存档点文件夹 {save_folder_path} 不存在，未进行读取。")

    def auto_sft_check_and_begin(
        self,
    ):
        if self.auto_sft:
            self.trainner.auto_sft(self.history_container.untrained_histories[:-1])
            self.sync_train_event.set()

    def auto_rl_check_and_begin(
        self,
    ):
        if self.auto_rl:
            self.trainner.auto_rl(self.history_container.untrained_histories[:-1])
            self.sync_train_event.set()

    def auto_train_reward_model_check_and_begin(
        self,
    ):
        if self.auto_train_reward_model:
            self.trainner.train_reward_model(
                self.history_container.untrained_histories[:-1]
            )

    def sync_train_model(self):
        if len(self.history_container.untrained_histories) <= 1:
            return "构成ctx_len的生成数不够，本轮不同步模型与训练"
        model_dir = f"{global_config.cache_dir}/weights/online_learning.pth"
        state_dir = f"{global_config.cache_dir}/weights/online_learning.state"

        if not os.path.exists(model_dir):
            raise Exception(f"模型文件 {model_dir} 不存在，无法同步模型。")
        if not os.path.exists(state_dir):
            raise Exception(f"state文件 {state_dir} 不存在，无法同步state。")

        package = {
            "load_dir": model_dir,
        }
        resp = requests.post(
            self.inference_server + "/load_weight",
            json=package,
        ).json()
        if resp["message"] != "success":
            raise Exception(f"同步模型失败，返回信息：{resp['message']}")

        package = {
            "load_dir": state_dir,
            "state_id": self.history_container.state_after_train_index,
        }

        resp = requests.post(
            self.inference_server + "/load_state",
        ).json()
        if resp["message"] != "success":
            raise Exception(f"同步state失败，返回信息：{resp['message']}")

        clist_list = [hist[-1] for hist in self.history_container.untrained_histories]
        if len(clist_list > 1):
            last_history_conversations = sum(clist_list[1:-1], clist_list[0])
            package = {
                "conversations": last_history_conversations.to_dict_list(),
                "state_idx": self.history_container.state_after_train_index,
                "save_logits": False,
                "save_to_now_state_idx": self.history_container.last_state_index,
            }
            resp = requests.post(
                self.train_server + "/infer",
                json=package,
            ).json()
            if resp["message"] != "success":
                raise Exception(f"同步当前state失败，返回信息：{resp['message']}")
        else:
            package = {
                "from_access_token": self.history_container.state_after_train_index,
                "to_access_token": self.history_container.last_state_index,
            }
            resp = requests.post(
                self.train_server + "/copy_state",
                json=package,
            ).json()

            if resp["message"] != "success":
                raise Exception(f"同步上一时刻state失败，返回信息：{resp['message']}")

        history_conversations = sum(clist_list[1:], clist_list[0])

        package = {
            "conversations": history_conversations.to_dict_list(),
            "state_idx": self.history_container.state_after_train_index,
            "save_logits": False,
            "save_to_now_state_idx": self.history_container.now_state_index,
        }
        resp = requests.post(
            self.train_server + "/infer",
            json=package,
        ).json()
        if resp["message"] != "success":
            raise Exception(f"同步当前state失败，返回信息：{resp['message']}")

        return "同步模型成功。"

    def save_data_hist(self, select, save_folder_dir):
        assert select in ["untrained", "trained", "full"]
        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        save_path = os.path.join(save_folder_dir, f"{time_str}_sft_hist.jsonl")
        self.history_container.to_sft_dataset(
            save=True, save_path=save_path, select=select
        )


class UsrAgent:
    def __init__(
        self,
        include_host: str,
        include_port: int,
        linked_main_agent: OurborousAgent,
    ):
        self.linked_main_agent = linked_main_agent
        self.include_host = include_host
        self.include_port = include_port
        self.time_out = 30
        self.msg = ""
        self.is_started = False
        self.receive_event = threading.Event()
        self.lock = threading.Lock()

        self.act_history = cList()

    def user_tool_interface(
        self,
        send_conversations: cList,
        sender_name,
        sender_role,
        tools: Ourboroustools,
        history_container: OurborousAgentHistoryContainer,
    ):
        self.is_started = True

        for resp_conversation, act_history in tools.router.chain(
            self,
            send_conversations[-1],
            send_conversations,
            sender_name=sender_name,
            rpy_role=sender_role,
            replier_name=sender_name,
        ):

            hist_clists = [
                hist[-1]
                for hist in history_container.trained_histories
                + history_container.untrained_histories
            ]
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
                    (
                        sum(hist_clists[1:], hist_clists[0])
                        if len(hist_clists)
                        else cList()
                    )
                    + act_history
                )
            ]
            self.act_history = act_history
            yield api_protocol_conversations
        self.is_started = False

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["app"]  # 从状态字典中删除难以保存的对象
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.app = Bottle()

    def receive_message(self, msg, sender_role, sender_name):
        api_protocol_conversations = []
        if msg.strip():
            if self.is_started:
                with self.lock:
                    self.msg = msg
                    self.receive_event.set()
            else:
                send_conversations = cList.from_single_conversation(
                    Conversation(
                        role=sender_role,
                        content=(f"{sender_name}: {msg}" if sender_name else msg),
                    )
                )
                for api_protocol_conversations in self.user_tool_interface(
                    send_conversations,
                    sender_name,
                    sender_role,
                    (
                        self.linked_main_agent.temp_tools[
                            self.linked_main_agent.lookat_idx
                        ]
                        if self.linked_main_agent.temp_tools
                        else self.linked_main_agent.tools
                    ),
                    self.linked_main_agent.history_container,
                ):
                    yield api_protocol_conversations
                yield api_protocol_conversations

    def generate(
        self,
        **kwargs,
    ):
        yield ["(交互未结束，请在命令行输入进一步的交互内容)"]
        self.receive_event.clear()
        self.receive_event.wait()
        speak_full_text_batch = [self.msg]
        yield speak_full_text_batch
        self.msg = ""
        self.receive_event.clear()

    def reset(self):
        self.is_started = False
        self.act_history = cList()
        self.msg = ""
        self.receive_event.set()
        time.sleep(0.02)
        self.receive_event.clear()
