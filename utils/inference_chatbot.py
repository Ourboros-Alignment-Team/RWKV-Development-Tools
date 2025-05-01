from utils.containers import ChatbotMessagesContainer
from utils.message_manager import cList, Conversation
import requests
import gc
import json, sys
import os
from config import global_config


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


class Chatbot:
    def __init__(
        self,
        server="http://0.0.0.0:4514",
        load_state_dir=None,
        temp=1,
        top_p=0.7,
        presence_penalty=0.2,
        frequency_penalty=0.2,
        decay_penalty=0.9961,
        bot_sp_token_role="response",
        usr_sp_token_role="conversation",
    ) -> None:
        self.server = server
        self.is_on = False
        self.init_state_access_token = None
        self.now_state_access_token = None
        self.last_state_access_token = None
        self.history = ChatbotMessagesContainer()
        self.is_on = self.reboot(load_state_dir)
        self.bot_sp_token_role = bot_sp_token_role
        self.usr_sp_token_role = usr_sp_token_role
        self.stop_with_tokens = global_config.role[self.bot_sp_token_role]["postfix"][
            :1
        ]
        self.stop_supp_tokens = global_config.role[self.bot_sp_token_role]["postfix"][
            1:-1
        ]

        self.temp = temp
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.decay_penalty = decay_penalty

    def change_sp_tokens(self, usr_sp_token_role, bot_sp_token_role):
        self.bot_sp_token_role = bot_sp_token_role
        self.usr_sp_token_role = usr_sp_token_role
        self.stop_with_tokens = global_config.role[self.bot_sp_token_role]["postfix"][
            :1
        ]
        self.stop_supp_tokens = global_config.role[self.bot_sp_token_role]["postfix"][
            1:-1
        ]
        print(f"change sp tokens: user:{usr_sp_token_role}, bot:{bot_sp_token_role}")

    def reboot(self, load_state_dir=None):
        if not test_server(self.server):
            return False
        if self.init_state_access_token is not None:
            requests.post(
                self.server + "/remove_state_id",
                json={"access_token": f"{self.init_state_access_token}"},
            )
        if self.now_state_access_token is not None:
            requests.post(
                self.server + "/remove_state_id",
                json={"access_token": f"{self.now_state_access_token}"},
            )
        if self.last_state_access_token is not None:
            requests.post(
                self.server + "/remove_state_id",
                json={"access_token": f"{self.last_state_access_token}"},
            )
        package = {"load_dir": load_state_dir}
        self.init_state_access_token = requests.post(
            self.server + "/regist_state_id", json=package
        ).json()["access_token"]
        self.now_state_access_token = requests.post(
            self.server + "/regist_state_id", json=package
        ).json()["access_token"]
        self.last_state_access_token = requests.post(
            self.server + "/regist_state_id", json=package
        ).json()["access_token"]
        self.is_on = True
        self.history.restart()

        return True

    def set_server(self, server):
        self.server = server

    def off(self):
        if test_server():
            if self.init_state_access_token is not None:
                requests.post(
                    self.server + "/remove_state_id",
                    json={"access_token": "self.init_state_access_token"},
                )
            if self.now_state_access_token is not None:
                requests.post(
                    self.server + "/remove_state_id",
                    json={"access_token": "self.now_state_access_token"},
                )
            if self.last_state_access_token is not None:
                requests.post(
                    self.server + "/remove_state_id",
                    json={"access_token": "self.last_state_access_token"},
                )
        self.init_state_access_token = None
        self.now_state_access_token = None
        self.last_state_access_token = None
        self.is_on = False

    def reset(self, re_init_state_dir=None):
        if re_init_state_dir:
            package = {
                "access_token": self.init_state_access_token,
                "load_dir": re_init_state_dir,
            }
            requests.post(self.server + "/load_state", json=package)
        package = {
            "from_access_token": self.init_state_access_token,
            "to_access_token": self.last_state_access_token,
        }
        requests.post(self.server + "/copy_state", json=package)
        package = {
            "from_access_token": self.init_state_access_token,
            "to_access_token": self.now_state_access_token,
        }
        requests.post(self.server + "/copy_state", json=package)
        self.history.restart()

    def back_to_last(self):
        package = {
            "from_access_token": self.last_state_access_token,
            "to_access_token": self.now_state_access_token,
        }
        requests.post(self.server + "/copy_state", json=package)
        last_conversation = self.history.back_to_last()
        return last_conversation

    def regenerate(self, rpy_prefix, stream, need_ppl=False):
        last_conversation = self.back_to_last()
        self.supplement_last_token()
        if last_conversation:
            resp_str = self.chat(
                cList([last_conversation]),
                rpy_prefix,
                stream,
                need_ppl=need_ppl,
            )
            return resp_str
        return None

    def stream_regenerate(
        self,
        rpy_prefix,
        max_resp_len=512,
        format_str=None,
        token_ban=[],
        need_ppl=False,
    ):
        last_conversation = self.back_to_last()
        self.supplement_last_token()
        if last_conversation:
            return self.stream_chat(
                cList([last_conversation]),
                rpy_prefix,
                max_resp_len=max_resp_len,
                format_constrain_str=format_str,
                token_ban=token_ban,
                need_ppl=need_ppl,
            )

        return None

    def chat(
        self,
        send_conversations: cList,
        rpy_prefix,
        stream,
        rpy_role=None,
        chunk_len=512,
        max_resp_len=2048,
        format_constrain_str=None,
        mark_last_conversation=True,
        token_ban=[],
        need_ppl=False,
    ):
        self.supplement_last_token()
        if mark_last_conversation:
            package = {
                "from_access_token": self.now_state_access_token,
                "to_access_token": self.last_state_access_token,
            }
            requests.post(self.server + "/copy_state", json=package)
        if rpy_role is None:
            rpy_role = self.bot_sp_token_role
        package = {
            "conversations": (
                send_conversations.to_dict_list() if send_conversations else None
            ),
            "resp_start_with_role": rpy_role,
            "resp_start_with_str": rpy_prefix,
            "stop_with_tokens": self.stop_with_tokens,
            "stop_supp_tokens": self.stop_supp_tokens,
            "temp": self.temp,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "decay_penalty": self.decay_penalty,
            "use_now_state_idx": self.now_state_access_token,
            "save_to_now_state_idx": self.now_state_access_token,
            "max_resp_len": max_resp_len,
            "format_constrain_str": format_constrain_str,
            "token_ban": token_ban,
            "need_ppl": need_ppl,
        }
        if stream:
            resp_full_str = ""
            with requests.post(
                self.server + "/chat_task", json=package, stream=True
            ) as response:
                if response.status_code != 200:
                    print(f"Error: Received status code {response.status_code}")
                for chunk in response.iter_lines():
                    result = json.loads(chunk)
                    res_text = result["next"]
                    print(f"{res_text}", end="")
                    sys.stdout.flush()
                    resp_full_str += res_text
                if need_ppl:
                    in_ppls = result["in_ppls"]
                    out_ppls = result["out_ppls"]
        else:
            resp = requests.post(
                self.server + "/chat",
                json=package,
            ).json()["response"]
            if need_ppl:
                resp_full_str, in_ppls, out_ppls = resp
            else:
                resp_full_str = resp
        cl = cList()
        cl += send_conversations
        cl.append(Conversation(role=rpy_role, content=rpy_prefix + resp_full_str))
        self.history.add_messages(mark_last_conversation, cl)
        if need_ppl:
            return resp_full_str, in_ppls, out_ppls
        return resp_full_str

    def estimate_desire(
        self,
        target_role: str = "response",
        target_prefix: str = "",
        start_with_conversations: cList = [],
        ignore_tokens: list = [11, 33, 261, 263, 41, 42],
        ignore_tolerance: int = 2,
    ):
        role_prefix_pairs = [(target_role, target_prefix)]
        start_with_conversations_dict_list = start_with_conversations.to_dict_list()
        package = {
            "role_prefix_pairs": role_prefix_pairs,
            "start_with_conversations": start_with_conversations_dict_list,
            "use_now_state_idx": self.now_state_access_token,
            "ignore_tokens": ignore_tokens,
            "ignore_tolerance": ignore_tolerance,
        }
        hit = requests.post(self.server + "/estimate_desires", json=package).json()["hit"]
        return hit >= 0
    
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
            "use_now_state_idx": self.now_state_access_token,
            "ignore_tokens": ignore_tokens,
            "ignore_tolerance": ignore_tolerance,
        }
        hit = requests.post(self.server + "/estimate_desires", json=package).json()["hit"]
        return hit

    def stream_chat(
        self,
        send_conversations: cList,
        rpy_prefix,
        rpy_role=None,
        chunk_len=512,
        max_resp_len=512,
        format_constrain_str=None,
        mark_last_conversation=True,
        token_ban=[],
        need_ppl=False,
    ):
        self.supplement_last_token()
        if mark_last_conversation:
            package = {
                "from_access_token": self.now_state_access_token,
                "to_access_token": self.last_state_access_token,
            }
            requests.post(self.server + "/copy_state", json=package)
        if rpy_role is None:
            rpy_role = self.bot_sp_token_role
        package = {
            "conversations": (
                send_conversations.to_dict_list() if send_conversations else None
            ),
            "resp_start_with_role": rpy_role,
            "resp_start_with_str": rpy_prefix,
            "stop_with_tokens": self.stop_with_tokens,
            "stop_supp_tokens": self.stop_supp_tokens,
            "temp": self.temp,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "decay_penalty": self.decay_penalty,
            "use_now_state_idx": self.now_state_access_token,
            "save_to_now_state_idx": self.now_state_access_token,
            "max_resp_len": max_resp_len,
            "format_constrain_str": format_constrain_str,
            "token_ban": token_ban,
            "need_ppl": need_ppl,
        }
        resp_full_str = ""
        with requests.post(
            self.server + "/chat_task", json=package, stream=True
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
            for chunk in response.iter_lines():
                result = json.loads(chunk)
                res_text = result["next"]
                print(f"{res_text}", end="")
                sys.stdout.flush()
                resp_full_str += res_text
                if need_ppl:
                    in_ppls = result["in_ppls"]
                    out_ppls = result["out_ppls"]
                    yield res_text, resp_full_str, in_ppls, out_ppls
                else:
                    yield res_text, resp_full_str
        cl = cList()
        if send_conversations is not None:
            cl += send_conversations
        cl.append(Conversation(role=rpy_role, content=rpy_prefix + resp_full_str))
        self.history.add_messages(mark_last_conversation, cl)

    def add_messages(self, add_conversations: cList, mark_last_conversation=False):
        if mark_last_conversation:
            package = {
                "from_access_token": self.now_state_access_token,
                "to_access_token": self.last_state_access_token,
            }
            requests.post(self.server + "/copy_state", json=package)
        self.supplement_last_token()
        package = {
            "conversations": (
                add_conversations.to_dict_list() if add_conversations else [11]
            ),
            "state_idx": self.now_state_access_token,
            "save_to_now_state_idx": self.now_state_access_token,
            "save_logits": False,
        }
        requests.post(
            self.server + "/infer",
            json=package,
        ).json()
        self.history.add_messages(mark_last_conversation, add_conversations)

    def supplement_last_token(self):
        if self.history.last_token:
            package = {
                "tokens": [self.history.last_token],
                "state_idx": self.now_state_access_token,
                "save_to_now_state_idx": self.now_state_access_token,
                "save_logits": False,
            }
            requests.post(
                self.server + "/infer_tokens",
                json=package,
            )

    def save_ckpt(self, ckpt_folder, ckpt_name):
        folder = os.path.join(ckpt_folder, ckpt_name)
        os.makedirs(folder, exist_ok=True)
        self.history.save_as_checkpoint(folder, ckpt_name)
        package = {
            "access_token": self.now_state_access_token,
            "to_dir": os.path.join(folder, "now.state"),
        }
        requests.post(self.server + "/save_state", json=package)
        package = {
            "access_token": self.init_state_access_token,
            "to_dir": os.path.join(folder, "init.state"),
        }
        requests.post(self.server + "/save_state", json=package)
        package = {
            "access_token": self.last_state_access_token,
            "to_dir": os.path.join(folder, "last.state"),
        }
        requests.post(self.server + "/save_state", json=package)
        args = {
            "last_token": self.history.last_token,
            "last_token_last_conversation": self.history.last_token_last_conversation,
            "now_access_token": self.now_state_access_token,
            "init_access_token": self.init_state_access_token,
            "last_access_token": self.last_state_access_token,
        }
        with open(os.path.join(folder, "args.json"), mode="w", encoding="utf-8") as f:
            f.write(json.dumps(args, ensure_ascii=False))

    def load_ckpt(self, ckpt_dir):
        self.history.load_checkpoint(ckpt_dir)
        now_dir = os.path.join(ckpt_dir, "now.state")
        init_dir = os.path.join(ckpt_dir, "init.state")
        last_dir = os.path.join(ckpt_dir, "last.state")
        args_dir = os.path.join(ckpt_dir, "args.json")
        if os.path.exists(args_dir):
            with open(args_dir, "r", encoding="utf-8") as f:
                args = json.load(f)
            self.history.last_token = args["last_token"]
            self.history.last_token_last_conversation = args[
                "last_token_last_conversation"
            ]
            if os.path.exists(now_dir):
                package = {
                    "access_token": args["now_access_token"],
                    "load_dir": now_dir,
                }
                requests.post(self.server + "/load_state", json=package)
            if os.path.exists(last_dir):
                package = {
                    "access_token": args["init_access_token"],
                    "load_dir": init_dir,
                }
                requests.post(self.server + "/load_state", json=package)
            if os.path.exists(args_dir):
                package = {
                    "access_token": args["last_access_token"],
                    "load_dir": last_dir,
                }
                requests.post(self.server + "/load_state", json=package)

    def save_as_dataset(self, folder_dir, name):
        conversations = self.history.conversations_history
        dic, txt = conversations.to_dataset()
        jsonl = os.path.join(folder_dir, f"{name}.jsonl")
        with open(jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(dic, ensure_ascii=False))

    def ctx_len(self, encoding_func):
        return self.history.conversations_to_train.calc_ctx_len(encoding_func)
