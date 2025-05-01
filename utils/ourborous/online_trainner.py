
from config import global_config
from utils.ourborous.history import OurborousAgentHistoryContainer
import requests
import os
from threading import Thread

class OurborousOnlineTrainner:
    def __init__(
        self,
        agent,
        train_server: str,
        auto_sft_lr: float,
        auto_rl_lr: float,
        auto_rl_bsz: int,
        auto_rl_times: int,
        accumulate_grad: bool,
        accumulate_grad_bsz: int,
        reward_model_server: str,
        save_rm_log_dir: str,
    ):
        self.agent = agent
        self.train_server = train_server
        self.auto_sft_lr = auto_sft_lr
        self.auto_rl_lr = auto_rl_lr
        self.auto_rl_bsz = auto_rl_bsz
        self.auto_rl_times = auto_rl_times
        self.accumulate_grad = accumulate_grad
        self.accumulate_grad_bsz = accumulate_grad_bsz
        self.reward_model_server = reward_model_server
        self.save_rm_log_dir=save_rm_log_dir
        os.makedirs(self.save_rm_log_dir,exist_ok=True)
        
        
    def auto_sft(
        self,history
    ):
        state_after_train_idx = self.agent.history_container.state_after_train_index
        state_after_train_dir = (
            f"{global_config.cache_dir}/{state_after_train_idx}.state"
        )

        package = {
            "history": OurborousAgentHistoryContainer.serialize_history(
                history
            ),
            "begin_with_state_dir": state_after_train_dir,
            "save_weight_folder": f"{global_config.cache_dir}/weights",
            "save_weight_name": "online_learning",
            "lr_init": self.auto_sft_lr,
            "lr_final": self.auto_sft_lr,
        }
        with requests.post(
            self.train_server + "/train_sft_online",
            json=package,
            stream=True,
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
            for chunk in response.iter_lines():
                pass
            return_text = f"""训练完成。
权重保存在： {global_config.cache_dir}/weights/online_learning.pth
state保存在： {global_config.cache_dir}/weights/online_learning.state"""
            print(return_text)
        return return_text

    def auto_rl(
        self,history
    ):
        state_after_train_idx = self.agent.history_container.state_after_train_index
        state_after_train_dir = (
            f"{global_config.cache_dir}/{state_after_train_idx}.state"
        )

        package = {
            "history": OurborousAgentHistoryContainer.serialize_history(
                history
            ),
            "ref_model_server": self.agent.inference_server,
            "begin_with_state_dir": state_after_train_dir,
            "save_weight_folder": f"{global_config.cache_dir}/weights",
            "save_weight_name": "online_learning",
            "lr_init": self.auto_rl_lr,
            "lr_final": self.auto_rl_lr,
            "lr_warmup": 100,
            "accumulate_grad": self.accumulate_grad,
            "train_batch_size": self.auto_rl_bsz,
            "n_train_each_episode": self.auto_rl_times,
        }
        print("grpo训练中>",end="")
        with requests.post(
            self.train_server + "/train_grpo_online",
            json=package,
            stream=True,
        ):
            print(">",end="")
        return_text = f"""\n训练完成。
权重保存在： {global_config.cache_dir}/weights/online_learning.pth
state保存在： {global_config.cache_dir}/weights/online_learning.state"""
        print(return_text)
        return return_text

    def rm_infer(self,history:str,response:str,):
        package={
            "history": history,
            "response": response,
        }
        resp=requests.post(
            self.reward_model_server + "/infer",
            json=package,
            
        ).json()
        score=resp["score"]
        return score 
    
    @staticmethod
    def history_to_reward_model_input(history):
        data=[]
        outer_history=""
        for turn in history:
            data_lines=[] #history,response_list
            for rollout in turn:
                inner_history = ""
                for conversation in rollout:
                    response_str=conversation().strip()
                    if conversation.score is not None:
                        history_str=outer_history+inner_history
                        reward=conversation.score
                        data_lines.append((history_str,response_str,reward))
                    inner_history += f"{response_str}\n"
            outer_history+="\n".join([c().strip() for c in turn[-1]])
            data.append(data_lines)
        return data
    
    def train_reward_model(self,history):
        data=self.history_to_reward_model_input(history)
        for data_lines in data:
            print(data_lines)
            package={
                "data_list": data_lines,
                "save_ckpt":True
            }
            
            requests.post(
                self.reward_model_server + "/train",
                json=package,
            )
            
            