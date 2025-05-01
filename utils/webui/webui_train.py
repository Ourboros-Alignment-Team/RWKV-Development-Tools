import requests
import json
import os
from utils.message_manager import cList
from utils.webui.webui_utils import loss_curve_plot
import time


class TrainAgent:
    def __init__(self, train_server="http://0.0.0.0:3000"):
        self.train_server = train_server

    def load_train_model(self, model_path):
        """加载模型的逻辑"""
        response = requests.post(
            self.train_server + "/load_model", json={"load_dir": model_path}
        ).json()
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
        return f"模型已加载：{model_path}"

    def train_single_folder(
        self,
        forder_dir,
        epoch,
        batch_size_per_gpu,
        n_save_ckpt,
        multi_scale_ctx,
        multi_scale_alpha,
        keep_states_mode,
        use_qa_mask=False,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
        begin_with_state_dir=None,
    ):
        """训练单个文件夹的逻辑"""
        variables = {
            "forder_dir": forder_dir,
            "epoch": epoch,
            "batch_size_per_gpu": batch_size_per_gpu,
            "n_save_ckpt": n_save_ckpt,
            "multi_scale_ctx": multi_scale_ctx,
            "multi_scale_alpha": multi_scale_alpha,
            "keep_states_mode": keep_states_mode,
            "begin_with_state_dir": begin_with_state_dir,
            "use_qa_mask": use_qa_mask,
            "lr_init": lr_init,
            "lr_final": lr_final,
            "warmup_steps": warmup_steps,
        }
        text_loss_list = []
        with requests.post(
            self.train_server + "/train_single_datafolder", json=variables, stream=True
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                yield 0, f"Error: Received status code {response.status_code}", loss_curve_plot(
                    []
                )
            for r in response.iter_lines():
                result = json.loads(r)
                if "over" in result:
                    to_dir = result["to_dir"]
                    prefix = "训练完成，" if result["over"] else ""
                    output_text = f"{prefix}已保存至{to_dir}"
                    yield 100, output_text, loss_curve_plot(text_loss_list)
                else:
                    epoch = result["epoch"]
                    step = result["step"]
                    mean_text_loss = result["mean_text_loss"]
                    text_loss = result["text_loss"]
                    text_loss_list.append(text_loss)
                    n_tokens = result["n_tokens"]
                    left_tokens = result["left_tokens"]
                    progress_percent = (
                        (n_tokens - left_tokens) / (n_tokens + 1e-4) * 100
                    )
                    output_text = (
                        f"Epoch: {epoch}, Step: {step}, Loss: {mean_text_loss}"
                    )

                    yield progress_percent, output_text, loss_curve_plot(text_loss_list)

    def train_multiple_folders(
        self,
        folder_weight_dir_list,
        epoch,
        batch_size_per_gpu,
        n_save_ckpt,
        save_step_on=False,
        n_save_step=None,
        use_qa_mask=False,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
    ):
        """训练多个文件夹的逻辑"""
        n_save_step = n_save_step if save_step_on else None
        variables = {
            "epoch": epoch,
            "batch_size_per_gpu": batch_size_per_gpu,
            "n_save_ckpt": n_save_ckpt,
            "n_save_step": n_save_step,
            "use_qa_mask": use_qa_mask,
            "lr_init": lr_init,
            "lr_final": lr_final,
            "warmup_steps": warmup_steps,
        }
        variables["folder_weight_dir_list"] = json.loads(folder_weight_dir_list)
        text_loss_list = []
        with requests.post(
            self.train_server + "/train_from_folders", json=variables, stream=True
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                yield 0, f"Error: Received status code {response.status_code}", loss_curve_plot(
                    []
                )
            for r in response.iter_lines():
                result = json.loads(r)
                if "over" in result:
                    to_dir = result["to_dir"]
                    prefix = "训练完成，" if result["over"] else ""
                    output_text = f"{prefix}已保存至{to_dir}"
                    yield 100, output_text, loss_curve_plot(text_loss_list)
                else:
                    epoch = result["epoch"]
                    step = result["step"]
                    mean_loss = result["mean_text_loss"]
                    text_loss = result["text_loss"]
                    text_loss_list.append(text_loss)
                    n_data = result["n_data"]
                    left_data = result["left_data"]
                    progress_percent = (n_data - left_data) / n_data * 100
                    output_text = f"Epoch: {epoch}, Step: {step}, Loss: {mean_loss}"
                    yield progress_percent, output_text, loss_curve_plot(text_loss_list)

    def train_gsm8k(
        self,
        parquet_file_path,
        n_rollout_questions,
        req_role,
        req_prefix,
        resp_role,
        resp_prefix,
        temperature,
        top_p,
        alpha_presence,
        alpha_frequency,
        alpha_decay,
        max_ctx,
        lr_init,
        lr_final,
        warmup_steps,
        accumulate_grad,
        n_save_ckpt,
        n_save_episode_ckpt,
        num_rollouts,
        rollout_tiny_batch_size,
        train_batch_size,
    ):
        print("开始训练 数据集：gsm8k...")
        variables = {
            "parquet_file_path": parquet_file_path,
            "n_rollout_questions": n_rollout_questions,
            "req_role": req_role,
            "req_prefix": req_prefix,
            "resp_role": resp_role,
            "resp_prefix": resp_prefix,
            "temperature": temperature,
            "top_p": top_p,
            "alpha_presence": alpha_presence,
            "alpha_frequency": alpha_frequency,
            "alpha_decay": alpha_decay,
            "max_ctx": max_ctx,
            "lr_init": lr_init,
            "lr_final": lr_final,
            "warmup_steps": warmup_steps,
            "accumulate_grad": accumulate_grad,
            "n_save_ckpt": n_save_ckpt,
            "n_save_episode_ckpt": n_save_episode_ckpt,
            "num_rollouts": num_rollouts,
            "tiny_batch_size": rollout_tiny_batch_size,
            "train_batch_size": train_batch_size,
        }
        loss_list = []
        rewards_list = []
        kl_list = []

        with requests.post(
            self.train_server + "/train_gsm8k", json=variables, stream=True
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                yield f"Error: Received status code {response.status_code}", loss_curve_plot(
                    []
                ), loss_curve_plot(
                    []
                ), loss_curve_plot(
                    []
                )
            for r in response.iter_lines():
                result = json.loads(r)
                if "over" in result:
                    to_dir = result["to_dir"]
                    prefix = "训练完成，" if result["over"] else ""
                    output_text = f"{prefix}已保存至{to_dir}"
                    yield output_text, loss_curve_plot(
                        loss_list, caption="RL Loss"
                    ), loss_curve_plot(
                        rewards_list, caption="Rewards"
                    ), loss_curve_plot(
                        kl_list, caption="KL"
                    )
                    time.sleep(0.01)
                else:
                    epoch = result["epoch"]
                    step = result["step"]
                    loss = result["loss"]
                    kl = result["kl"]
                    sum_rewards = result["sum_rewards"]
                    loss_list.append(loss)
                    rewards_list.append(sum_rewards / num_rollouts)
                    kl_list.append(kl)
                    output_text = f"正在训练，Epoch: {epoch}, Step: {step}, Loss: {loss}, KL: {kl}, Rewards: {sum_rewards}。"
                    yield output_text, loss_curve_plot(
                        loss_list, caption="RL Loss"
                    ), loss_curve_plot(
                        rewards_list, caption="Rewards"
                    ), loss_curve_plot(
                        kl_list, caption="KL"
                    )
                    time.sleep(0.01)

    def train_ourborous_rl(
        self,
        dataset_dir,
        n_save_episode_ckpt,
        train_batch_size,
        lr_init,
        lr_final,
        warmup_steps,
        accumulate_grad,
    ):
        print("开始训练 数据集类型：ourborous...")
        variables = {
            "dataset_dir": dataset_dir,
            "n_save_episode_ckpt": n_save_episode_ckpt,
            "train_batch_size": train_batch_size,
            "lr_init": lr_init,
            "lr_final": lr_final,
            "warmup_steps": warmup_steps,
            "accumulate_grad": accumulate_grad,
        }
        loss_list = []
        rewards_list = []
        kl_list = []
        with requests.post(
            self.train_server + "/train_grpo_from_group_dataset",
            json=variables,
            stream=True,
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                yield f"Error: Received status code {response.status_code}", loss_curve_plot(
                    []
                ), loss_curve_plot(
                    []
                ), loss_curve_plot(
                    []
                )
            for r in response.iter_lines():
                result = json.loads(r)
                if "over" in result:
                    to_dir = result["to_dir"]
                    prefix = "训练完成，" if result["over"] else ""
                    output_text = f"{prefix}已保存至{to_dir}"
                    yield output_text, loss_curve_plot(
                        loss_list, caption="RL Loss"
                    ), loss_curve_plot(
                        rewards_list, caption="Rewards"
                    ), loss_curve_plot(
                        kl_list, caption="KL"
                    )
                    time.sleep(0.01)
                else:
                    i_file = result["i_file"]
                    step = result["step"]
                    loss = result["loss"]
                    kl = result["kl"]
                    sum_rewards = result["sum_rewards"]
                    loss_list.append(loss)
                    rewards_list.append(sum_rewards)
                    kl_list.append(kl)
                    output_text = f"正在训练，File_idx: {i_file}, Step: {step}, Loss: {loss}, KL: {kl}, Rewards: {sum_rewards}。"
                    yield output_text, loss_curve_plot(
                        loss_list, caption="RL Loss"
                    ), loss_curve_plot(
                        rewards_list, caption="Rewards"
                    ), loss_curve_plot(
                        kl_list, caption="KL"
                    )
                    time.sleep(0.01)

    def train_grpo_pair_rl(
        self,
        dataset_fp,
        n_samples_episode,
        n_episodes,
        role_system,
        system_prefix,
        role_sender,
        sender_prefix,
        role_receiver,
        receiver_prefix,
        lr_init,
        lr_final,
        warmup_steps,
        accumulate_grad,
        n_save_episode_ckpt,
        train_batch_size,
    ):
        print("开始训练 数据集类型：dpo pair...")
        variables = {
            "dataset_fp": dataset_fp,
            "n_samples_episode": n_samples_episode,
            "n_episodes": n_episodes,
            "role_system": role_system,
            "system_prefix": system_prefix,
            "role_sender": role_sender,
            "sender_prefix": sender_prefix,
            "role_receiver": role_receiver,
            "receiver_prefix": receiver_prefix,
            "lr_init": lr_init,
            "lr_final": lr_final,
            "warmup_steps": warmup_steps,
            "accumulate_grad": accumulate_grad,
            "n_save_episode_ckpt": n_save_episode_ckpt,
            "train_batch_size": train_batch_size,
        }
        loss_list = []
        rewards_list = []
        kl_list = []
        with requests.post(
            self.train_server + "/train_grpo_from_pair_dataset",
            json=variables,
            stream=True,
        ) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                yield f"Error: Received status code {response.status_code}", loss_curve_plot(
                    []
                ), loss_curve_plot(
                    []
                ), loss_curve_plot(
                    []
                )
            for r in response.iter_lines():
                result = json.loads(r)
                if "over" in result:
                    to_dir = result["to_dir"]
                    prefix = "训练完成，" if result["over"] else ""
                    output_text = f"{prefix}已保存至{to_dir}"
                    yield output_text, loss_curve_plot(
                        loss_list, caption="RL Loss"
                    ), loss_curve_plot(
                        rewards_list, caption="Rewards"
                    ), loss_curve_plot(
                        kl_list, caption="KL"
                    )
                    time.sleep(0.01)
                else:
                    i_file = result["i_file"]
                    step = result["step"]
                    loss = result["loss"]
                    kl = result["kl"]
                    sum_rewards = result["sum_rewards"]
                    loss_list.append(loss)
                    rewards_list.append(sum_rewards)
                    kl_list.append(kl)
                    output_text = f"正在训练，File_idx: {i_file}, Step: {step}, Loss: {loss}, KL: {kl}, Rewards: {sum_rewards}。"
                    yield output_text, loss_curve_plot(
                        loss_list, caption="RL Loss"
                    ), loss_curve_plot(
                        rewards_list, caption="Rewards"
                    ), loss_curve_plot(
                        kl_list, caption="KL"
                    )
                    time.sleep(0.01)


# def grpo_router(
#     train_agent: TrainAgent,
#     mode,
#     *args,
#     **kwargs,
# ):
#     if mode == "gsm8k":
#         for a, b, c, d in train_agent.train_gsm8k(*args, **kwargs):
#             yield a, b, c, d
#             time.sleep(0.01)
#     elif mode == "ourborous":
#         for a, b, c, d in train_agent.train_ourborous_rl(*args, **kwargs):
#             yield a, b, c, d
#             time.sleep(0.01)
#     elif mode == "dpo_pair":
#         for a, b, c, d in train_agent.train_grpo_pair_rl(*args, **kwargs):
#             yield a, b, c, d
#             time.sleep(0.01)
