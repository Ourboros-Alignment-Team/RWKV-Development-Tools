from config import global_config

grpo_config = global_config.grpo
infer_config = global_config.infer_service_config
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import requests
from utils.message_manager import Conversation, cList
from utils.rl.grpo.functions import (
    zero_pad_sequences,
    group_advantages,
    get_batch_log_probs,
)

from typing import List
from utils.rl.grpo.replay import ReplaySlidingWindow, ExperienceHist
from utils.rl.grpo.loss import GRPOLoss, kl_div
from RWKV.functions import batch_chat
from config import RWKVStates
import gc
from typing import Tuple, List

cache_dir = ".cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class GRPOTrainer:
    def __init__(self, rwkv_policy, ref_model_server, tokenizer):
        self.ref_model_server = ref_model_server
        self.rwkv_policy = rwkv_policy
        self.tokenizer = tokenizer
        self.state_idx = None

    @torch.no_grad()
    def rollout(
        self,
        input_conversations: cList,
        resp_start_with_tokens: list,
        reward_func: callable,
        reward_func_ground_truth=None,
        num_rollouts: int = 1,
        tiny_batch_size: int = 1,
        temperature: float = 1,
        top_p: float = 0.85,
        alpha_frequency: float = 0.2,
        alpha_presence: float = 0.2,
        alpha_decay: float = 0.9961,
        max_ctx: int = 1000,
        token_stop: list = [65535],
        token_ban: list = [0],
        begin_with_state: RWKVStates = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
        self.rwkv_policy.eval()
        upload_tokens = (
            input_conversations.to_tokens(self.tokenizer.encode)[0]
            + resp_start_with_tokens
            if input_conversations
            else resp_start_with_tokens if resp_start_with_tokens else [0]
        )

        assert num_rollouts >= tiny_batch_size and num_rollouts % tiny_batch_size == 0
        n_tiny_batch = num_rollouts // tiny_batch_size

        speak_tokens_batch, speak_text_batch, upload_tokens_batch = [], [], []
        for i in range(n_tiny_batch):
            upload_tokens_tiny_batch = [upload_tokens] * tiny_batch_size
            if begin_with_state is not None:
                state_tiny_batch = begin_with_state.duplicate(tiny_batch_size)
            else:
                state_tiny_batch = None
            speak_tokens_tiny_batch, speak_text_tiny_batch, _ = batch_chat(
                rwkv=self.rwkv_policy,
                start_with_tokens_batch=upload_tokens_tiny_batch,
                tokenizer=self.tokenizer,
                stop_with_tokens=token_stop,
                stop_supp_tokens=[],
                temp=temperature,
                top_p=top_p,
                presence_penalty=alpha_presence,
                frequency_penalty=alpha_frequency,
                decay_penalty=alpha_decay,
                batch_state=state_tiny_batch,
                max_resp_len=max_ctx,
                token_ban=token_ban,
            )

            speak_tokens_batch += speak_tokens_tiny_batch
            speak_text_batch += speak_text_tiny_batch
            upload_tokens_batch += upload_tokens_tiny_batch

        reward_func_ground_truth_batch = [reward_func_ground_truth] * num_rollouts
        req_text_batch = [input_conversations()] * num_rollouts
        for k, v in kwargs.items():
            kwargs[k] = [v] * num_rollouts

        # reward_func: callable(List[List[int]], List[List[int]], List[Any]) -> Torch.FloatTensor[num_rollouts, 1]
        reward_batch = reward_func(
            req_text_batch,
            speak_text_batch,
            upload_tokens_batch,
            speak_tokens_batch,
            reward_func_ground_truth_batch,
            **kwargs,
        ).to(device=next(self.rwkv_policy.parameters()).device)
        t_full_seq_batch = [
            torch.tensor(
                upload_t + speak_t,
                dtype=torch.long,
                device=next(self.rwkv_policy.parameters()).device,
            )
            for upload_t, speak_t in zip(upload_tokens_batch, speak_tokens_batch)
        ]
        action_mask_batch = [
            torch.zeros(
                len(upload_t) + len(speak_t),
                dtype=torch.bool,
                device=next(self.rwkv_policy.parameters()).device,
            )
            for upload_t, speak_t in zip(upload_tokens_batch, speak_tokens_batch)
        ]
        for i, mask in enumerate(action_mask_batch):
            mask[len(upload_tokens_batch[i]) :] = True
        assert t_full_seq_batch[0].shape == action_mask_batch[0].shape
        t_full_seq_batch = zero_pad_sequences(t_full_seq_batch)
        action_mask_batch = zero_pad_sequences(action_mask_batch)
        action_mask_batch = action_mask_batch[:, 1:]
        self.rwkv_policy.train()
        return t_full_seq_batch, reward_batch, action_mask_batch, speak_text_batch

    def sequence_log_probs_from_logits(
        self, logits: torch.tensor, output_ids: torch.tensor
    ) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

    def get_log_probs_ref(
        self,
        token_batch: torch.Tensor,
        state_idx_list=None,
    ) -> torch.Tensor:
        token_batch_list = token_batch.tolist()
        package = {
            "tokens_list": token_batch_list,
            "save_cache_dir": f"{cache_dir}/infer_batch_logits.ckpt",
            "state_idx_list": state_idx_list,
        }
        requests.post(
            self.ref_model_server + "/infer_batch",
            json=package,
        ).json()
        out_logits_batch = torch.load(f"{cache_dir}/infer_batch_logits.ckpt")["logits"]
        out_logits_batch = torch.tensor(
            out_logits_batch,
            device=next(self.rwkv_policy.parameters()).device,
            dtype=torch.float32,
        )

        log_probs = self.sequence_log_probs_from_logits(
            logits=out_logits_batch[:, :-1],
            output_ids=token_batch[:, 1:],
        )
        return log_probs

    def request_state_idx(self,load_dir=None):
        resp = requests.post(
            self.ref_model_server + "/regist_state_id",
            json={"load_dir": load_dir},
        ).json()
        self.state_idx = resp["access_token"]

    def update_ref_state(self, update_tokens, state_idx_list, to_state_idx):
        package = {
            "tokens_list": update_tokens,
            "state_idx_list": state_idx_list,
            "save_to_now_state_idx": to_state_idx,
        }
        requests.post(
            self.ref_model_server + "/infer_batch",
            json=package,
        ).json()

    def train_reward_model(self, t_full_seq_batch, reward_batch, action_mask_batch):
        pass

    @torch.no_grad()
    def act_episode(
        self,
        replay_buffer: ReplaySlidingWindow,
        input_conversations_batch: List[cList],
        resp_start_with_tokens_batch: List[list],
        ground_truth_batch,
        reward_func: callable,
        rlhf_func: callable,
        temperature: float = 1,
        top_p: float = 0.85,
        alpha_frequency: float = 0.2,
        alpha_presence: float = 0.2,
        alpha_decay: float = 0.9961,
        max_ctx: int = 1000,
        token_stop: list = [65535],
        token_ban: list = [0],
        num_rollouts: int = 1,
        tiny_batch_size: int = 1,
        train_batch_size: int = 1,
        begin_with_state_batch: List[RWKVStates] = None,
        **kwargs,
    ):

        reward_list = []
        for (
            i,
            (
                req_conversations,
                resp_start_with_tokens,
                ground_truth,
                begin_with_state,
            ),
        ) in enumerate(
            zip(
                input_conversations_batch,
                resp_start_with_tokens_batch,
                ground_truth_batch,
                begin_with_state_batch,
            )
        ):

            (
                t_full_seq_batch,
                reward_batch,
                action_mask_batch,
                speak_text_batch,
            ) = self.rollout(
                input_conversations=req_conversations,
                resp_start_with_tokens=resp_start_with_tokens,
                reward_func=reward_func,
                reward_func_ground_truth=ground_truth,
                num_rollouts=num_rollouts,
                tiny_batch_size=tiny_batch_size,
                temperature=temperature,
                top_p=top_p,
                alpha_frequency=alpha_frequency,
                alpha_presence=alpha_presence,
                alpha_decay=alpha_decay,
                max_ctx=max_ctx,
                token_stop=token_stop,
                token_ban=token_ban,
                begin_with_state=begin_with_state,
                **kwargs,
            )

            t_full_seq_batch, reward_batch, action_mask_batch, speak_text_batch = (
                rlhf_func(
                    t_full_seq_batch, reward_batch, action_mask_batch, speak_text_batch
                )
            )

            reward_list.append(reward_batch)
            advantages_batch = group_advantages(reward_batch)

            log_probs = get_batch_log_probs(
                rwkv=self.rwkv_policy,
                t_batch_tokens=t_full_seq_batch,
                begin_with_states=begin_with_state,
            )
            if begin_with_state is not None:
                self.request_state_idx()
                state_idx_list = [self.state_idx] * len(num_rollouts)
                self.update_ref_state(t_full_seq_batch, state_idx_list, self.state_idx)
            log_probs_ref = self.get_log_probs_ref(t_full_seq_batch, self.state_idx)
            kl = kl_div(log_probs, log_probs_ref, action_mask_batch)

            exp = ExperienceHist(
                history_tokens=t_full_seq_batch.cpu(),
                action_log_probs=log_probs.cpu(),
                log_probs_ref=log_probs_ref.cpu(),
                rewards=reward_batch.cpu(),
                advantages=advantages_batch.cpu(),
                action_mask=action_mask_batch.cpu(),
                kl=kl if kl is None else kl.cpu(),
                begin_with_states=(
                    begin_with_state.duplicate(num_rollouts)
                    if begin_with_state is not None
                    else None
                ),
            )
            replay_buffer.add(exp)

        gc.collect()
        torch.cuda.empty_cache()

        episode_reward_sum = torch.stack(reward_list).sum()
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=ExperienceHist.gather,
        )

        return episode_reward_sum, experience_sampler
