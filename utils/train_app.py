import os
from config import global_config

train_config = global_config.train_service_config

import gc
import math
import copy
import torch
import deepspeed
from utils.message_manager import cList, Conversation
import requests
import json
from config import RWKV
from config import RWKVStates
import torch.nn.functional as F
from RWKV.functions import (
    train_forward,
    train_forward_from_embds,
    speak,
    ppl,
    speak_next_token,
    calc_cross_entropy_loss,
    clear_gpu_memory,
)
from utils.rl.grpo.functions import (
    zero_pad_sequences,
)
# from RWKV.multimodal_functions import (
#     voice_encode_and_adapt,
# )
from torch.utils.data import Dataset, DataLoader
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import random
import wandb
import sys
import psutil

from utils.functions import pad_and_batch
from utils.dataset.dataset import (
    MultimodalDataset,
    read_bin_wav,
    RLGroupDatasetOnline,
    SFTDatasetOnline,
)
from utils.dataset.dataset_functions import (
    UnitStreamProcessor,
    TraversalDataloader,
    MyDataloader,
    EpochSampleDataloader,
    rl_collate_fn,
)
from utils.rl.grpo.train import GRPOTrainer
from utils.rl.grpo.replay import ReplaySlidingWindow, ExperienceHist
from utils.rl.grpo.functions import get_batch_log_probs, group_advantages
from utils.rl.grpo.loss import GRPOLoss, kl_div
from torch.nn.utils import clip_grad_norm_

# from config import vocoder

from typing import List
from functools import partial
import argparse

from deepspeed import comm as dist

deepspeed.init_distributed()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload_model_dir", type=str, help="重新加载训练模型位置")
    parser.add_argument("--lr_init", type=float, help="初始学习率")
    parser.add_argument("--lr_final", type=float, help="最终学习率")
    parser.add_argument("--warmup_steps", type=int, help="warmup步数")
    args = parser.parse_known_args()[0]
    return args


class OnlineTrainingAPP:
    def __init__(self):
        self.args = train_config
        cmd_args = get_args()
        if cmd_args.reload_model_dir:
            print(f"overwrite model dir: {cmd_args.reload_model_dir}")
            train_config.model.load_model = cmd_args.reload_model_dir
        if cmd_args.lr_init:
            print(f"overwrite lr_init: {cmd_args.lr_init}")
            self.args.train.lr_init = float(cmd_args.lr_init)
        if cmd_args.lr_final:
            print(f"overwrite lr_final: {cmd_args.lr_final}")
            self.args.train.lr_final = float(cmd_args.lr_final)
        if cmd_args.warmup_steps:
            print(f"overwrite warmup_steps: {cmd_args.warmup_steps}")
            self.args.train.warmup_steps = int(cmd_args.warmup_steps)
        print(f"load model from: {train_config.model.load_model}")
        self.model = RWKV(self.args, global_config.voice_on)
        self.model_engine = self.build_engine(
            lr_init=self.args.train.lr_init,
            lr_final=self.args.train.lr_final,
            warmup_steps=self.args.train.warmup_steps,
        )
        self.infer_tokenizer = global_config.tokenizer_eval
        self.train_tokenizer = global_config.tokenizer_train

        self.train_state = None  # 训练时的滚动state
        self.online_lr_state = None  # 在线学习state

        self.rank = dist.get_rank()
        self.total_gpus = dist.get_world_size()

        if global_config.wandb_proj and self.rank == 0:
            wandb.init(project=global_config.wandb_proj)

        # if global_config.voice_on:
        #     self.feat_extractor = torch.load(
        #         train_config.vocoder.load_feature_extractor_dir
        #     )
        #     self.voice_losses = vocoder.Losses(train_config)
        #     self.discriminator = vocoder.VocoderDiscriminator(train_config.vocoder)
        #     self.discriminator_engine, _ = self.discriminator.build_engine()
        #     self.voice_unit_len = (
        #         self.args.vocoder.head.hop_length * self.args.vocoder.adapter.chunk_len
        #     )

    def load_model(
        self,
        ckpt_dir: str,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
    ):
        print(f"从{ckpt_dir}重新读取模型...")
        if hasattr(self, "model_engine"):
            current_process = psutil.Process()
            parent_process = current_process.parent().parent()

            cmd_line = parent_process.cmdline()
            ds_idx, ds = next(
                (
                    (idx, item)
                    for idx, item in enumerate(cmd_line)
                    if item.endswith("/deepspeed")
                ),
                None,
            )
            cmd = ["deepspeed"] + parent_process.cmdline()[ds_idx + 1 :]
            cmd += [
                "--reload_model_dir",
                ckpt_dir,
                "--lr_init",
                str(lr_init) if lr_init else str(self.args.train.lr_init),
                "--lr_final",
                str(lr_final) if lr_final else str(self.args.train.lr_final),
                "--warmup_steps",
                (
                    str(warmup_steps)
                    if warmup_steps
                    else str(self.args.train.warmup_steps)
                ),
            ]
            print("====================================================")
            print("run", " ".join(cmd))
            print("====================================================")
            os.execv(ds, cmd)

    def save_weight(
        self, name: str, save_train_state: bool = False, folder: str = None
    ):
        folder = folder if folder else global_config.ckpt_dir
        fpath = f"{folder}/{name}.pth"
        state_path = f"{folder}/{name}.state"
        self.model.load_state_dict(self.model_engine.module.state_dict())
        torch.save(self.model.state_dict(), fpath)
        if save_train_state and self.train_state is not None:
            torch.save(self.train_state, state_path)
        clear_gpu_memory(force=True)
        return fpath

    # def save_vocoder(self, name: str):
    #     fpath_gen = f"{global_config.ckpt_dir}/{name}_vocoder_gen.pth"
    #     temp_state_dict = {
    #         k.replace("module.", ""): v
    #         for k, v in self.vocoder_engine.gen_engine.state_dict().items()
    #     }
    #     self.vocoder_engine.model.generator.load_state_dict(temp_state_dict)
    #     torch.save(self.vocoder_engine.model.generator.state_dict(), fpath_gen)
    #     fpath_disc = f"{global_config.ckpt_dir}/{name}_vocoder_disc.pth"
    #     if train_config.vocoder.GAN_on:
    #         temp_state_dict = {
    #             k.replace("module.", ""): v
    #             for k, v in self.vocoder_engine.disc_engine.state_dict().items()
    #         }
    #         self.vocoder_engine.model.discriminators.load_state_dict(temp_state_dict)
    #         torch.save(
    #             self.vocoder_engine.model.discriminators.state_dict(), fpath_disc
    #         )
    #     clear_gpu_memory(force=True)

    def train_text_from_messages(
        self,
        messages: List[cList],
        batch_size: int = 1,
        n_save_ckpt: int = -1,
        min_loss: float = None,
        max_loss: float = None,
        min_loss_fix: float = None,
        max_loss_fix: float = None,
        multi_scale_ctx: int = None,
        multi_scale_alpha: float = None,
        keep_train_states: bool = False,
        use_ego_mask: bool = False,
        ignore_ctx: bool = False,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
    ):
        if lr_init:
            self.build_engine(lr_init, lr_final, warmup_steps)
        assert batch_size % self.total_gpus == 0
        dp_chunk_len = batch_size // self.total_gpus
        """
        单batch文本训练
        """
        min_loss = self.args.train.min_loss if min_loss is None else min_loss
        max_loss = self.args.train.max_loss if max_loss is None else max_loss
        min_loss_fix = (
            self.args.train.min_loss_fix if min_loss_fix is None else min_loss_fix
        )
        max_loss_fix = (
            self.args.train.max_loss_fix if max_loss_fix is None else max_loss_fix
        )
        multi_scale_ctx = (
            self.args.model.ctx_len if multi_scale_ctx is None else multi_scale_ctx
        )
        multi_scale_alpha = (
            self.args.train.multi_scale_alpha
            if multi_scale_alpha is None
            else multi_scale_alpha
        )
        assert multi_scale_ctx * multi_scale_alpha > 2
        assert 0 < multi_scale_alpha <= 1
        self.model_engine.train()
        all_tokens = []
        all_masks = []
        losses = []
        for clist in messages:
            tokens, masks = clist.to_tokens(
                self.train_tokenizer.encode, use_ego_mask=use_ego_mask
            )
            all_tokens += tokens
            all_masks += masks
            all_tokens += [0]
            all_masks += [0]
        all_tokens = pad_and_batch(all_tokens, batch_size)
        all_masks = pad_and_batch(all_masks, batch_size)
        dp_tokens = all_tokens[
            self.rank * dp_chunk_len : (self.rank + 1) * dp_chunk_len
        ]
        dp_masks = all_masks[self.rank * dp_chunk_len : (self.rank + 1) * dp_chunk_len]
        assert len(dp_masks) == len(dp_tokens)
        for step, (mean_loss, train_tokens) in enumerate(
            self.learn_tokens(
                tokens=dp_tokens,
                masks=dp_masks,
                min_loss=min_loss,
                min_loss_fix=min_loss_fix,
                max_loss=max_loss,
                max_loss_fix=max_loss_fix,
                states=None,
                multi_scale_ctx=multi_scale_ctx,
                multi_scale_alpha=multi_scale_alpha,
                keep_train_states=keep_train_states,
                ignore_ctx=ignore_ctx,
            ),
            1,
        ):
            # for b in range(batch_size):
            #     while 0 in train_tokens[b]:
            #         train_tokens[b].remove(0)
            print(f"gpu{self.rank}: mean-loss->{mean_loss}")
            print(
                f"gpu{self.rank}->{self.infer_tokenizer.decode(train_tokens)[self.rank][:45]}"
            )
            clear_gpu_memory(force=True)
            if self.rank == 0:
                if global_config.wandb_proj:
                    wandb.log({"text_loss": mean_loss})
                if n_save_ckpt > 0 and step % n_save_ckpt == 0:
                    print(f"====save at step={step}====")
                    self.save_weight(f"train_folder_step={step}")
            losses.append(mean_loss)
        mean_loss = sum(losses) / len(losses)
        return mean_loss

    def learn_tokens(
        self,
        tokens: list,
        masks: list,
        min_loss: float = None,
        max_loss: float = None,
        min_loss_fix: float = None,
        max_loss_fix: float = None,
        states: RWKVStates = None,
        multi_scale_ctx: int = None,
        multi_scale_alpha: float = 1,
        keep_train_states: bool = False,
        ignore_ctx: bool = False,
        return_left_token: bool = False,
    ):
        """
        文本训练
        """
        min_loss = self.args.train.min_loss if min_loss is None else min_loss
        max_loss = self.args.train.max_loss if max_loss is None else max_loss
        min_loss_fix = (
            self.args.train.min_loss_fix if min_loss_fix is None else min_loss_fix
        )
        max_loss_fix = (
            self.args.train.max_loss_fix if max_loss_fix is None else max_loss_fix
        )
        multi_scale_ctx = (
            self.args.model.ctx_len if multi_scale_ctx is None else multi_scale_ctx
        )
        multi_scale_alpha = (
            self.args.train.multi_scale_alpha
            if multi_scale_alpha is None
            else multi_scale_alpha
        )
        self.model_engine.train()
        assert multi_scale_ctx * multi_scale_alpha > 2
        assert 0 < multi_scale_alpha <= 1
        assert len(tokens) != 0
        total = 0
        mean_loss = 0
        i = 0

        tokens = torch.tensor(tokens, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.float32)
        assert tokens.shape[1] != 0
        while tokens.shape[1] > 0:
            i += 1
            ctx_len = (
                random.randint(
                    int(multi_scale_ctx * multi_scale_alpha),
                    multi_scale_ctx,
                )
                if not ignore_ctx
                else 99999999999999999
            )

            output = tokens[:, :ctx_len]
            output_masks = masks[:, :ctx_len]
            tokens = tokens[:, ctx_len - 1 :]
            masks = masks[:, ctx_len - 1 :]
            if not keep_train_states:
                states = None
            batch_tokens = copy.deepcopy(output).to(
                next(self.model_engine.parameters()).device
            )
            batch_masks = copy.deepcopy(output_masks).to(
                next(self.model_engine.parameters()).device
            )
            print(batch_tokens.shape, batch_masks.shape)
            m, states = train_forward(
                self.model_engine, batch_tokens, batch_masks, states
            )
            self.train_state = states
            loss = m.item()
            print(f"loss={m}")
            if loss < min_loss:
                m = m * min_loss_fix
                print(f"(<min)fixed_loss:{m}")
            elif loss > max_loss:
                print(f"(>max)before_fixed_loss:{m}")
                m = m * max_loss_fix
                print(f"(>max)fixed_loss:{m}")
            self.model_engine.backward(m)
            self.model_engine.step()
            total += loss
            mean_loss = total / i
            if tokens.shape[1] == 0:
                break
            if return_left_token:
                yield mean_loss, output, tokens.shape[1]
            else:
                yield mean_loss, output


    def train_from_folder(
        self,
        forder_dir: str,
        epoch: int,
        batch_size_per_gpu: int = 1,
        n_save_ckpt: int = 1,
        multi_scale_ctx: int = None,
        multi_scale_alpha: float = None,
        min_loss: float = None,
        max_loss: float = None,
        min_loss_fix: float = None,
        max_loss_fix: float = None,
        n_save_step: int = None,
        keep_states_mode: str = "never",
        dataloader_workers_per_gpu: int = 2,
        begin_with_state_dir=None,
        use_qa_mask: bool = False,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
        chunk_len: int = 2,
    ):
        if lr_init:
            self.build_engine(lr_init, lr_final, warmup_steps)
        min_loss = self.args.train.min_loss if min_loss is None else min_loss
        max_loss = self.args.train.max_loss if max_loss is None else max_loss
        min_loss_fix = (
            self.args.train.min_loss_fix if min_loss_fix is None else min_loss_fix
        )
        max_loss_fix = (
            self.args.train.max_loss_fix if max_loss_fix is None else max_loss_fix
        )

        multi_scale_ctx = (
            self.args.model.ctx_len if multi_scale_ctx is None else multi_scale_ctx
        )
        multi_scale_alpha = (
            self.args.train.multi_scale_alpha
            if multi_scale_alpha is None
            else multi_scale_alpha
        )
        assert multi_scale_ctx * multi_scale_alpha > 2
        assert 0 < multi_scale_alpha <= 1
        self.model_engine.train()

        assert keep_states_mode in ["never", "step", "epoch"]

        self.train_state = (
            None if begin_with_state_dir is None else torch.load(begin_with_state_dir)
        )
        # check batch state
        if self.train_state is not None:
            n_state_batch = self.train_state.tmix_shift_states.shape[1]
            total_batch = batch_size_per_gpu * self.total_gpus
            if total_batch > n_state_batch:
                alpha = (n_state_batch + total_batch - 1) // n_state_batch
                self.train_state = self.train_state.duplicate(alpha)[:n_state_batch]
            elif total_batch < n_state_batch:
                print(
                    "警告: 训练的batch数量小于读取state的batch数量, 可能会导致信息损失。"
                )
                self.train_state = self.train_state[:total_batch]

        total_text_loss = []

        dataset = MultimodalDataset(
            dataset_dir=forder_dir,
            tokenizer=self.train_tokenizer,
            voice_read_func=None if not global_config.voice_on else read_bin_wav,
            video_load_func=None,
            ctx_len=multi_scale_ctx,
            qa_mask_on=use_qa_mask,
        )

        dataloader = TraversalDataloader(
            dataset=dataset,
            batch_size=batch_size_per_gpu,
            num_workers=dataloader_workers_per_gpu,
            multi_scale_alpha=multi_scale_alpha,
        )

        stream_processor = UnitStreamProcessor(train_config)

        for e in range(epoch):
            print(f"====gpu:{self.rank},train epoch:{e}====")
            if keep_states_mode == "step":
                self.train_state = None
            for step, (batch_units, batch_masks) in enumerate(dataloader):
                unit_dicts = stream_processor.encode(
                    self.model_engine,
                    # self.feat_extractor if global_config.voice_on else None,
                    None,
                    batch_units,
                    # voice_encode_and_adapt,
                    None,
                    device=next(self.model_engine.parameters()).device,
                    dtype=next(self.model_engine.parameters()).dtype,
                )
                dist.barrier()
                embds = unit_dicts["main"]
                token_targets = torch.tensor(
                    unit_dicts["tokens_target"],
                    dtype=torch.long,
                    device=next(self.model_engine.parameters()).device,
                )
                token_masks = (token_targets != 0).long()
                batch_masks = torch.tensor(
                    batch_masks,
                    device=next(self.model_engine.parameters()).device,
                    dtype=torch.float32,
                )[:, 1:]
                mask_targets = token_masks * batch_masks

                if chunk_len > 1:
                    n_ctx = embds.shape[1]
                    if (n_ctx - 1) % chunk_len != 0:
                        n_supp = (n_ctx - 1) % chunk_len
                        embds = torch.cat(
                            [
                                embds,
                                torch.zeros(
                                    (
                                        embds.shape[0],
                                        chunk_len - n_supp,
                                        embds.shape[2],
                                    ),
                                    device=next(self.model_engine.parameters()).device,
                                    dtype=next(self.model_engine.parameters()).dtype,
                                ),
                            ],
                            dim=1,
                        )
                        token_targets = torch.cat(
                            [
                                token_targets,
                                torch.zeros(
                                    (token_targets.shape[0], chunk_len - n_supp),
                                    device=next(self.model_engine.parameters()).device,
                                    dtype=torch.long,
                                ),
                            ],
                            dim=1,
                        )
                        mask_targets = torch.cat(
                            [
                                mask_targets,
                                torch.zeros(
                                    (mask_targets.shape[0], chunk_len - n_supp),
                                    device=next(self.model_engine.parameters()).device,
                                    dtype=torch.float32,
                                ),
                            ],
                            dim=1,
                        )

                if keep_states_mode == "never":
                    self.train_state = None
                out_latent, out_logits, self.train_state = (
                    self.model_engine.forward_from_embeddings(
                        embds[:, :-1, :], self.train_state
                    )
                )
                text_loss = calc_cross_entropy_loss(
                    out_logits, token_targets, mask_targets
                )
                text_loss_item = text_loss.item()
                if text_loss < min_loss:
                    text_loss = text_loss * min_loss_fix
                    print(f"(<min)fixed_loss:{text_loss}")
                elif text_loss > max_loss:
                    print(f"(>max)before_fixed_loss:{text_loss}")
                    text_loss = text_loss * max_loss_fix
                    print(f"(>max)fixed_loss:{text_loss}")
                # TODO: 音频loss
                #
                # ===============
                m = text_loss
                self.model_engine.zero_grad()
                self.model_engine.backward(m)
                self.model_engine.step()
                total_text_loss.append(text_loss_item)
                total_text_loss = total_text_loss[-100:]
                mean_text_loss = sum(total_text_loss) / len(total_text_loss)

                yield json.dumps(
                    {
                        "epoch": e,
                        "step": step,
                        "mean_text_loss": mean_text_loss,
                        "text_loss": text_loss_item,
                        "n_tokens": dataloader.n_dataset_ctx,
                        "left_tokens": dataloader.n_dataset_ctx
                        - dataloader.current_ctx,
                    },
                    ensure_ascii=False,
                ) + "\n"
                print(
                    f"gpu{self.rank}: mean-text-loss->{mean_text_loss} | now-text-loss->{text_loss_item}"
                )

                dist.barrier()
                clear_gpu_memory(force=True)
                if self.rank == 0:
                    if global_config.wandb_proj:
                        wandb.log({"mean_text_loss": mean_text_loss})

                    if n_save_step and step % n_save_step == 0:
                        print(f"====epoch: {e},save at step={step}====")
                        self.save_weight(f"train_single_folder_epoch={e}_step={step}")

            if e % n_save_ckpt == 0 and self.rank == 0:
                svpath = self.save_weight(
                    f"train_single_folder_epoch={e}", save_train_state=True
                )
                yield json.dumps(
                    {
                        "over": False,
                        "to_dir": svpath,
                    },
                    ensure_ascii=False,
                ) + "\n"
                print(f"====save at epoch={e}====")
        yield json.dumps(
            {
                "over": True,
                "to_dir": svpath,
            },
            ensure_ascii=False,
        ) + "\n"

    def train_from_folders(
        self,
        folder_weight_dir_list,
        epoch: int,
        batch_size_per_gpu: int = 1,
        n_save_ckpt: int = 1,
        min_loss: float = None,
        max_loss: float = None,
        min_loss_fix: float = None,
        max_loss_fix: float = None,
        n_save_step: int = None,
        dataloader_workers_per_gpu: int = 2,
        use_qa_mask: bool = False,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
    ):
        if lr_init:
            self.build_engine(lr_init, lr_final, warmup_steps)
        min_loss = self.args.train.min_loss if min_loss is None else min_loss
        max_loss = self.args.train.max_loss if max_loss is None else max_loss
        min_loss_fix = (
            self.args.train.min_loss_fix if min_loss_fix is None else min_loss_fix
        )
        max_loss_fix = (
            self.args.train.max_loss_fix if max_loss_fix is None else max_loss_fix
        )

        self.model_engine.train()

        total_text_loss = []

        dataset_folder_list = [
            folder_dir for folder_dir, n_sample_lines in folder_weight_dir_list
        ]
        n_sample_list = [
            n_sample_lines for folder_dir, n_sample_lines in folder_weight_dir_list
        ]

        epoch_sample_dataloader = EpochSampleDataloader(
            dataset_folder_list,
            n_sample_list,
            batch_size_per_gpu,
            num_workers=dataloader_workers_per_gpu,
            tokenizer=self.train_tokenizer,
            voice_read_func=None if not global_config.voice_on else read_bin_wav,
            video_load_func=None,
            ctx_len=self.args.model.ctx_len,
            total_epoch=epoch,
            use_qa_mask=use_qa_mask,
        )
        stream_processor = UnitStreamProcessor(train_config)

        for e, (epoch_units, epoch_masks) in enumerate(epoch_sample_dataloader):
            print(f"====gpu:{self.rank},train epoch:{e}====")
            n_data = len(epoch_units)
            for step, (batch_units, batch_masks) in enumerate(
                zip(epoch_units, epoch_masks)
            ):
                unit_dicts = stream_processor.encode(
                    self.model_engine,
                    # self.feat_extractor if global_config.voice_on else None,
                    None,
                    batch_units,
                    # voice_encode_and_adapt,
                    None,
                    device=next(self.model_engine.parameters()).device,
                    dtype=next(self.model_engine.parameters()).dtype,
                )
                dist.barrier()
                embds = unit_dicts["main"]
                token_targets = torch.tensor(
                    unit_dicts["tokens_target"],
                    dtype=torch.long,
                    device=next(self.model_engine.parameters()).device,
                )
                token_masks = (token_targets != 0).long()
                batch_masks = torch.tensor(
                    batch_masks,
                    device=next(self.model_engine.parameters()).device,
                    dtype=torch.float32,
                )[:, 1:]
                mask_targets = token_masks * batch_masks
                out_latent, out_logits, self.train_state = (
                    self.model_engine.forward_from_embeddings(
                        embds[:, :-1, :], self.train_state
                    )
                )
                text_loss = calc_cross_entropy_loss(
                    out_logits, token_targets, mask_targets
                )
                text_loss_item = text_loss.item()
                if text_loss < min_loss:
                    text_loss = text_loss * min_loss_fix
                    print(f"(<min)fixed_loss:{text_loss}")
                elif text_loss > max_loss:
                    print(f"(>max)before_fixed_loss:{text_loss}")
                    text_loss = text_loss * max_loss_fix
                    print(f"(>max)fixed_loss:{text_loss}")
                # TODO: 音频loss
                #
                # ===============
                m = text_loss
                self.model_engine.zero_grad()
                self.model_engine.backward(m)
                self.model_engine.step()
                total_text_loss.append(text_loss_item)
                total_text_loss = total_text_loss[-100:]
                mean_text_loss = sum(total_text_loss) / len(total_text_loss)

                yield json.dumps(
                    {
                        "epoch": e,
                        "step": step,
                        "mean_text_loss": mean_text_loss,
                        "text_loss": text_loss_item,
                        "n_data": n_data,
                        "left_data": n_data - step - 1,
                    },
                    ensure_ascii=False,
                ) + "\n"
                print(
                    f"gpu{self.rank}: mean-text-loss->{mean_text_loss} | now-text-loss->{text_loss_item}"
                )

                dist.barrier()
                clear_gpu_memory(force=True)
                if self.rank == 0:
                    if global_config.wandb_proj:
                        wandb.log({"mean_text_loss": mean_text_loss})

                    if n_save_step and step % n_save_step == 0:
                        print(f"====epoch: {e},save at step={step}====")
                        self.save_weight(f"train_single_folder_epoch={e}_step={step}")

                if step >= n_data - 1:
                    break
            if e % n_save_ckpt == 0 and self.rank == 0:
                svpath = self.save_weight(
                    f"train_single_folder_epoch={e}", save_train_state=True
                )
                yield json.dumps(
                    {
                        "over": False,
                        "to_dir": svpath,
                    },
                    ensure_ascii=False,
                ) + "\n"
                print(f"====save at epoch={e}====")
        yield json.dumps(
            {
                "over": True,
                "to_dir": svpath,
            },
            ensure_ascii=False,
        ) + "\n"

    def _get_batch_logps(
        self,
        logits,
        labels: torch.LongTensor,
        input_mask=None,
        average_log_prob: bool = True,
    ):
        logits = logits.to(
            device=next(self.model_engine.parameters()).device,
            dtype=self.model.args.dtype,
        )
        labels = labels.to(
            device=next(self.model_engine.parameters()).device,
        )
        if input_mask is not None:
            input_mask = input_mask.to(
                device=next(self.model_engine.parameters()).device,
            )

        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != 0

        # dummy token; we'll ignore the losses on these tokens later
        if input_mask is not None:
            mask = input_mask[:, 1:]
            loss_mask[mask == 0] = False
            labels[mask == 0] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def build_engine(self, lr_init, lr_final, warmup_steps):
        if hasattr(self, "model_engine"):
            print("调整引擎调度器参数...")
            print(
                f"lr_init={lr_init}, lr_final={lr_final}, warmup_steps={warmup_steps}"
            )
            for param_group in self.model_engine.optimizer.param_groups:
                param_group["lr"] = lr_init
            self.lr_scheduler.warmup_min_lr = lr_init  # 更新学习率调度器的初始学习率
            self.lr_scheduler.warmup_max_lr = lr_final  # 更新学习率调度器的最终学习率
            self.lr_scheduler.warmup_num_steps = (
                warmup_steps  # 更新学习率调度器的预热步数
            )
            if hasattr(self.lr_scheduler, "last_epoch"):
                self.lr_scheduler.last_epoch = -1  # 重置调度器状态
            if hasattr(self.lr_scheduler, "min_lrs"):
                for i in range(len(self.lr_scheduler.min_lrs)):
                    self.lr_scheduler.min_lrs[i] = lr_init
            if hasattr(self.lr_scheduler, "max_lrs"):
                for i in range(len(self.lr_scheduler.max_lrs)):
                    self.lr_scheduler.max_lrs[i] = lr_final
            self.lr_scheduler.step()
            for attr, value in vars(self.lr_scheduler).items():
                print(f"{attr}: {value}")
            print("=========================================")
            return self.model_engine
        else:
            ds_config = {
                "bfloat16": {"enabled": "auto"},
                "gradient_accumulation_steps": self.args.deepspeed.gradient_accumulation_steps,
                "gradient_clipping": self.args.train.grad_cp,
                "train_micro_batch_size_per_gpu": 1,
            }
            if train_config.deepspeed.zero:
                ds_config["zero_optimization"] = {
                    "stage": train_config.deepspeed.ds_stage,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e6,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e6,
                    "contiguous_gradients": True,
                }

                if train_config.deepspeed.offload_optimizer:
                    ds_config["zero_optimization"]["offload_optimizer"] = {
                        "device": "cpu",
                        "pin_memory": True,
                    }
                if (
                    train_config.deepspeed.offload_param_stage3
                    and train_config.deepspeed.ds_stage == 3
                ):
                    ds_config["zero_optimization"]["offload_param"] = {
                        "device": "cpu",
                        "pin_memory": True,
                    }

            self.optimizer = (
                DeepSpeedCPUAdam(
                    self.model.get_optim_groups(),
                    lr=lr_init,
                    betas=(self.args.train.beta1, self.args.train.beta2),
                    eps=self.args.train.adam_eps,
                    adamw_mode=self.args.train.adamw_mode,
                    weight_decay=self.args.train.weight_decay,
                    amsgrad=False,
                    bias_correction=True,
                )
                if train_config.deepspeed.zero
                and train_config.deepspeed.offload_optimizer
                else FusedAdam(
                    self.model.get_optim_groups(),
                    lr=lr_init,
                    betas=(self.args.train.beta1, self.args.train.beta2),
                    eps=self.args.train.adam_eps,
                    bias_correction=True,
                    adam_w_mode=self.args.train.adamw_mode,
                    weight_decay=self.args.train.weight_decay,
                    amsgrad=False,
                )
            )

            self.lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                self.optimizer,
                warmup_min_lr=lr_init,
                warmup_max_lr=lr_final,
                warmup_num_steps=warmup_steps,
                warmup_type="linear",
            )

            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                config=ds_config,
            )
            print("cuda available", torch.cuda.device_count())
        return self.model_engine

    def train_grpo(
        self,
        rl_dataset: Dataset,
        ref_model_server: str,
        reward_func: callable,
        rlhf_func: callable,
        n_epoch: int = 1,
        n_rollout_questions: int = 1,
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
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
        n_save_ckpt: int = 1,
        n_save_episode_ckpt: int = 5,
        n_replay_sliding_window: int = 0,
        clear_replay_on_episode: bool = True,
        n_train_each_episode: int = 1,
        train_batch_size: int = 1,
        clip_eps: float = 0.2,
        kl_weight: float = 0.01,
        grad_cp_max_norm: float = 1.0,
        accumulate_grad: bool = True,
    ):
        grpo_trainer = GRPOTrainer(
            self.model_engine, ref_model_server, tokenizer=self.train_tokenizer
        )
        grpo_loss = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)
        if lr_init:
            self.build_engine(lr_init, lr_final, warmup_steps)
        replay_buffer = ReplaySlidingWindow(n_replay_sliding_window)

        dataloader = DataLoader(
            rl_dataset,
            batch_size=n_rollout_questions,
            shuffle=True,
            collate_fn=rl_collate_fn,
        )
        for epoch in range(n_epoch):
            for episode, (
                input_conversations_batch,
                resp_start_with_tokens_batch,
                cleaned_answer_batch,
                ground_truth_batch,
                begin_with_state_batch,
                kwargs_batch,
            ) in enumerate(dataloader):
                if clear_replay_on_episode:
                    replay_buffer.clear()
                    clear_gpu_memory(force=True)
                episode_reward_sum, experience_sampler = grpo_trainer.act_episode(
                    replay_buffer=replay_buffer,
                    input_conversations_batch=input_conversations_batch,
                    resp_start_with_tokens_batch=resp_start_with_tokens_batch,
                    ground_truth_batch=ground_truth_batch,
                    reward_func=reward_func,
                    rlhf_func=rlhf_func,
                    temperature=temperature,
                    top_p=top_p,
                    alpha_frequency=alpha_frequency,
                    alpha_presence=alpha_presence,
                    alpha_decay=alpha_decay,
                    max_ctx=max_ctx,
                    token_stop=token_stop,
                    token_ban=token_ban,
                    begin_with_state_batch=begin_with_state_batch,
                    num_rollouts=num_rollouts,
                    tiny_batch_size=tiny_batch_size,
                    train_batch_size=train_batch_size,
                    **kwargs_batch,
                )
                self.model_engine.train()

                print(f"episode {episode}: sum reward->{episode_reward_sum}")
                print("======================================================")
                for step in range(n_train_each_episode):
                    for exp in experience_sampler:
                        exp: ExperienceHist
                        if accumulate_grad:
                            is_fault = False
                            loss = 0
                            kl = 0
                            self.model_engine.zero_grad()
                            for rb in range(num_rollouts):
                                (
                                    history_tokens,
                                    action_log_probs,
                                    log_probs_ref,
                                    rewards,
                                    advantages,
                                    action_mask,
                                ) = (
                                    exp.history_tokens[rb : rb + 1],
                                    exp.action_log_probs[rb : rb + 1],
                                    exp.log_probs_ref[rb : rb + 1],
                                    exp.rewards[rb : rb + 1],
                                    exp.advantages[rb : rb + 1],
                                    exp.action_mask[rb : rb + 1],
                                )
                                train_hist = ExperienceHist(
                                    history_tokens=history_tokens,
                                    action_log_probs=action_log_probs,
                                    log_probs_ref=log_probs_ref,
                                    rewards=rewards,
                                    advantages=advantages,
                                    action_mask=action_mask,
                                )
                                train_hist = train_hist.to(
                                    device=next(self.model_engine.parameters()).device
                                )
                                step_loss, step_kl = grpo_loss(
                                    rwkv=self.model_engine,
                                    experience_hist=train_hist,
                                )
                                loss += step_loss
                                kl += step_kl
                                if not step_loss.isfinite():
                                    print(
                                        f"WARNING: loss is infinite, skip this batch, loss={loss}"
                                    )
                                    is_fault = True
                                    break
                                self.model_engine.backward(
                                    step_loss / num_rollouts,
                                    retain_graph=(
                                        True if rb < num_rollouts - 1 else False
                                    ),
                                )
                                print(f"batch: {rb} loss: {step_loss}")
                                train_hist = train_hist.to("cpu")
                                del train_hist
                                clear_gpu_memory(force=True)
                            if is_fault:
                                del train_hist
                                self.model_engine.zero_grad()
                                clear_gpu_memory(force=True)
                                continue
                            loss = loss / num_rollouts
                            kl = kl / num_rollouts
                            grad_norm = clip_grad_norm_(
                                self.model_engine.parameters(),
                                max_norm=grad_cp_max_norm,
                            )
                            self.model_engine.step()

                        else:
                            exp = exp.to(next(self.model_engine.parameters()).device)

                            loss, kl = grpo_loss(self.model_engine, exp)
                            if not loss.isfinite():
                                print(
                                    f"Warning: Loss not finite, skipping backward, loss={loss}"
                                )
                                continue

                            self.model_engine.zero_grad()
                            self.model_engine.backward(loss)
                            grad_norm = clip_grad_norm_(
                                self.model_engine.parameters(),
                                max_norm=grad_cp_max_norm,
                            )

                            print(
                                f"rl step={step}: loss={loss}, advantage={exp.advantages}"
                            )
                            print(
                                f"rl step={step}: kl={kl: .4f}, grad_norm={grad_norm.item()}"
                            )
                            self.model_engine.step()
                            exp = exp.to("cpu")
                        clear_gpu_memory(force=True)
                        # yield (
                        #     json.dumps(
                        #         {
                        #             "epoch": epoch,
                        #             "step": step,
                        #             "loss": loss.item(),
                        #             "kl": kl.item(),
                        #             "sum_rewards": episode_reward_sum.item(),
                        #             "grad_norm": grad_norm.item(),
                        #         },
                        #         ensure_ascii=False,
                        #     )
                        #     + "\n"
                        # )

                if episode % n_save_episode_ckpt == 0 and self.rank == 0:
                    svpath = self.save_weight(
                        f"train_grpo_episode={episode}", save_train_state=True
                    )
                    # yield (
                    #     json.dumps(
                    #         {
                    #             "over": False,
                    #             "to_dir": svpath,
                    #         },
                    #         ensure_ascii=False,
                    #     )
                    #     + "\n"
                    # )

            if epoch % n_save_ckpt == 0 and self.rank == 0:
                svpath = self.save_weight(
                    f"train_grpo_epoch={epoch}", save_train_state=True
                )
                # yield (
                #     json.dumps(
                #         {
                #             "over": False,
                #             "to_dir": svpath,
                #         },
                #         ensure_ascii=False,
                #     )
                #     + "\n"
                # )

    def train_grpo_from_group_dataset(
        self,
        group_dataset: Dataset,
        ref_model_server: str,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
        n_save_episode_ckpt: int = 5,
        n_replay_sliding_window: int = 0,
        clear_replay_on_episode: bool = True,
        n_train_each_episode: int = 1,
        train_batch_size: int = 1,
        clip_eps: float = 0.2,
        kl_weight: float = 0.01,
        grad_cp_max_norm: float = 1.0,
        accumulate_grad: bool = True,
        continuous_history: bool = False,
    ):
        self.model_engine.train()
        grpo_trainer = GRPOTrainer(
            self.model_engine, ref_model_server, tokenizer=self.train_tokenizer
        )
        grpo_loss = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)
        if lr_init:
            self.build_engine(lr_init, lr_final, warmup_steps)
        replay_buffer = ReplaySlidingWindow(n_replay_sliding_window)

        for i_file, datas in enumerate(group_dataset):
            for episode, lines in enumerate(datas):
                reward_list = []
                if clear_replay_on_episode:
                    replay_buffer.clear()
                    clear_gpu_memory(force=True)
                update_units_list = []
                now_state = None
                grpo_trainer.request_state_idx()
                with torch.no_grad():
                    for t, turn in enumerate(lines):
                        if t > 0 and continuous_history:
                            _, now_state = self.model_engine(
                                update_units_list[t - 1 : t], now_state
                            )
                            grpo_trainer.update_ref_state(
                                update_units_list[t - 1 : t],
                                [grpo_trainer.state_idx],
                                grpo_trainer.state_idx,
                            )
                        turn_units = []
                        turn_masks = []
                        turn_rewards = []
                        for n, choice_dict in enumerate(turn):
                            choice_units = choice_dict["units"]
                            choice_masks = choice_dict["masks"]
                            choice_score = choice_dict["score"]

                            turn_units.append(
                                torch.tensor(
                                    choice_units,
                                    device=next(self.model_engine.parameters()).device,
                                    dtype=torch.long,
                                )
                            )
                            turn_masks.append(
                                torch.tensor(
                                    choice_masks,
                                    device=next(self.model_engine.parameters()).device,
                                    dtype=torch.bool,
                                )
                            )
                            if choice_score is None:
                                choice_score = int(choice_dict["is_best"]) * 2 - 1
                            turn_rewards.append(choice_score)

                            if n == len(turn) - 1:
                                update_units_list.append(choice_units)

                        bsz = len(turn_units)

                        if bsz < 2:
                            print(f"第{t}轮replay的对比回复个数小于2，跳过采集。")
                            continue

                        turn_units = zero_pad_sequences(turn_units)
                        turn_masks = zero_pad_sequences(turn_masks)
                        turn_rewards = torch.tensor(
                            turn_rewards, device="cpu", dtype=torch.float
                        ).unsqueeze(1)
                        turn_masks = turn_masks[:, 1:]
                        reward_list.append(turn_rewards)
                        turn_advantages = group_advantages(turn_rewards)
                        begin_with_states = (
                            now_state.duplicate(turn_units.shape[0])
                            if now_state is not None
                            else None
                        )
                        log_probs = get_batch_log_probs(
                            rwkv=self.model_engine,
                            t_batch_tokens=turn_units,
                            begin_with_states=begin_with_states,
                        )
                        log_probs_ref = grpo_trainer.get_log_probs_ref(
                            turn_units,
                            state_idx_list=[grpo_trainer.state_idx]
                            * turn_units.shape[0],
                        )

                        kl = kl_div(log_probs, log_probs_ref, turn_masks)

                        exp = ExperienceHist(
                            history_tokens=turn_units.cpu(),
                            action_log_probs=log_probs.cpu(),
                            log_probs_ref=log_probs_ref.cpu(),
                            rewards=turn_rewards.cpu(),
                            advantages=turn_advantages.cpu(),
                            action_mask=turn_masks.cpu(),
                            kl=kl if kl is None else kl.cpu(),
                            begin_with_states=(
                                begin_with_states.cpu()
                                if begin_with_states is not None
                                else None
                            ),
                        )
                        replay_buffer.add(exp)

                        clear_gpu_memory(force=True)

                # 训练
                del now_state
                clear_gpu_memory(force=True)
                episode_reward_sum = (
                    torch.sum(torch.cat(reward_list, dim=0)) if reward_list else 0
                )
                experience_sampler = DataLoader(
                    replay_buffer,
                    batch_size=train_batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=ExperienceHist.gather,
                )

                for step in range(n_train_each_episode):
                    for tt, exp in enumerate(experience_sampler):
                        exp: ExperienceHist

                        if accumulate_grad:
                            is_fault = False
                            loss = 0
                            kl = 0
                            self.model_engine.zero_grad()
                            bsz = exp.history_tokens.size(0)
                            for rb in range(bsz):
                                (
                                    history_tokens,
                                    action_log_probs,
                                    log_probs_ref,
                                    rewards,
                                    advantages,
                                    action_mask,
                                    begin_with_states,
                                ) = (
                                    exp.history_tokens[rb : rb + 1],
                                    exp.action_log_probs[rb : rb + 1],
                                    exp.log_probs_ref[rb : rb + 1],
                                    exp.rewards[rb : rb + 1],
                                    exp.advantages[rb : rb + 1],
                                    exp.action_mask[rb : rb + 1],
                                    (
                                        exp.begin_with_states.batchof(rb)
                                        if exp.begin_with_states is not None
                                        else None
                                    ),
                                )
                                _kl = exp.kl if exp.kl is not None else None
                                train_hist = ExperienceHist(
                                    history_tokens=history_tokens,
                                    action_log_probs=action_log_probs,
                                    log_probs_ref=log_probs_ref,
                                    rewards=rewards,
                                    advantages=advantages,
                                    action_mask=action_mask,
                                    kl=_kl,
                                    begin_with_states=begin_with_states,
                                )
                                train_hist = train_hist.to(
                                    device=next(self.model_engine.parameters()).device
                                )
                                step_loss, step_kl = grpo_loss(
                                    rwkv=self.model_engine,
                                    experience_hist=train_hist,
                                )
                                loss += step_loss
                                kl += step_kl
                                if not step_loss.isfinite():
                                    print(
                                        f"WARNING: loss is infinite, skip this batch, loss={loss}"
                                    )
                                    is_fault = True
                                    break
                                self.model_engine.backward(
                                    step_loss / bsz,
                                    retain_graph=(True if rb < bsz - 1 else False),
                                )
                                print(f"batch: {rb} loss: {step_loss}")
                                train_hist = train_hist.to("cpu")
                                del train_hist
                                clear_gpu_memory(force=True)

                            if is_fault:
                                del train_hist
                                self.model_engine.zero_grad()
                                clear_gpu_memory(force=True)
                                continue
                            loss = loss / bsz
                            kl = kl / bsz
                            grad_norm = clip_grad_norm_(
                                self.model_engine.parameters(),
                                max_norm=grad_cp_max_norm,
                            )
                            self.model_engine.step()
                        else:
                            exp = exp.to(next(self.model_engine.parameters()).device)

                            loss, kl = grpo_loss(
                                self.model_engine,
                                exp,
                            )
                            if not loss.isfinite():
                                print(
                                    f"Warning: Loss not finite, skipping backward, loss={loss}"
                                )
                                continue

                            self.model_engine.zero_grad()
                            self.model_engine.backward(loss)
                            grad_norm = clip_grad_norm_(
                                self.model_engine.parameters(),
                                max_norm=grad_cp_max_norm,
                            )

                            print(
                                f"rl step={step}: loss={loss}, advantage={exp.advantages}"
                            )
                            print(
                                f"rl step={step}: kl={kl: .4f}, grad_norm={grad_norm.item()}"
                            )
                            self.model_engine.step()
                            exp = exp.to("cpu")
                            del exp
                        clear_gpu_memory(force=True)
                        yield (
                            json.dumps(
                                {
                                    "i_file": i_file,
                                    "step": step,
                                    "loss": loss.item(),
                                    "kl": kl.item(),
                                    "sum_rewards": episode_reward_sum.item(),
                                    "grad_norm": grad_norm.item(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            if episode % n_save_episode_ckpt == 0 and self.rank == 0:
                svpath = self.save_weight(
                    f"train_grpo_episode={episode}", save_train_state=True
                )
                yield (
                    json.dumps(
                        {
                            "over": False,
                            "to_dir": svpath,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def train_grpo_online(
        self,
        history,
        ref_model_server: str,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
        train_batch_size: int = 1,
        clip_eps: float = 0.2,
        kl_weight: float = 0.01,
        grad_cp_max_norm: float = 1.0,
        accumulate_grad: bool = True,
        n_train_each_episode: int = 1,
        save_weight_folder=None,
        save_weight_name="train_grpo_online",
        begin_with_state_dir=None,
    ):
        os.makedirs(save_weight_folder, exist_ok=True)
        self.model_engine.train()
        grpo_trainer = GRPOTrainer(
            self.model_engine, ref_model_server, tokenizer=self.train_tokenizer
        )
        grpo_loss = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)
        if lr_init:
            self.build_engine(lr_init, lr_final, warmup_steps)

        dataset = RLGroupDatasetOnline(history, self.train_tokenizer)

        replay_buffer = ReplaySlidingWindow()
        reward_list = []
        turns = dataset.get_episode_data()
        update_units_list = []
        # begin_with_states = (
        #     torch.load(begin_with_state_dir) if begin_with_state_dir else None
        # )
        begin_with_states = (
            torch.load(
                begin_with_state_dir,
                map_location=next(self.model_engine.parameters()).device,
            )
            if begin_with_state_dir
            else None
        )

        now_state = (
            begin_with_states.to(device=next(self.model_engine.parameters()).device)
            if begin_with_states
            else None
        )
        grpo_trainer.request_state_idx(begin_with_state_dir)

        with torch.no_grad():
            for t, turn in enumerate(turns):
                if t > 0:
                    _, now_state = self.model_engine(
                        update_units_list[t - 1 : t], now_state
                    )
                    grpo_trainer.update_ref_state(
                        update_units_list[t - 1 : t],
                        [grpo_trainer.state_idx],
                        grpo_trainer.state_idx,
                    )
                turn_units = []
                turn_masks = []
                turn_rewards = []
                for n, choice_dict in enumerate(turn):
                    choice_units = choice_dict["units"]
                    choice_masks = choice_dict["masks"]
                    choice_score = (
                        choice_dict["score"] if choice_dict["score"] is not None else 0
                    )

                    if choice_score is not None:
                        turn_units.append(
                            torch.tensor(
                                choice_units,
                                device=next(self.model_engine.parameters()).device,
                                dtype=torch.long,
                            )
                        )
                        turn_masks.append(
                            torch.tensor(
                                choice_masks,
                                device=next(self.model_engine.parameters()).device,
                                dtype=torch.bool,
                            )
                        )
                        turn_rewards.append(choice_score)

                    if n == len(turn) - 1:
                        update_units_list.append(choice_units)

                bsz = len(turn_units)
                if bsz < 2:
                    print(f"第{t}轮replay的对比回复个数小于2，跳过采集。")
                    continue
                elif all(score == turn_rewards[0] for score in turn_rewards):
                    print(f"第{t}轮replay的所有score都相等，跳过采集。")
                    continue

                turn_units = zero_pad_sequences(turn_units)
                turn_masks = zero_pad_sequences(turn_masks)
                turn_rewards = torch.tensor(
                    turn_rewards, device="cpu", dtype=torch.float
                ).unsqueeze(1)
                turn_masks = turn_masks[:, 1:]
                reward_list.append(turn_rewards)
                turn_advantages = group_advantages(turn_rewards)
                begin_with_states = (
                    now_state.duplicate(turn_units.shape[0])
                    if now_state is not None
                    else None
                )
                log_probs = get_batch_log_probs(
                    rwkv=self.model_engine,
                    t_batch_tokens=turn_units,
                    begin_with_states=begin_with_states,
                )
                log_probs_ref = grpo_trainer.get_log_probs_ref(
                    turn_units,
                    state_idx_list=[grpo_trainer.state_idx] * turn_units.shape[0],
                )
                kl = kl_div(log_probs, log_probs_ref, turn_masks)
                exp = ExperienceHist(
                    history_tokens=turn_units.cpu(),
                    action_log_probs=log_probs.cpu(),
                    log_probs_ref=log_probs_ref.cpu(),
                    rewards=turn_rewards.cpu(),
                    advantages=turn_advantages.cpu(),
                    action_mask=turn_masks.cpu(),
                    kl=kl if kl is None else kl.cpu(),
                    begin_with_states=(
                        begin_with_states.cpu()
                        if begin_with_states is not None
                        else None
                    ),
                )
                replay_buffer.add(exp)

                clear_gpu_memory(force=True)

        # 训练
        _, now_state = self.model_engine(update_units_list[t - 1 : t], now_state)
        self.train_state = now_state
        del now_state
        clear_gpu_memory(force=True)
        episode_reward_sum = (
            torch.sum(torch.cat(reward_list, dim=0)) if reward_list else 0
        )
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=ExperienceHist.gather,
        )
        for step in range(n_train_each_episode):
            for tt, exp in enumerate(experience_sampler):
                exp: ExperienceHist

                if accumulate_grad:
                    is_fault = False
                    loss = 0
                    kl = 0
                    self.model_engine.zero_grad()
                    bsz = exp.history_tokens.size(0)
                    for rb in range(bsz):
                        (
                            history_tokens,
                            action_log_probs,
                            log_probs_ref,
                            rewards,
                            advantages,
                            action_mask,
                            begin_with_states,
                        ) = (
                            exp.history_tokens[rb : rb + 1],
                            exp.action_log_probs[rb : rb + 1],
                            exp.log_probs_ref[rb : rb + 1],
                            exp.rewards[rb : rb + 1],
                            exp.advantages[rb : rb + 1],
                            exp.action_mask[rb : rb + 1],
                            (
                                exp.begin_with_states.batchof(rb)
                                if exp.begin_with_states is not None
                                else None
                            ),
                        )
                        _kl = exp.kl if exp.kl is not None else None
                        train_hist = ExperienceHist(
                            history_tokens=history_tokens,
                            action_log_probs=action_log_probs,
                            log_probs_ref=log_probs_ref,
                            rewards=rewards,
                            advantages=advantages,
                            action_mask=action_mask,
                            kl=_kl,
                            begin_with_states=begin_with_states,
                        )
                        train_hist = train_hist.to(
                            device=next(self.model_engine.parameters()).device
                        )
                        step_loss, step_kl = grpo_loss(
                            rwkv=self.model_engine,
                            experience_hist=train_hist,
                        )
                        loss += step_loss
                        kl += step_kl
                        if not step_loss.isfinite():
                            print(
                                f"WARNING: loss is infinite, skip this batch, loss={loss}"
                            )
                            is_fault = True
                            break
                        self.model_engine.backward(
                            step_loss / bsz,
                            retain_graph=(True if rb < bsz - 1 else False),
                        )
                        print(f"batch: {rb} loss: {step_loss}")
                        train_hist = train_hist.to("cpu")
                        del train_hist
                        clear_gpu_memory(force=True)

                    if is_fault:
                        del train_hist
                        self.model_engine.zero_grad()
                        clear_gpu_memory(force=True)
                        continue
                    loss = loss / bsz
                    kl = kl / bsz
                    grad_norm = clip_grad_norm_(
                        self.model_engine.parameters(),
                        max_norm=grad_cp_max_norm,
                    )
                    self.model_engine.step()
                else:
                    exp = exp.to(next(self.model_engine.parameters()).device)

                    loss, kl = grpo_loss(
                        self.model_engine,
                        exp,
                    )
                    if not loss.isfinite():
                        print(
                            f"Warning: Loss not finite, skipping backward, loss={loss}"
                        )
                        continue

                    self.model_engine.zero_grad()
                    self.model_engine.backward(loss)
                    grad_norm = clip_grad_norm_(
                        self.model_engine.parameters(),
                        max_norm=grad_cp_max_norm,
                    )

                    print(
                        f"online rl step={step}: loss={loss}, advantage={exp.advantages}"
                    )
                    print(
                        f"online rl step={step}: kl={kl: .4f}, grad_norm={grad_norm.item()}"
                    )
                    self.model_engine.step()
                    exp = exp.to("cpu")
                    del exp
                clear_gpu_memory(force=True)
                yield (
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss.item(),
                            "kl": kl.item(),
                            "sum_rewards": episode_reward_sum.item(),
                            "grad_norm": grad_norm.item(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        svpath = self.save_weight(
            save_weight_name, save_train_state=True, folder=save_weight_folder
        )
        self.train_state = None
        yield (
            json.dumps(
                {
                    "over": True,
                    "to_dir": svpath,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def train_sft_online(
        self,
        history,
        lr_init: float = None,
        lr_final: float = None,
        warmup_steps: int = None,
        begin_with_state_dir: str = None,
        save_weight_folder=None,
        save_weight_name="train_sft_online",
        ctx_len=3072,
    ):
        os.makedirs(save_weight_folder, exist_ok=True)
        self.model_engine.train()
        if lr_init:
            self.build_engine(lr_init, lr_final, warmup_steps)
        dataset = SFTDatasetOnline(history, self.train_tokenizer)

        turns = dataset.get_epoch_data()

        chosen_units = [turn["units"] for turn in turns]
        chosen_masks = [turn["masks"] for turn in turns]

        train_units = sum(chosen_units[1:], chosen_units[0])
        train_masks = sum(chosen_masks[1:], chosen_masks[0])

        assert len(train_units) == len(train_masks)

        content_ctx = len(train_units)

        n_ctx = (
            content_ctx // ctx_len
            if content_ctx % ctx_len == 0
            else content_ctx // ctx_len + 1
        )
        split_ctx_len = content_ctx // n_ctx + 1

        train_units = [
            train_units[i : i + split_ctx_len]
            for i in range(0, content_ctx, split_ctx_len)
        ]
        train_masks = [
            train_masks[i : i + split_ctx_len]
            for i in range(0, content_ctx, split_ctx_len)
        ]
        states = (
            torch.load(
                begin_with_state_dir,
                map_location=next(self.model_engine.parameters()).device,
            )
            if begin_with_state_dir
            else None
        )

        for i, (units, masks) in enumerate(zip(train_units, train_masks)):
            step_units = torch.tensor(
                [units],
                device=next(self.model_engine.parameters()).device,
                dtype=torch.long,
            )
            step_masks = torch.tensor(
                [masks],
                device=next(self.model_engine.parameters()).device,
                dtype=torch.float32,
            )
            # 训练
            out_logits, states = self.model_engine(step_units[:, :-1], states)
            m = calc_cross_entropy_loss(
                out_logits, step_units[:, 1:], step_masks[:, 1:]
            )
            loss = m.item()
            print(
                f"online sft: step={i}: loss={loss}, grad_norm={clip_grad_norm_(self.model_engine.parameters(), max_norm=1.0).item()}"
            )
            self.model_engine.backward(m, retain_graph=i < len(train_units) - 1)

            del step_units, step_masks
            clear_gpu_memory(force=True)
        self.model_engine.step()
        self.train_state = states
        svpath = self.save_weight(
            save_weight_name, save_train_state=True, folder=save_weight_folder
        )
        self.train_state = None
        yield (
            json.dumps(
                {
                    "over": True,
                    "to_dir": svpath,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
