import os
os.environ["WORKING_MODE"] = "pretrain"
from config import global_config
pretrain_config = global_config.pretrain_script_config

import os, sys
import gc
import math
import copy
import torch
import deepspeed
import json
from config import RWKV
from config import RWKVStates
import torch.nn.functional as F
from RWKV.functions import (
    calc_cross_entropy_loss,
    # calc_voice_loss,
    clear_gpu_memory
)
# from RWKV.multimodal_functions import (
#     voice_encode_and_adapt,
# )
from utils.dataset.dataset import MultimodalDataset
from utils.dataset.dataset_functions import UnitStreamProcessor, MyDataloader
from torch.utils.data import DataLoader
from typing import List
import wandb

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import comm as dist

# from config import vocoder

deepspeed.init_distributed()
import torchaudio
import tqdm
import time


class PretrainingAPP:
    def __init__(self):
        self.args = pretrain_config
        self.model = RWKV(self.args, global_config.voice_on)
        self.model_engine = self.build_engine()
        self.train_state = None  # 训练时的滚动state
        self.rank = dist.get_rank()
        self.total_gpus = dist.get_world_size()
        self.pretrain_dataset = MultimodalDataset(
            pretrain_config.dataset_folder,
            global_config.tokenizer_train,
            # self.read_bin_wav,
            None,
            None,
            ctx_len=pretrain_config.model.ctx_len,
        )
        self.data_loader = MyDataloader(
            self.pretrain_dataset,
            pretrain_config.batch_size,
            num_workers=pretrain_config.num_workers_per_gpu,
        )
        self.unit_stream_pressor = UnitStreamProcessor(pretrain_config)
        self.tokenizer = global_config.tokenizer_train

        if global_config.wandb_proj and self.rank == 0:
            wandb.init(project=global_config.wandb_proj)

        # if global_config.voice_on:
        #     self.feat_extractor = torch.load(
        #         pretrain_config.vocoder.load_feature_extractor_dir
        #     )
        #     self.feat_extractor.to(f"cuda:{self.rank}")
        #     self.discriminator = vocoder.VocoderDiscriminator(pretrain_config.vocoder)
        #     self.discriminator_engine, _ = self.discriminator.build_engine()
        #     self.voice_losses = vocoder.Losses(pretrain_config).to(
        #         self.discriminator_engine.device
        #     )
        #     self.voice_unit_len = (
        #         self.args.vocoder.head.hop_length * self.args.vocoder.adapter.chunk_len
        #     )

    def train(
        self,
    ):

        for e in range(pretrain_config.epoches):
            loop = tqdm.tqdm(enumerate(self.data_loader, 0))
            for step, (batch_units, batch_masks) in loop:
                # print(step, len(batch_units), len(batch_units[0]))
                # print(
                # f"===============step:{step},rank: {self.rank}:{time.time()}==============="
                # )
                unit_dicts = self.unit_stream_pressor.encode(
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
                # print(f"===============rank: {self.rank}:{time.time()}===============")

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
                out_latent, out_logits, new_states = (
                    self.model_engine.forward_from_embeddings(embds[:, :-1, :], None)
                )
                text_loss = calc_cross_entropy_loss(
                    out_logits, token_targets, mask_targets
                )
                voice_loss_gen = 0
                voice_loss_disc = 0
                # batch_voice_target_groups = unit_dicts["voice_groups_target"]
                # for batch, voice_target_groups in enumerate(batch_voice_target_groups):
                #     for origin_voice, start_idx, end_idx in voice_target_groups:
                #         voice_latent = out_latent[batch : batch + 1, start_idx:end_idx]
                #         # print(end_idx - start_idx, "---", voice_latent.shape)
                #         out_voice = self.model_engine.vocoder_d(voice_latent)
                #         print(origin_voice.shape, "===", out_voice.shape)
                #         B, ch, N = origin_voice.shape
                #         origin_voice_splited = origin_voice.view(B * ch, N)
                #         out_voice_splited = out_voice.view(B * ch, N)
                #         # 要不要随机切N段，然后做这N段的loss？ --显存3b一次能承受40个units 预计
                #         disc_loss, gen_loss = calc_voice_loss(
                #             self.discriminator,
                #             self.voice_losses,
                #             origin_voice_splited,
                #             out_voice_splited,
                #             pretrain_config,
                #             (
                #                 wandb
                #                 if global_config.wandb_proj and self.rank == 0
                #                 else None
                #             ),
                #         )
                #         voice_loss_disc += disc_loss
                #         voice_loss_gen += gen_loss

                # voice_loss_gen /= len(batch_voice_target_groups)
                # voice_loss_disc /= len(batch_voice_target_groups)
                # if voice_loss_disc:
                #     self.discriminator_engine.zero_grad()
                #     self.discriminator_engine.backward(voice_loss_disc)
                #     self.discriminator_engine.step()

                m = text_loss if voice_loss_gen == 0 else text_loss + voice_loss_gen
                self.model_engine.zero_grad()
                self.model_engine.backward(m)
                self.model_engine.step()

                if global_config.voice_on:
                    loop.set_postfix(
                        text_loss=text_loss.item(), voice_loss_gen=voice_loss_gen
                    )
                else:
                    loop.set_postfix(text_loss=text_loss.item())
                if global_config.wandb_proj and self.rank == 0:
                    wandb.log({"text_loss": text_loss.item(), "total_loss": m.item()})
                dist.barrier()

                if step % pretrain_config.save_weight_steps == 0 and self.rank == 0:
                    save_path = self.save_weight(
                        f"pretrain_step_{step}", save_train_state=False
                    )
                    if global_config.voice_on:
                        save_disc_path = self.save_disc(f"pretrain_step_{step}")
                        print(f"====save disc at step: {step}, to={save_path}====")
                    print(f"====save ckpt at step: {step}, to={save_path}====")

            if e % pretrain_config.save_weight_epochs == 0 and self.rank == 0:
                save_path = self.save_weight(
                    f"pretrain_epoch_{e}", save_train_state=False
                )
                if global_config.voice_on:
                    save_disc_path = self.save_disc(f"pretrain_epoch_{e}")
                    print(f"====save disc at step: {step}, to={save_disc_path}====")
                print(f"====save ckpt at step: {step}, to={save_path}====")

    # def voice_encode_and_adapt(self, x):
    #     l, r = torch.split(x, 1, dim=1)
    #     vs = []
    #     for v in (l, r):

    #         B, ch, N = v.size()

    #         # 假设采样率为24000Hz，计算音频时长
    #         duration_in_seconds = N / 24000.0

    #         # 如果音频时长超过1秒，计算切分段数
    #         # TODO 随机切段不太行，需要按照24000切
    #         if duration_in_seconds > 1:
    #             v_segments = torch.split(v, 24000, dim=2)
    #         else:
    #             v_segments = [v]

    #         processed_segments = []

    #         # 处理每个切分的音频段
    #         for segment in v_segments:
    #             segment = self.feat_extractor(
    #                 segment.to(
    #                     device=next(self.feat_extractor.parameters()).device,
    #                     dtype=next(self.feat_extractor.parameters()).dtype,
    #                 )
    #             ).to(
    #                 device=next(self.model_engine.parameters()).device,
    #                 dtype=next(self.model_engine.parameters()).dtype,
    #             )
    #             processed_segments.append(segment)

    #         # 将处理后的段合并
    #         v = torch.cat(processed_segments, dim=2)
    #         vs.append(v)
    #     x = self.model_engine.track_mixing(*vs)
    #     x = self.model_engine.adapter_e(x)
    #     return x

    def load_model(self, ckpt_dir: str):
        model_weights = torch.load(ckpt_dir, map_location="cpu")
        self.load_state_dict(model_weights, strict=False)
        self.model_engine = self.build_engine()
        clear_gpu_memory(force=True)

        self.rank = dist.get_rank()
        self.total_gpus = dist.get_world_size()

        if global_config.wandb_proj and self.rank == 0:
            wandb.init(project=global_config.wandb_proj)

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

    def save_disc(self, name: str, folder: str = None):
        folder = folder if folder else global_config.ckpt_dir
        fpath = f"{folder}/{name}_disc.pth"
        self.discriminator.load_state_dict(
            self.discriminator_engine.module.state_dict()
        )
        torch.save(self.discriminator.state_dict(), fpath)
        clear_gpu_memory(force=True)
        return fpath

    def build_engine(self):
        ds_config = {
            "bfloat16": {"enabled": True},
            "gradient_accumulation_steps": self.args.deepspeed.gradient_accumulation_steps,
            "gradient_clipping": self.args.train.grad_cp,
            "train_micro_batch_size_per_gpu": 1,
        }
        if pretrain_config.deepspeed.zero:
            ds_config["zero_optimization"] = {
                "stage": pretrain_config.deepspeed.ds_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e6,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e6,
                "contiguous_gradients": True,
            }

            if pretrain_config.deepspeed.offload_optimizer:
                ds_config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
            if (
                pretrain_config.deepspeed.offload_param_stage3
                and pretrain_config.deepspeed.ds_stage == 3
            ):
                ds_config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

        self.optimizer = (
            DeepSpeedCPUAdam(
                self.model.get_optim_groups(),
                lr=self.args.train.lr_init,
                betas=(self.args.train.beta1, self.args.train.beta2),
                eps=self.args.train.adam_eps,
                adamw_mode=self.args.train.adamw_mode,
                weight_decay=self.args.train.weight_decay,
                amsgrad=False,
                bias_correction=True,
            )
            if pretrain_config.deepspeed.zero
            and pretrain_config.deepspeed.offload_optimizer
            else FusedAdam(
                self.model.get_optim_groups(),
                lr=self.args.train.lr_init,
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
            warmup_min_lr=self.args.train.lr_init,
            warmup_max_lr=self.args.train.lr_final,
            warmup_num_steps=self.args.train.warmup_steps,
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

    # @torch.no_grad()
    # def read_wav(self, fp):
    #     y, sr = torchaudio.load(fp)
    #     # if y.size(0) > 1:
    #     #     # mix to mono
    #     #     y = y.mean(dim=0, keepdim=True)
    #     gain = -3
    #     y, _ = torchaudio.sox_effects.apply_effects_tensor(
    #         y, sr, [["norm", f"{gain:.2f}"]]
    #     )
    #     if sr != pretrain_config.vocoder.sample_rate:
    #         y = torchaudio.functional.resample(
    #             y, orig_freq=sr, new_freq=pretrain_config.vocoder.sample_rate
    #         )
    #     last_length = y.size(-1) % (
    #         pretrain_config.vocoder.head.hop_length
    #         * pretrain_config.vocoder.adapter.chunk_len
    #     )
    #     if last_length != 0:
    #         padding_tensor = torch.zeros(
    #             1,
    #             pretrain_config.vocoder.head.hop_length
    #             * pretrain_config.vocoder.adapter.chunk_len
    #             - last_length,
    #         )
    #         y = torch.cat((y, padding_tensor), dim=-1)
    #     return y

    # @torch.no_grad()
    # def read_bin_wav(
    #     self,
    #     fp: str,
    # ):
    #     wav = self.read_wav(fp)
    #     ch, N = wav.size()
    #     if ch == 1:
    #         return torch.cat([wav.clone(), wav.clone()], dim=0)
    #     else:
    #         return wav


if __name__ == "__main__":
    app = PretrainingAPP()
    app.train()
