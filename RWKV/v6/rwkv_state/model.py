##############################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
##############################################
import importlib, gc
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import types
from torch.cuda.amp import autocast

if importlib.util.find_spec("deepspeed"):
    import deepspeed
from .block import Block
from RWKV.v6.state import RWKVStates
from .lora import LoraLinear
from .vocoder import AdapterE, VoiceDecoder, TrackMixing
from functools import partial
from typing import Union, Optional, List


class RWKV(nn.Module):
    def __init__(self, args_in, voice_on=False):
        super().__init__()
        args = types.SimpleNamespace(**vars(args_in.model))
        args.lora = args_in.lora
        args.train = args_in.train
        args.model = args_in.model
        args.dropout = args_in.train.dropout
        args.grad_cp = args_in.train.grad_cp
        args.lora_on = args_in.lora.lora_on
        args.voice_on = voice_on
        if voice_on:
            args.vocoder = args_in.vocoder
        self.args = args
        if self.args.model.dtype == "fp32":
            self.dtype = torch.float
        elif self.args.model.dtype == "fp16":
            self.dtype = torch.half
        elif self.args.model.dtype == "bf16":
            self.dtype = torch.bfloat16
        # load weight
        if args.load_model is not None:
            model_weights = torch.load(args.load_model, map_location="cpu")
        else:
            model_weights = self.state_dict()

        model_keys = list(model_weights.keys())
        # calc init layer
        if args.n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if "blocks." in x:
                    block_id = int(x.split(".")[1])
                    max_block_id = max(max_block_id, block_id)
            args.n_layer = max_block_id + 1

        # calc n_embd
        if args.n_embd < 0:
            args.n_embd = model_weights["head.weight"].shape[1]
        print("embd size:", args.n_embd)
        args.size = "7b" if args.n_embd == 4096 else "3b"

        # clac vocab_size
        if args.vocab_size < 0:
            args.vocab_size = model_weights["head.weight"].shape[0]

        args.dim_att = args.n_embd
        args.n_head = args.dim_att // args.head_size
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # init dropout
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

        if args.lora.lora_on:
            if args.lora.train_state:
                pass
            else:
                model_weights = self.load_lora(model_weights)

        self.load_state_dict(model_weights, strict=False)
        del model_weights

        self.adapter_e = None
        if args.voice_on:
            self.track_mixing = TrackMixing(args.vocoder.vocos_backbone.input_channels)
            self.adapter_e = AdapterE(
                args.vocoder.adapter.chunk_len,
                args.vocoder.vocos_backbone.input_channels,
                n_embd=args.n_embd,
            )
            self.vocoder_d = VoiceDecoder(args_in)
            if args.vocoder.load_disc_model:
                ckpt = torch.load(args.vocoder.load_disc_model)
                self.adapter_e.load_state_dict(ckpt["e"])
                self.vocoder_d.load_state_dict(ckpt["d"])

        for p in self.parameters():
            p.data = p.data.to(dtype=self.dtype)

        gc.collect()
        torch.cuda.empty_cache()
        
    @property
    def device(self):
        return next(self.parameters()).device

    def save_vocoder(self, to_dir):
        d = {"e": self.adapter_e.state_dict(), "d": self.vocoder_d.state_dict()}
        torch.save(d, to_dir)

    def get_optim_groups(self):
        args = self.args
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        lr_10x = set()
        if args.lora.lora_on:
            for n, p in self.named_parameters():
                if args.lora.train_state:
                    if "encode" in n:
                        print("======load state===========")
                        lr_1x.add(n)
                    if "decode" in n:
                        print("======load state===========")
                        lr_1x.add(n)
                    else:
                        pass
                else:
                    print("======load lora===========")
                    if "time_state" in n:
                        lr_10x.add(n)
                    elif "time_decay" in n:
                        lr_2x.add(n)
                    elif "lora_" in n:
                        lr_1x.add(n)
                    elif (".ln" in n) and ("ln" in args.lora.parts):
                        lr_1x.add(n)
                    elif "time" in n:
                        lr_1x.add(n)
                    elif ("emb.weight" == n) and ("emb" in args.lora.parts):
                        lr_1x.add(n)
                    elif ("head.weight" == n) and ("head" in args.lora.parts):
                        lr_1x.add(n)
                    elif ("gate" == n) and ("gate" in args.lora.parts):
                        lr_1x.add(n)
                    else:
                        pass
        else:
            all_params = set()

            for n, p in self.named_parameters():
                if (("_w1" in n) or ("_w2" in n)) and (args.train.layerwise_lr > 0):
                    if n not in all_params:
                        lr_1x.add(n)
                        all_params.add(n)
                elif (("time_mix" in n) or ("time_maa" in n)) and (
                    args.train.layerwise_lr > 0
                ):
                    if args.train.my_pile_stage == 2:
                        if n not in all_params:
                            lr_2x.add(n)
                            all_params.add(n)
                    else:
                        if n not in all_params:
                            lr_1x.add(n)
                            all_params.add(n)
                elif (("time_decay" in n) or ("time_daaaa" in n)) and (
                    args.train.layerwise_lr > 0
                ):
                    if args.train.my_pile_stage == 2:
                        if n not in all_params:
                            lr_3x.add(n)
                            all_params.add(n)
                    else:
                        if n not in all_params:
                            lr_2x.add(n)
                            all_params.add(n)
                elif ("time_faaaa" in n) and (args.train.layerwise_lr > 0):
                    if args.train.my_pile_stage == 2:
                        if n not in all_params:
                            lr_2x.add(n)
                            all_params.add(n)
                    else:
                        if n not in all_params:
                            lr_1x.add(n)
                            all_params.add(n)
                elif ("time_first" in n) and (args.train.layerwise_lr > 0):
                    if n not in all_params:
                        lr_3x.add(n)
                        all_params.add(n)
                elif "track_mixing" in n:
                    lr_1x.add(n)
                elif "adapter_e" in n or "vocoder_d" in n:
                    if ("vocos_backbone" in n) or ("head" in n):
                        lr_10x.add(n)
                    else:
                        lr_1x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (args.train.weight_decay > 0):
                    if n not in all_params:
                        lr_decay.add(n)
                        all_params.add(n)
                else:
                    if n not in all_params:
                        lr_1x.add(n)
                        all_params.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        lr_10x = sorted(list(lr_10x))
        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[n] for n in lr_1x],
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,
            },
            {
                "params": [param_dict[n] for n in lr_2x],
                "weight_decay": 0.0,
                "my_lr_scale": 2.0,
            },
            {
                "params": [param_dict[n] for n in lr_3x],
                "weight_decay": 0.0,
                "my_lr_scale": 3.0,
            },
            {
                "params": [param_dict[n] for n in lr_10x],
                "weight_decay": 0.0,
                "my_lr_scale": 10.0,
            },
        ]

        if args.train.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": args.train.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
        return optim_groups

    def forward_from_embeddings(self, embeddings, states, allow_torch_checkpoint=True):
        """
        embeddings : (b, N, n_embd)
        output :  (b, N, n_embd)
        """
        args = self.args
        B, T, n_embd = embeddings.size()
        C = args.n_embd
        H = args.dim_att // args.head_size
        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert n_embd == C

        new_states = RWKVStates.create(
            args.n_layer,
            B,
            C,
            args.n_head,
            args.head_size,
            embeddings.device,
            self.emb.weight.dtype,
        )
        if states is None:
            states = RWKVStates.create(
                args.n_layer,
                B,
                C,
                args.n_head,
                args.head_size,
                embeddings.device,
                self.emb.weight.dtype,
            )

        x = embeddings
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            state = states[i]

            if int(args.grad_cp) == 1:
                if args.lora.r > 0 and allow_torch_checkpoint:
                    x, state = torch_checkpoint(block, x, state, use_reentrant=False)
                elif allow_torch_checkpoint:
                    x, state = deepspeed.checkpointing.checkpoint(block, x, state)
                else:
                    x, state = block(x, state)
            else:
                x, state = block(x, state)
            new_states[i] = state

        out_latent = x
        x = self.ln_out(x)
        logits = self.head(x)

        return out_latent, logits, new_states

    def forward(
        self,
        idx: Union[torch.Tensor, list],
        states: RWKVStates = None,
        overwrite_states: bool = False,
        latent_output=False,
    ):
        args = self.args
        # idx [B,N]
        idx = torch.tensor(idx, device=self.device, dtype=torch.long)

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size

        assert C == H * args.head_size

        if states is None:
            states = RWKVStates.create(
                args.n_layer,
                B,
                C,
                args.n_head,
                args.head_size,
                idx.device,
                self.emb.weight.dtype,
            )
        states = states.to(self.device)
        new_states = (
            RWKVStates.create(
                args.n_layer,
                B,
                C,
                args.n_head,
                args.head_size,
                idx.device,
                self.emb.weight.dtype,
            )
            if not overwrite_states
            else states
        )

        x = self.emb(idx)

        if args.dropout > 0:
            x = self.drop0(x)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            state = states[i]

            if int(args.grad_cp) == 1 and self.training:
                if args.lora.r > 0:
                    x, state = torch_checkpoint(block, x, state, use_reentrant=False)
                else:
                    x, state = deepspeed.checkpointing.checkpoint(block, x, state)
            else:
                x, state = block(x, state)
            new_states[i] = state
            
        if latent_output:
            latent = x
        x = self.ln_out(x)
        logits = self.head(x)
        
        if latent_output:
            return logits, new_states, latent
        return logits, new_states
    
    def load_lora(self, weight, debug=True):
        if len(self.args.lora.path) != 0:
            try:
                print("===========load from path======================")
                lora_weight = torch.load(self.args.lora.path, map_location="cpu")
                if debug:
                    for k, v in lora_weight.items():
                        print(k)
                weight.update(lora_weight)
            except:
                print("======load error====")
        return weight

    def embedding(self, idx):
        args = self.args
        idx = idx.to(self.device, dtype=torch.long)

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size
        assert C == H * args.head_size
        x = self.emb(idx)
        return x

    def to_logits(self, x):
        x = x.to(self.device, dtype=self.dtype)
        x = self.ln_out(x)
        logits = self.head(x)
        return logits
