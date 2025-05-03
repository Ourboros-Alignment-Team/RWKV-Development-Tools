##############################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
##############################################
import importlib, gc
import torch
import torch.nn as nn
import types

from .block import Block
from RWKV.v6.state import RWKVStates
from torch.nn import functional as F
from typing import Union


class RWKV(nn.Module):
    def __init__(self, args_in):
        super().__init__()
        args = types.SimpleNamespace(**vars(args_in.model))
        self.args = args
        if self.args.dtype == "fp32":
            self.dtype = torch.float
        elif self.args.dtype == "fp16":
            self.dtype = torch.half
        elif self.args.dtype == "bf16":
            self.dtype = torch.bfloat16
        self.args.train = types.SimpleNamespace()
        self.args.train.dropout = 0
        args.dropout = 0
        # load weight
        model_weights = torch.load(args.load_model, map_location="cpu")
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

        self.load_state_dict(model_weights, strict=False)
        del model_weights
        with torch.no_grad():
            self.emb.weight=nn.Parameter(F.layer_norm(
                self.emb.weight,
                (args.n_embd,),
                weight=self.blocks[0].ln0.weight,
                bias=self.blocks[0].ln0.bias,
            ))

        self.to(device=self.args.device, dtype=self.dtype)

        gc.collect()
        torch.cuda.empty_cache()

    @property
    def device(self):
        return next(self.parameters()).device
    
    @torch.no_grad()
    def infer(self, idx, states=None, overwrite_states=False):
        args = self.args
        # idx
        idx = torch.tensor([idx], dtype=torch.long).to(next(self.parameters()).device)

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size

        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
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
        states = states.to(next(self.parameters()).device)
    
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
        ).to(next(self.parameters()).device)


        x = self.emb(idx)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            state = states[i]
            x, state = block(x, state)
            new_states[i] = state

        x = self.ln_out(x)
        logits = self.head(x)

        return logits, new_states

    def tokens_list2tensor(self, tokens_batches):
        # 找到最大长度
        max_length = max(len(batch) for batch in tokens_batches)

        # 初始化张量和补零位置索引列表
        tensor_batches = torch.full(
            (len(tokens_batches), max_length), fill_value=0, dtype=torch.long
        )
        indices = []

        # 填充张量并记录补零位置
        for i, batch in enumerate(tokens_batches):
            length = len(batch)
            tensor_batches[i, :length] = torch.tensor(batch, dtype=torch.long)
            indices.append(length)

        return tensor_batches, indices

    @torch.no_grad()
    def forward(
        self,
        tokens_batches: Union[torch.Tensor, list],
        states: RWKVStates = None,
        latent_output: bool = False,
    ):
        args = self.args
        # tensor_batches, _ = self.tokens_list2tensor(tokens_batches)
        idx = torch.tensor(tokens_batches, dtype=torch.long).to(
            next(self.parameters()).device
        )

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size

        assert T <= self.args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        assert C == H * args.head_size

        new_states = RWKVStates.create(
            args.n_layer,
            B,
            C,
            args.n_head,
            args.head_size,
            self.emb.weight.device,
            self.emb.weight.dtype,
        )
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
        states = states.to(next(self.parameters()).device)

        x = self.emb(idx)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            state = states[i]
            x, state = block(x, state)
            new_states[i] = state

        if latent_output:
            latent = x
        x = self.ln_out(x)
        logits = self.head(x)

        if latent_output:
            return logits, new_states, latent
        return logits, new_states

    @torch.no_grad()
    def forward_from_embeddings(self, embeddings, states):
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

            x, state = block(x, state)
            new_states[i] = state

        out_latent = x
        x = self.ln_out(x)
        logits = self.head(x)

        return out_latent, logits, new_states

    def load_weights(self, load_dir):
        model_weights = torch.load(load_dir, map_location="cpu")
        self.load_state_dict(model_weights, strict=False)
        self.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def to_logits(self, x):
        x = x.to(next(self.parameters()).device, dtype=next(self.parameters()).dtype)
        x = self.ln_out(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def embedding(self, idx):
        args = self.args
        idx = idx.to(next(self.parameters()).device, dtype=torch.long)

        B, T = idx.size()
        C = args.n_embd
        H = args.dim_att // args.head_size
        assert C == H * args.head_size
        x = self.emb(idx)
        return x
