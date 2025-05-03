########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import types
import gc
from typing import Union, Optional, List
from RWKV.v7.state import RWKVStates, LayerRWKVStates
import threading


@torch.jit.script
def RWKV_x070_TMix_seq(
    layer_id: int,
    H: int,
    N: int,
    x,
    shift_state,
    v_first,
    wkv_state,
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    w0,
    w1,
    w2,
    a0,
    a1,
    a2,
    v0,
    v1,
    v2,
    g1,
    g2,
    k_k,
    k_a,
    r_k,
    R_,
    K_,
    V_,
    O_,
    ln_w,
    ln_b,
):
    B, T, _ = x.shape
    xx = torch.cat((shift_state.unsqueeze(1), x[:, :-1, :]), dim=1) - x
    xr, xw, xk, xv, xa, xg = (
        x + xx * x_r,
        x + xx * x_w,
        x + xx * x_k,
        x + xx * x_v,
        x + xx * x_a,
        x + xx * x_g,
    )

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(B, T, H, N), dim=-1, p=2.0).view(
        B, T, H * N
    )
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    #
    # w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
    # x, wkv_state = RUN_CUDA_RWKV7s(r, w, k, v, -kk, kk * a, wkv_state)
    #
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float()))  # 0.606531 = exp(-0.5)
    for t in range(T):
        r_, w_, k_, v_, kk_, a_ = r[:, t], w[:, t], k[:, t], v[:, t], kk[:, t], a[:, t]
        vk = v_.view(B, H, N, 1) @ k_.view(B, H, 1, N)
        ab = (-kk_).view(B, H, N, 1) @ (kk_ * a_).view(B, H, 1, N)
        wkv_state = (
            wkv_state * w_.view(B, H, 1, N) + wkv_state @ ab.float() + vk.float()
        )
        xx[:, t] = (wkv_state.to(dtype=x.dtype) @ r_.view(B, H, N, 1)).view(B, H * N)

    xx = torch.nn.functional.group_norm(
        xx.view(B * T, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5
    ).view(B, T, H * N)
    xx = xx + (
        (r * k * r_k).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)
    ).view(B, T, H * N)
    return (xx * g) @ O_, x[:, -1, :], wkv_state, v_first


@torch.jit.script
def RWKV_x070_CMix_seq(x, shift_state, x_k, K_, V_):
    xx = torch.cat((shift_state.unsqueeze(1), x[:, :-1, :]), dim=1) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[:, -1, :]


class RWKV(torch.jit.ScriptModule):
    def __init__(self, args_in):
        super().__init__()
        self.configs = types.SimpleNamespace(**vars(args_in))
        args = self.configs.model
        self.args = args
        model_weights = torch.load(args.load_model, map_location=args.device)
        self.device = torch.device(args.device)
        model_keys = list(model_weights.keys())

        if self.args.dtype == "fp32":
            self.dtype = torch.float
        elif self.args.dtype == "fp16":
            self.dtype = torch.half
        elif self.args.dtype == "bf16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError("dtype must be fp32, fp16 or bf16")
        args.vocab_size, args.n_embd = model_weights["emb.weight"].shape
        # if args.n_embd < 0:
        #     args.n_embd = model_weights["head.weight"].shape[1]
        print("embd size:", args.n_embd)

        # if args.vocab_size < 0:
        #     args.vocab_size = model_weights["head.weight"].shape[0]
        print("vocab_size:", args.vocab_size)

        args.n_head, args.head_size = model_weights["blocks.0.att.r_k"].shape
        self.n_head = args.n_head
        self.head_size = args.head_size

        args.n_layer = 0
        for k in model_keys:
            layer_id = int(k.split(".")[1]) if ("blocks." in k) else 0
            args.n_layer = max(args.n_layer, layer_id + 1)
            if (
                "key.weight" in k
                or "value.weight" in k
                or "receptance.weight" in k
                or "output.weight" in k
                or "head.weight" in k
            ):
                model_weights[k] = model_weights[k].t()
            model_weights[k] = model_weights[k].squeeze().to(dtype=self.dtype)
            if k.endswith("att.r_k"):
                model_weights[k] = model_weights[k].flatten()
        print("n_layer:", args.n_layer)

        model_weights["emb.weight"] = F.layer_norm(
            model_weights["emb.weight"],
            (args.n_embd,),
            weight=model_weights["blocks.0.ln0.weight"],
            bias=model_weights["blocks.0.ln0.bias"],
        )
        model_weights["blocks.0.att.v0"] = model_weights[
            "blocks.0.att.a0"
        ]  # actually ignored
        model_weights["blocks.0.att.v1"] = model_weights[
            "blocks.0.att.a1"
        ]  # actually ignored
        model_weights["blocks.0.att.v2"] = model_weights[
            "blocks.0.att.a2"
        ]  # actually ignored

        self.model_weights = model_weights
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer

        self.empty_state_queue = RWKVStates.create(
            args.n_layer,
            self.configs.batching_batch_size + 1,
            args.n_embd,
            args.n_head,
            args.head_size,
            self.device,
            self.dtype,
        )
        self.state_lock = threading.Lock()

        gc.collect()
        torch.cuda.empty_cache()

    def forward(
        self,
        tokens_batches: list,
        states: RWKVStates = None,
        latent_output: bool = False,
    ):
        args = self.args
        idx = torch.tensor(tokens_batches, dtype=torch.long).to(self.device)

        B, _ = idx.size()
        C = args.n_embd
        if states is None:
            with self.state_lock:
                self.prepare_state()
                states = self.empty_state_queue.batchof(slice(0, B))
                self.empty_state_queue.pop(slice(0, B))

        thread = threading.Thread(target=self.prepare_state, daemon=True)
        thread.start()

        # if idx.shape[1] > 1:
        idx = self.embedding(idx)
        res = self.forward_seq_from_embeddings(
            idx,
            states.to_rwkv_list_states(),
        )
        x, new_states, out_latent = res
        new_states = RWKVStates.from_rwkv_list_states(new_states)
        if latent_output:
            return x, new_states, out_latent
        return x, new_states

    @torch.jit.script_method
    @torch.no_grad()
    def forward_seq_from_embeddings(
        self,
        x: torch.Tensor,
        state: List[torch.Tensor],
    ):

        z = self.model_weights

        v_first = torch.empty_like(x)
        for i in range(self.n_layer):
            bbb = f"blocks.{i}."
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."

            xx = F.layer_norm(
                x,
                (self.n_embd,),
                weight=z[bbb + "ln1.weight"],
                bias=z[bbb + "ln1.bias"],
            )

            xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_seq(
                i,
                self.n_head,
                self.head_size,
                xx,
                state[i * 3 + 0],
                v_first,
                state[i * 3 + 1],
                z[att + "x_r"],
                z[att + "x_w"],
                z[att + "x_k"],
                z[att + "x_v"],
                z[att + "x_a"],
                z[att + "x_g"],
                z[att + "w0"],
                z[att + "w1"],
                z[att + "w2"],
                z[att + "a0"],
                z[att + "a1"],
                z[att + "a2"],
                z[att + "v0"],
                z[att + "v1"],
                z[att + "v2"],
                z[att + "g1"],
                z[att + "g2"],
                z[att + "k_k"],
                z[att + "k_a"],
                z[att + "r_k"],
                z[att + "receptance.weight"],
                z[att + "key.weight"],
                z[att + "value.weight"],
                z[att + "output.weight"],
                z[att + "ln_x.weight"],
                z[att + "ln_x.bias"],
            )
            x = x + xx
            xx = F.layer_norm(
                x,
                (self.n_embd,),
                weight=z[bbb + "ln2.weight"],
                bias=z[bbb + "ln2.bias"],
            )
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(
                xx,
                state[i * 3 + 2],
                z[ffn + "x_k"],
                z[ffn + "key.weight"],
                z[ffn + "value.weight"],
            )

            x = x + xx

        latent = x
        x = F.layer_norm(
            x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]
        )
        x = x @ z["head.weight"]

        return x, state, latent

    @torch.no_grad()
    def load_weights(self, load_dir):
        print(f"load weights from {load_dir}...")
        model_weights = torch.load(load_dir, map_location=self.device)
        model_keys = list(model_weights.keys())
        args = self.args

        args.vocab_size, args.n_embd = model_weights["emb.weight"].shape
        print("embd size:", args.n_embd)
        print("vocab_size:", args.vocab_size)

        args.n_head, args.head_size = model_weights["blocks.0.att.r_k"].shape
        self.n_head = args.n_head
        self.head_size = args.head_size
        args.n_layer = 0
        for k in model_keys:
            layer_id = int(k.split(".")[1]) if ("blocks." in k) else 0
            args.n_layer = max(args.n_layer, layer_id + 1)
            if (
                "key.weight" in k
                or "value.weight" in k
                or "receptance.weight" in k
                or "output.weight" in k
                or "head.weight" in k
            ):
                model_weights[k] = model_weights[k].t()
            model_weights[k] = model_weights[k].squeeze().to(dtype=self.dtype)
            if k.endswith("att.r_k"):
                model_weights[k] = model_weights[k].flatten()
        print("n_layer:", args.n_layer)

        model_weights["emb.weight"] = F.layer_norm(
            model_weights["emb.weight"],
            (args.n_embd,),
            weight=model_weights["blocks.0.ln0.weight"],
            bias=model_weights["blocks.0.ln0.bias"],
        )
        model_weights["blocks.0.att.v0"] = model_weights[
            "blocks.0.att.a0"
        ]  # actually ignored
        model_weights["blocks.0.att.v1"] = model_weights[
            "blocks.0.att.a1"
        ]  # actually ignored
        model_weights["blocks.0.att.v2"] = model_weights[
            "blocks.0.att.a2"
        ]  # actually ignored

        self.model_weights = model_weights
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def infer(self, idx, states=None, overwrite_states=False):
        logits, new_states = self.forward([idx], states)
        return logits, new_states

    @torch.no_grad()
    def embedding(self, x: torch.Tensor):
        z = self.model_weights
        x = z["emb.weight"][x]
        return x

    @torch.no_grad()
    @torch.jit.script_method
    def forward_from_embeddings(self, embeddings, states):
        args = self.args
        embeddings = embeddings.to(self.device, self.dtype)
        B, _, _ = embeddings.size()
        C = args.n_embd
        new_states = RWKVStates.create(
            args.n_layer,
            B,
            C,
            args.n_head,
            args.head_size,
            self.device,
            self.dtype,
        )
        if states is None:
            states = RWKVStates.create(
                args.n_layer,
                B,
                C,
                args.n_head,
                args.head_size,
                self.device,
                self.dtype,
            )
        res = self.forward_seq_from_embeddings(
            embeddings,
            states.to_rwkv_list_states(),
        )
        x, new_states, out_latent = res
        new_states = RWKVStates.from_rwkv_list_states(new_states)
        return out_latent, x, new_states

    @torch.no_grad()
    def to_logits(self, x):
        z = self.model_weights
        x = x.to(self.device, dtype=self.dtype)
        x = x @ z["head.weight"]

        return x

    def prepare_state(self):
        last_n_batch = self.empty_state_queue.get_batch_size()
        if last_n_batch <= self.configs.batching_batch_size:
            supplement_n_batch = self.configs.batching_batch_size + 1 - last_n_batch
            supp_states = RWKVStates.create(
                self.args.n_layer,
                supplement_n_batch,
                self.args.n_embd,
                self.args.n_head,
                self.args.head_size,
                self.device,
                self.dtype,
            )
            a = self.empty_state_queue + supp_states
            self.empty_state_queue = self.empty_state_queue + supp_states
        last_n_batch = self.empty_state_queue.get_batch_size()
