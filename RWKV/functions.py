import torch
from torch.nn import functional as F
import numpy as np
import copy
import sys
from config import RWKVStates
import gc
from typing import List

torch.autograd.set_detect_anomaly(True)


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


def ppl(tokens, logits):
    with torch.no_grad():
        cel = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            torch.tensor(tokens, device=logits.device),
            reduction="none",
        )
        ppl = [x.item() for x in torch.exp(cel).unbind(0)]
        return ppl


def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        # 'privateuseone' is the type of custom devices like `torch_directml.device()`
        if probs.device.type in ["cpu", "privateuseone"]:
            probs = probs.cpu().numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)


def sample_logits_batch(logits, temperature=1.0, top_p=0.85, top_k=0):
    logits = logits[:, -1, :]
    # logits shape (B, 65536)
    with torch.no_grad():
        if temperature == 0:
            # 直接返回最大概率的token
            return torch.argmax(logits, dim=-1)

        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        # 'privateuseone' is the type of custom devices like `torch_directml.device()`
        if probs.device.type in ["cpu", "privateuseone"]:
            probs = probs.cpu().numpy()
            # Sort along last dimension
            sorted_ids = np.argsort(probs, axis=-1)
            sorted_probs = np.take_along_axis(probs, sorted_ids, axis=-1)[..., ::-1]
            cumulative_probs = np.cumsum(sorted_probs, axis=-1)
            # Find cutoff for each sequence
            cutoff_idx = np.argmax(cumulative_probs >= top_p, axis=-1)
            cutoff = sorted_probs[np.arange(len(cutoff_idx)), cutoff_idx]
            cutoff = cutoff[..., np.newaxis]
            probs = np.where(probs < cutoff, 0, probs)
            if top_k < probs.shape[-1] and top_k > 0:
                probs.scatter_(1, sorted_ids[:, :-(top_k)], 0)
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            # Renormalize
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            out = np.array([np.random.choice(probs.shape[-1], p=p) for p in probs])
            return out.astype(np.int64)
        else:
            batch_size = probs.shape[0]
            sorted_ids = torch.argsort(probs, dim=-1)
            sorted_probs = torch.gather(probs, -1, sorted_ids).flip(dims=[-1])
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff_idx = np.argmax(cumulative_probs >= top_p, axis=-1)
            cutoff = sorted_probs[torch.arange(batch_size), cutoff_idx].unsqueeze(-1)
            probs = torch.where(probs < cutoff, 0, probs)
            if top_k < probs.shape[-1] and top_k > 0:
                probs.scatter_(1, sorted_ids[:, :-(top_k)], 0)
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            out = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return out


@torch.no_grad()
def speak_next_token(
    rwkv,
    last_token,
    states,
    occurrence,
    temperature,
    top_p,
    alpha_presence,
    alpha_frequency,
    alpha_decay,
    token_ban=[],
    token_allow=None,
):

    idx = torch.tensor(
        [
            [last_token],
        ],
        dtype=torch.long,
    ).to(rwkv.args.device)

    logits, states = rwkv(
        idx, states, allow_torch_checkpoint=False, overwrite_states=True
    )

    for xxx in occurrence:
        occurrence[xxx] *= alpha_decay
    for n in occurrence:
        logits[0, -1, n] -= alpha_presence + occurrence[n] * alpha_frequency
    for n in token_ban:
        logits[0, -1, n] = -1e38
    if token_allow is not None:
        mask = torch.full_like(logits[0, -1, :], -1e38, device=logits.device)
        mask[token_allow] = logits[0, -1, token_allow]
        logits[0, -1, :] = mask
    next_token = sample_logits(logits[:, -1, :].squeeze(), temperature, top_p)
    return next_token, states

@torch.no_grad()
def speak(
    rwkv,
    start_with_token: int,
    states,
    temperature,
    top_p,
    alpha_presence,
    alpha_frequency,
    alpha_decay,
    tokenizer,
    token_stop=[65535],
    max_tokens=100,
    m_postfix_token=[],
):
    with torch.no_grad():
        rwkv.eval()
        speak_sequence = []
        next_token = start_with_token
        args = rwkv.args
        B, T = 1, 1
        C = args.n_embd
        H = args.dim_att // args.head_size
        assert C == H * args.head_size

        if states is None:
            new_states = RWKVStates.create(
                args.n_layer,
                B,
                args.n_embd,
                args.n_head,
                args.head_size,
                rwkv.args.device,
                rwkv.args.dtype,
            )
        else:
            new_states = states

        occurrence = {}

        for i in range(max_tokens):
            next_token, new_states, _, _ = speak_next_token(
                rwkv,
                last_token=next_token,
                states=new_states,
                occurrence=occurrence,
                temperature=temperature,
                top_p=top_p,
                alpha_presence=alpha_presence,
                alpha_frequency=alpha_frequency,
                alpha_decay=alpha_decay,
            )
            if 49 <= next_token <= 58:
                pass
            elif next_token not in occurrence:
                occurrence[next_token] = 1
            else:
                occurrence[next_token] += 1
            speak_sequence.append(next_token)
            xxx = tokenizer.decode([next_token])
            print(xxx, end="")
            sys.stdout.flush()

            if next_token in token_stop:
                if m_postfix_token:
                    postfix = torch.tensor([[m_postfix_token]], dtype=torch.long).to(
                        rwkv.args.device
                    )
                    logits, new_states = rwkv(postfix, new_states)
                speak_sequence.pop()
                break
        return speak_sequence, new_states

@torch.no_grad()
def speak_next_token_batch(
    rwkv,
    last_tokens_batch: torch.Tensor,
    states: RWKVStates,
    batch_occurrence: List[dict],
    temperature: float,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    token_ban: List[int] = [],
    batch_allow_tokens: List[List[int]] = None,
):
    assert len(batch_occurrence) == last_tokens_batch.shape[0]
    logits, states = rwkv(
        last_tokens_batch,
        states,
    )

    for b, occurence in enumerate(batch_occurrence):
        for xxx in occurence:
            occurence[xxx] *= alpha_decay
        for n in occurence:
            logits[b, -1, n] -= alpha_presence + occurence[n] * alpha_frequency
    if batch_allow_tokens is not None:
        mask = torch.full_like(logits[:, -1, :], -1e38, device=logits.device)
        for b, allow_tokens in enumerate(batch_allow_tokens):
            mask[b, allow_tokens] = logits[b, -1, allow_tokens]
        logits[:, -1, :] = mask
    for n in token_ban:
        logits[:, -1, n] = -1e38

    last_logits = logits[:, -1:, :]
    next_tokens_batch = sample_logits_batch(last_logits, temperature, top_p)
    return next_tokens_batch, states


@torch.no_grad()
def batch_generate(
    rwkv,
    start_with_batch_tokens: torch.Tensor,
    states: RWKVStates,
    temperature: float,
    top_p: float,
    alpha_presence: float,
    alpha_frequency: float,
    alpha_decay: float,
    token_stop=[65535],
    max_tokens=500,
    m_postfix_token=[],
    token_ban=[],
):
    rwkv.eval()
    next_tokens_batch = start_with_batch_tokens
    args = rwkv.args
    B, T = start_with_batch_tokens.size()
    if states is None:
        new_states = RWKVStates.create(
            args.n_layer,
            B,
            args.n_embd,
            args.n_head,
            args.head_size,
            rwkv.args.device,
            rwkv.args.dtype,
        )
    else:
        new_states = states

    speak_sequences = [[] for _ in range(B)]
    batch_occurrence = [{} for _ in range(B)]
    end_sample_batch = []
    out_states = RWKVStates.empty(
        args.n_layer,
        B,
        args.n_embd,
        args.n_head,
        args.head_size,
        rwkv.args.device,
        rwkv.args.dtype,
    )
    for i in range(max_tokens):
        next_tokens_batch, new_states = speak_next_token_batch(
            rwkv,
            last_tokens_batch=next_tokens_batch,
            states=new_states,
            batch_occurrence=batch_occurrence,
            temperature=temperature,
            top_p=top_p,
            alpha_presence=alpha_presence,
            alpha_frequency=alpha_frequency,
            alpha_decay=alpha_decay,
            token_ban=token_ban,
        )
        next_tokens_batch = next_tokens_batch.unsqueeze(-1)
        for b, next_token in enumerate(next_tokens_batch):
            next_token = int(next_token)
            if 49 <= next_token <= 58:
                pass
            elif next_token not in batch_occurrence[b]:
                batch_occurrence[b][next_token] = 1
            else:
                batch_occurrence[b][next_token] += 1
            if b not in end_sample_batch:
                speak_sequences[b].append(next_token)

                if next_token in token_stop:
                    target_state = states.batchof(b)
                    if m_postfix_token:
                        postfix = torch.tensor([m_postfix_token], dtype=torch.long).to(
                            rwkv.args.device
                        )
                        _, target_state = rwkv(postfix, target_state)
                    out_states.tmix_shift_states[:, b, :] = copy.deepcopy(
                        target_state.tmix_shift_states[:, 0, :]
                    )
                    out_states.cmix_shift_states[:, b, :] = copy.deepcopy(
                        target_state.cmix_shift_states[:, 0, :]
                    )
                    out_states.tmix_wkv_states[:, b, :, :, :] = copy.deepcopy(
                        target_state.tmix_wkv_states[:, 0, :, :, :]
                    )
                    end_sample_batch.append(b)
                    speak_sequences[b].pop()

        if len(end_sample_batch) == B:
            break

    for b in range(B):
        if b not in end_sample_batch:
            out_states.tmix_shift_states[:, b, :] = copy.deepcopy(
                new_states.batchof(b).tmix_shift_states[:, 0, :]
            )
            out_states.cmix_shift_states[:, b, :] = copy.deepcopy(
                new_states.batchof(b).cmix_shift_states[:, 0, :]
            )
            out_states.tmix_wkv_states[:, b, :, :, :] = copy.deepcopy(
                new_states.batchof(b).tmix_wkv_states[:, 0, :, :, :]
            )

    return speak_sequences, out_states


def batch_block_infer(
    rwkv, tokens_batches: list, state: RWKVStates, chunk_len: int = 512
):
    out = None
    t_batches = [x[:chunk_len] for x in tokens_batches]
    last_len = len(t_batches[0])
    assert all(len(x) == last_len for x in t_batches)
    while last_len > 0:
        out, state = rwkv(t_batches, state)
        tokens_batches = [x[chunk_len:] for x in tokens_batches]
        t_batches = [x[:chunk_len] for x in tokens_batches]
        last_len = len(t_batches[0])
    return out, state


def batch_chat(
    rwkv,
    start_with_tokens_batch: List[List[int]],
    tokenizer,
    stop_with_tokens: List[int],
    stop_supp_tokens: List[int],
    temp: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    decay_penalty: float,
    batch_state: RWKVStates = None,
    max_resp_len: int = 512,
    token_ban=[],
):
    """
    效率有待改进，暂时用这个
    """
    B = len(start_with_tokens_batch)

    # 组装states
    if batch_state is None:
        states = [
            RWKVStates.create(
                rwkv.args.n_layer,
                1,
                rwkv.args.n_embd,
                rwkv.args.n_head,
                rwkv.args.head_size,
                rwkv.args.device,
                rwkv.args.dtype,
            )
            for _ in range(B)
        ]
    else:
        states = batch_state.unbind()
        assert len(states) == B

    # 判断start_with_tokens_batch中所有元素都长度相等
    if all(len(x) == len(start_with_tokens_batch[0]) for x in start_with_tokens_batch):
        out, states_in = batch_block_infer(
            rwkv,
            [x[:-1] for x in start_with_tokens_batch],
            sum(states[1:], states[0]),
            512,
        )
    else:
        new_states = []
        for i in range(B):
            out, new_state = batch_block_infer(
                rwkv,
                [x[:-1] for x in start_with_tokens_batch[i : i + 1]],
                states[i],
                512,
            )
            new_states.append(new_state)

        states_in = sum(new_states[1:], new_states[0])

    start_with_batch_tokens = torch.tensor(
        [x[-1:] for x in start_with_tokens_batch],
        dtype=torch.long,
        device=rwkv.args.device,
    )

    # 开始推理
    speak_sequences_batch, out_states = batch_generate(
        rwkv,
        start_with_batch_tokens,
        states_in,
        temp,
        top_p,
        presence_penalty,
        frequency_penalty,
        decay_penalty,
        token_stop=stop_with_tokens,
        max_tokens=max_resp_len,
        m_postfix_token=stop_supp_tokens,
        token_ban=token_ban,
    )

    speak_texts_batch = [tokenizer.decode(speak_sequences_batch[i]) for i in range(B)]
    return speak_sequences_batch, speak_texts_batch, out_states


def calc_cross_entropy_loss(logits, targets, mask, reshape_mode=False):
    if reshape_mode:
        mask = mask.reshape(-1)
    else:
        mask = mask.view(-1)
    sum_mask = torch.sum(mask).item()

    if sum_mask == mask.shape[0]:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # print('rank', rwkv.global_rank, 'loss', loss.item())
    else:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none"
        )
        # loss_raw = loss
        loss = torch.sum(loss * mask)
        if sum_mask > 0:
            loss = loss / sum_mask
    return L2Wrap.apply(loss, logits)


def train_forward(
    rwkv,
    batch_idx,
    batch_masks,
    states=None,
):
    # batch_idx [B,N] torch.LongTensor
    # batch_masks [B, N] torch.tensor
    # data process
    inputs = batch_idx[:, :-1]  # torch.LongTensor
    targets = batch_idx[:, 1:]  # torch.LongTensor
    if batch_masks == None:
        print("===no mask==")
        batch_masks = torch.where(batch_idx != 0, 1, torch.tensor(0))
        # batch_masks = [int(x != 0) for x in idx]
    batch_masks = batch_masks[:, 1:]

    # process mask
    mask = batch_masks.to(device=next(rwkv.parameters()).device, dtype=torch.float32)
    logits, new_states = rwkv(inputs, states)
    
    loss= calc_cross_entropy_loss(logits, targets, mask, reshape_mode=True)


    return loss, new_states


def train_forward_from_embds(
    rwkv, embds_latent, targets, batch_masks_targets, states=None
):
    # embds_latent [B,N,n_embd] torch.tensor
    # targets [B, N] torch.LongTensor
    # batch_masks [B, N] torch.tensor
    embds_latent = embds_latent.to(next(rwkv.blocks[0].parameters()).device)

    mask = batch_masks_targets.to(
        device=next(rwkv.parameters()).device, dtype=torch.float32
    )

    out_embds, logits, new_states = rwkv.forward_from_embeddings(embds_latent, states)

    targets = targets.to(next(rwkv.parameters()).device)
    LM_loss = calc_cross_entropy_loss(logits, targets, mask)
    return out_embds, LM_loss, new_states


def get_gpu_memory_usage():
    """获取当前GPU内存使用情况"""
    if not torch.cuda.is_available():
        return 0, 0, 0

    # 获取当前设备
    device = torch.cuda.current_device()

    # 获取分配的内存
    allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB

    # 获取缓存的内存
    reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB

    # 获取总内存
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB

    return allocated, reserved, total


def clear_gpu_memory(force=False):
    MEM_MONITOR_THRESHOLD = 0.8  # 默认为80%
    """清理GPU内存"""
    if not torch.cuda.is_available():
        return

    allocated, reserved, total = get_gpu_memory_usage()
    usage_ratio = allocated / total if total > 0 else 0

    # 只有当内存使用率超过阈值或强制清理时才执行
    if force or usage_ratio > MEM_MONITOR_THRESHOLD:
        print(f"清理GPU内存 (使用率: {usage_ratio:.2f})")
        gc.collect()
        torch.cuda.empty_cache()

    # 清理后再次检查
    new_allocated, _, _ = get_gpu_memory_usage()
    print(
        f"清理后GPU内存: {new_allocated:.2f}GB (减少了 {allocated - new_allocated:.2f}GB)"
    )
