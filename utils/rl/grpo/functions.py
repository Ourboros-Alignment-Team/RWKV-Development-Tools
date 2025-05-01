import torch
import torch.nn.functional as F


def zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)



def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Reward组标准化
    """
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def get_batch_log_probs(
    rwkv,
    t_batch_tokens: torch.Tensor,
    begin_with_states=None,
) -> torch.Tensor:
    # 对应前向传播，求logp
    logits, states = rwkv(
        idx=t_batch_tokens,
        states=begin_with_states,
    )
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=t_batch_tokens[:, 1:],
    )
    return log_probs

