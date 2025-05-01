import torch
import torch.nn as nn
from .time_mix import TimeMix
from .channel_mix import ChannelMix


from torch.nn import functional as F
from torch.nn import functional
from RWKV.v6.state import LayerRWKVStates
import math
import copy


class EncoderDecoderLora(nn.Module):
    def __init__(self, head_size, emb, r=8, dropout=0.01, alpha=32):
        super().__init__()
        self.encode_weight = nn.Parameter(torch.empty((head_size, head_size)))
        self.encode = nn.Parameter(torch.empty(r, head_size))
        self.decode = nn.Parameter(torch.empty(head_size, r))
        self.encode_dropout = nn.Dropout(dropout)
        self.encode_ln = nn.LayerNorm(head_size)
        self.scaling = alpha / r

        # 用于融合 att_shift 信息的全连接层
        nn.init.kaiming_uniform_(self.encode_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.encode, a=math.sqrt(5))
        nn.init.zeros_(self.decode)

    def forward(self, att_state, ffn_shift):
        # x.shape (1 40 64 64)
        # att_shift (1, 2560)
        x = att_state[1]
        att_shift = att_state[0]
        x = x.to(dtype=torch.bfloat16)
        x = functional.linear(x, self.encode_weight) + self.scaling * functional.linear(
            functional.linear(self.encode_dropout(x), self.encode), self.decode
        )
        # output (1 40 64 64)
        return (att_shift, x.float()), ffn_shift


class EncoderDecoder(nn.Module):
    def __init__(self, head_size, emb, r=64):
        super().__init__()
        self.encode = nn.Linear(r, head_size, bias=False)
        self.encode.weight.data = torch.eye(r, head_size)
        # self.encode_decode_middle =  nn.Linear(r, r, bias=False)
        # self.encode_decode_middle.data =  torch.eye(r, r)
        # self.decode = nn.Linear(head_size, r, bias=False)
        # self.decode.weight.data = torch.eye(head_size, r)
        self.encode_ln = nn.LayerNorm(head_size)
        self.encode_dropout = nn.Dropout(0.05)
        # 用于融合 att_shift 信息的全连接层

    def forward(self, layer_state):
        # x.shape (1 40 64 64)
        # att_shift (1, 2560)
        x = layer_state.tmix_wkv_states

        if self.training:
            x = self.encode_dropout(x)

        x = x.to(dtype=torch.bfloat16)
        # x = self.encode_ln(x)
        x = self.encode(x)
        layer_state.tmix_wkv_states = x.float()
        # x = self.encode_decode_middle(x)
        # x = self.decode(x)
        # output (1 40 64 64)
        return layer_state



class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        # add state encoder decoder
        if self.args.lora.train_state:
            self.state_encoder = EncoderDecoder(self.args.head_size, self.args.n_embd)
            # self.state_encoder = LSTMModelEncoder()

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = TimeMix(args, layer_id)

        self.ffn = ChannelMix(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x, layer_state: LayerRWKVStates):
        if self.layer_id == 0:
            x = self.ln0(x)

        # state encoder
        if self.args.lora.train_state:
            layer_state = self.state_encoder(
                layer_state
            )
        # print(next(self.parameters()).dtype,"<----<")
        # print(next(self.ln1.parameters()).dtype,"<<<<<")
        # print(x.dtype,">>>>>") #为什么这里是float
        att_out, layer_state = self.att(self.ln1(x), layer_state)

        if self.args.dropout > 0.0:
            # Handle with dropout
            x = self.drop0(x + att_out)
            ffn_out, layer_state = self.ffn(self.ln2(x), layer_state)
            x = self.drop1(x + ffn_out)
        else:
            # Handle without dropout
            x = x + att_out
            ffn_out, layer_state = self.ffn(self.ln2(x), layer_state)
            x = x + ffn_out

        return x, layer_state
