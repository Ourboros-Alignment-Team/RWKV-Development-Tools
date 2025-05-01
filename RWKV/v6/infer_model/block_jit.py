
import torch
import torch.nn as nn
from .time_mix_jit import TimeMix
from .channel_mix_jit import ChannelMix


from torch.nn import functional as F
from torch.nn import functional
import math


class EncoderDecoder(nn.Module):
    def __init__(self, head_size, emb, r=64):
        super().__init__()
        self.encode = nn.Linear(r, head_size, bias=False)
        self.encode.weight.data = torch.eye(r, head_size)
        #self.encode_decode_middle =  nn.Linear(r, r, bias=False)
        #self.encode_decode_middle.data =  torch.eye(r, r)
        # self.decode = nn.Linear(head_size, r, bias=False)
        # self.decode.weight.data = torch.eye(head_size, r)
        self.encode_ln = nn.LayerNorm(head_size)
        self.encode_dropout = nn.Dropout(0.05)
        # 用于融合 att_shift 信息的全连接层

    def forward(self,  att_state, ffn_shift):
        # x.shape (1 40 64 64)
        # att_shift (1, 2560)
        x = att_state[1]
        att_shift = att_state[0]

        if self.training:
            x = self.encode_dropout(x)

        x = x.to(dtype=torch.bfloat16) 
        #x = self.encode_ln(x)
        x = self.encode(x)
        #x = self.encode_decode_middle(x)
        #x = self.decode(x)          
        # output (1 40 64 64)
        return (att_shift, x.float()), ffn_shift


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)


        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = TimeMix(args, layer_id)

        self.ffn = ChannelMix(args, layer_id)
        
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x, layer_state):
        # if self.layer_id == 0:
        #     x = self.ln0(x)


        att_out, layer_state = self.att(
            self.ln1(x),
            layer_state
        )

        if self.args.dropout > 0.0:
            # Handle with dropout
            x = self.drop0(x + att_out)
            ffn_out, layer_state = self.ffn(
                self.ln2(x),
                layer_state
            )
            x = self.drop1(x + ffn_out)
        else:
            # Handle without dropout
            x = x + att_out
            ffn_out, layer_state = self.ffn(
                self.ln2(x),
                layer_state
            )
            x = x + ffn_out

        return x, layer_state
