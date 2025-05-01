import torch
import torch.nn as nn
from .time_mix import TimeMix
from .channel_mix import ChannelMix

from RWKV.v6.state import LayerRWKVStates
from torch.nn import functional as F



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

    def forward(self, x, layer_state: LayerRWKVStates):
        # if self.layer_id == 0:
        #     x = self.ln0(x)

        
        att_out, layer_state = self.att(self.ln1(x), layer_state)

        if self.args.dropout > 0.0:
            # Handle with dropout
            x = self.drop0(x + att_out)
            ffn_out, ffn_state = self.ffn(self.ln2(x), ffn_state)
            x = self.drop1(x + ffn_out)
        else:
            # Handle without dropout
            x = x + att_out
            ffn_out, layer_state = self.ffn(self.ln2(x), layer_state)
            x = x + ffn_out

        return x, layer_state