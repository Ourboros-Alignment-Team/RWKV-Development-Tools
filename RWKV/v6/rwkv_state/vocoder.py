import torch.nn as nn
import torch


from .vocos.heads import ISTFTHead
from .vocos.vocos import VocosBackbone
from .vocos.loss import (
    DiscriminatorLoss,
    GeneratorLoss,
    FeatureMatchingLoss,
    MelSpecReconstructionLoss,
)
from .vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from .transformer import Transformer

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import torchaudio
from einops import rearrange
import torch.nn.init as init


class EncodecEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_pretrained(dir):
        return torch.load(dir)


def posemb_sincos_1d(patches, temperature=10000, dtype=torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device=device)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe.type(dtype)

class TrackMixing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 方案1：创建和输入同形状的权重矩阵 [dim,1]，广播到N
        self.P = nn.Parameter(torch.zeros(dim, 1))
        
        self.dim = dim
    
    def forward(self, a, b):
        # a, b shape: [B, dim, N]
        diff = b - a
        # 点乘：P[dim,1] * diff[B,dim,N] -> [B,dim,N]
        weighted_diff = self.P * diff
        result = a + weighted_diff
        return result

class AdapterE(nn.Module):
    def __init__(self, chunk_len=15, v_feature_dim=128, n_embd=4096, use_vit=False):
        super().__init__()
        self.chunk_len = chunk_len
        self.adapter = nn.Linear(
            chunk_len * v_feature_dim, n_embd
        )  # 消融线性层、vit1d等，ctx太短不适合rwkv
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(n_embd)
        self.use_vit = use_vit
        if use_vit:
            self.transformer = Transformer(
                dim=n_embd,
                depth=1,
                heads=8,
                mlp_dim=n_embd,
                dim_head=64,
            )

    def forward(self, v_features: torch.Tensor):
        pass
        return v_features


class AdapterD(nn.Module):
    def __init__(self, chunk_len=15, v_feature_dim=128, n_embd=4096, n_layers=5):
        super().__init__()
        self.chunk_len = chunk_len
        self.v_feature_dim = v_feature_dim
        self.n_layers = n_layers
        self.n_embd = n_embd

        self.adapter = nn.Linear(n_embd, chunk_len * v_feature_dim)
        
        init.zeros_(self.adapter.weight)
        if self.adapter.bias is not None:
            init.zeros_(self.adapter.bias)


    def forward(self, v_features):
        pass
        return v_features


class VoiceDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adapter = AdapterD(
            chunk_len=config.vocoder.adapter.chunk_len,
            v_feature_dim=config.vocoder.vocos_backbone.input_channels,
            n_embd=config.vocoder.adapter.n_embd,
            n_layers=config.vocoder.adapter.n_layers,
        )
        self.vocos_backbone_l = VocosBackbone(**vars(config.vocoder.vocos_backbone))
        self.vocos_backbone_r = VocosBackbone(**vars(config.vocoder.vocos_backbone))
        self.head_l = ISTFTHead(**vars(config.vocoder.head))
        self.head_r = ISTFTHead(**vars(config.vocoder.head))
        self._initialize_weights()

    def forward(self, features):
        pass
    
    def _initialize_weights(self):
        # Initialize the weights of the VocosBackbone layers with zeros
        for module in [self.vocos_backbone_l, self.vocos_backbone_r, self.head_l, self.head_r]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    init.zeros_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv1d):
                    init.zeros_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)


class VocoderDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()
        self.config = config

    def build_engine(self):
        ds_config = {
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1,
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e6,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e6,
                "contiguous_gradients": True,
            },
        }
        disc_params = [
            {
                "params": self.multiperioddisc.parameters(),
                "lr": self.config.training.disc_lr,
            },
            {
                "params": self.multiresddisc.parameters(),
                "lr": self.config.training.disc_lr,
            },
        ]
        opt_disc = DeepSpeedCPUAdam(
            disc_params,
            betas=(0.8, 0.9),
            adamw_mode=True,
            amsgrad=False,
            bias_correction=True,
        )
        model_engine, optim, _, _ = deepspeed.initialize(
            model=self, optimizer=opt_disc, config=ds_config
        )
        return model_engine, optim


class Losses(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gen_loss = GeneratorLoss()
        self.disc_loss = DiscriminatorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(
            sample_rate=config.vocoder.sample_rate
        )

    def generator_losses(self, disc_engine, y, y_hat):
       pass

    def discriminator_losses(self, disc_engine, y, y_hat):
        pass
