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
        B, C, T = v_features.size()
        assert T % self.chunk_len == 0
        v_features = v_features.to(
            next(self.parameters()).device, next(self.parameters()).dtype
        )
        v_features = v_features.transpose(1, 2).reshape(B, T // self.chunk_len, -1)
        v_features = self.adapter(v_features)
        v_features = self.gelu(v_features)
        v_features = self.ln(v_features)
        if self.use_vit:
            pe = posemb_sincos_1d(v_features)
            v_features = rearrange(v_features, "b ... d -> b (...) d") + pe
            v_features = self.transformer(v_features)
        return v_features


# 之后改为接入到现有冻结TTS的潜空间，或者改为flow
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
        B, T, C = v_features.size()
        v_features = v_features.to(
            next(self.parameters()).device, next(self.parameters()).dtype
        )
        v_features = self.adapter(v_features)
        v_features = v_features.reshape(B, self.v_feature_dim, -1)

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
        x = self.adapter(features)
        xl = self.vocos_backbone_l(x)
        xr = self.vocos_backbone_r(x)
        xl = self.head_l(xl)
        xr = self.head_r(xr)
        return torch.cat([xl.unsqueeze(0),xr.unsqueeze(0)],dim=1)
    
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
        _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = disc_engine.multiperioddisc(
            y=y,
            y_hat=y_hat,
        )
        _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = disc_engine.multiresddisc(
            y=y,
            y_hat=y_hat,
        )
        loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
        loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
        loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
        loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
        loss_fm_mp = self.feat_matching_loss(
            fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp
        ) / len(fmap_rs_mp)
        loss_fm_mrd = self.feat_matching_loss(
            fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd
        ) / len(fmap_rs_mrd)

        mel_loss = self.melspec_loss(y_hat, y)
        return loss_gen_mp, loss_gen_mrd, loss_fm_mp, loss_fm_mrd, mel_loss

    def discriminator_losses(self, disc_engine, y, y_hat):
        real_score_mp, gen_score_mp, _, _ = disc_engine.multiperioddisc(
            y=y.detach(),
            y_hat=y_hat.detach(),
        )
        real_score_mrd, gen_score_mrd, _, _ = disc_engine.multiresddisc(
            y=y.detach(),
            y_hat=y_hat.detach(),
        )
        loss_mp, loss_mp_real, _ = self.disc_loss(
            disc_real_outputs=real_score_mp,
            disc_generated_outputs=gen_score_mp,
        )
        loss_mrd, loss_mrd_real, _ = self.disc_loss(
            disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
        )
        loss_mp /= len(loss_mp_real)
        loss_mrd /= len(loss_mrd_real)
        return loss_mp, loss_mrd
