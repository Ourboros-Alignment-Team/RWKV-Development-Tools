import torch
from typing import Optional
import os
from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
full_parent_dir = os.path.dirname(os.path.abspath(__file__))
flags = [
    "-res-usage",
    f"-D_C_={HEAD_SIZE}",
    "--use_fast_math",
    "-O3",
    "-Xptxas -O3",
    "--extra-device-vectorization",
]
load(
    name="wbs",
    sources=[
        f"{full_parent_dir}/cuda/wkv7_cuda_eval.cu",
        f"{full_parent_dir}/cuda/wkv7_op_eval.cpp",
    ],
    is_python_module=False,
    verbose=True,
    extra_cuda_cflags=flags,
)



class WindBacksteppingS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b, state):
        B, T, H, C = w.shape
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        s = torch.empty(
            B, H, T, C, C, dtype=torch.float32, device=w.device
        )
        s[:, :, 0] = state.to(device=s.device,dtype=s.dtype)
        s=s.contiguous()
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wbs.forward(w, q, k, v, z, b, y, s, sa)
        new_state = s[:, :, -1]
        return y, new_state


def RUN_CUDA_RWKV7s(q, w, k, v, a, b, state):
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC // 64, 64) for i in [q, w, k, v, a, b]]
    y, new_state=WindBacksteppingS.apply(w, q, k, v, a, b,state)
    return y.view(B, T, HC),new_state
