import math
import triton
import torch
from jaxtyping import Bool, Float
from einops import rearrange
from torch.library import wrap_triton, triton_op
import triton.testing

from trifast.triton import (
    _fwd,
    _bwd_kv,
    _bwd_q,
    _bwd_b,
)

@triton_op("trifast::triangle_attention", mutates_args={})
def _triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sm_scale = q.shape[-1] ** -0.5

    bh, _, n, dim = q.shape

    def grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), n, bh)

    o = torch.zeros_like(q)
    l = torch.zeros((bh, n, n), device=q.device, dtype=torch.float32)

    CLOSEST_N = 2 ** int(math.ceil(math.log2(n)))

    # fmt: off
    wrap_triton(_fwd)[grid](
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        l, l.stride(0), l.stride(1), l.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        neg_inf=torch.finfo(q.dtype).min,
        #sm_scale=sm_scale, N=n, H=h, DIM=dim,
        sm_scale=sm_scale, N=n, H=4, DIM=dim, #h or bh? - hacky to hardcode here
        CLOSEST_N=CLOSEST_N,
    )

    return o, l

def triangle_attention(
    q: Float[torch.Tensor, "b h n n d"],
    k: Float[torch.Tensor, "b h n n d"],
    v: Float[torch.Tensor, "b h n n d"],
    b: Float[torch.Tensor, "b h n n"],
    mask: Bool[torch.Tensor, "b n n"],
) -> Float[torch.Tensor, "b h n n d"]:
    o, _ = _triangle_attention(q, k, v, b, mask)
    return o