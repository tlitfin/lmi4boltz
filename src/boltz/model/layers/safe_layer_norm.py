"""SafeLayerNorm: workaround for PyTorch LayerNorm CUDA uint32 overflow bug.

When a tensor has >= 2**32 elements, PyTorch's CUDA LayerNorm kernel uses 32-bit
integer indexing which overflows, producing wrong (or NaN) outputs for elements
beyond index 2**32.

This subclass splits the input into chunks that stay below the overflow threshold
and concatenates the results, matching the semantics of nn.LayerNorm exactly.

Reference: https://github.com/pytorch/pytorch/issues/149301
"""

import torch
import torch.nn as nn


class SafeLayerNorm(nn.LayerNorm):
    """Workaround for the PyTorch uint32 overflow bug in the CUDA LayerNorm kernel.

    Reference: https://github.com/pytorch/pytorch/issues/149301
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if input.is_cuda and input.numel() >= 2**32:
            input_shape = input.shape
            input_flat = input.view(-1, input_shape[-1])
            output = torch.empty_like(input_flat)
            # Maximum number of rows that keeps total elements below the limit
            batch_len = max(1, int((2**32 - 1) // input_shape[-1]))
            for i in range(0, input_flat.shape[0], batch_len):
                output[i : i + batch_len] = super().forward(
                    input_flat[i : i + batch_len]
                )
            return output.view(input_shape)
        return super().forward(input)


# Monkey-patch torch.nn.LayerNorm globally so all existing code automatically
# uses the safe implementation without any other file changes.
# Reference: https://github.com/pytorch/pytorch/issues/149301
torch.nn.LayerNorm = SafeLayerNorm
