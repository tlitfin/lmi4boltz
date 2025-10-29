import torch
from torch import Tensor, nn

from boltz.model.layers import initialize as init


class TriangleMultiplicationOutgoing(nn.Module):
    """TriangleMultiplicationOutgoing."""

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        init.bias_init_one_(self.norm_in.weight)
        init.bias_init_zero_(self.norm_in.bias)

        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)

        init.bias_init_one_(self.norm_out.weight)
        init.bias_init_zero_(self.norm_out.bias)

        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor, triangle_mult_gate_nchunks: int=1) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x

        chunk_sizes = torch.linspace(0, x.shape[2], steps=triangle_mult_gate_nchunks+1, device=x.device).long()
        x = torch.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3]*2), device=x.device)

        for i in range(triangle_mult_gate_nchunks):
            start = chunk_sizes[i].item()
            end = chunk_sizes[i+1].item()
            x[:,:,start:end,:] = self.p_in(x_in[:,:,start:end,:])*self.g_in(x_in[:,:,start:end,:]).sigmoid()

        #x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        #x = x * mask.unsqueeze(-1)
        x *= mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        # This becomes a bottleneck - can easily be chunked
        x = torch.einsum("bikd,bjkd->bijd", a, b)
        del a, b

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x


class TriangleMultiplicationIncoming(nn.Module):
    """TriangleMultiplicationIncoming."""

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        init.bias_init_one_(self.norm_in.weight)
        init.bias_init_zero_(self.norm_in.bias)

        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)

        init.bias_init_one_(self.norm_out.weight)
        init.bias_init_zero_(self.norm_out.bias)

        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor, triangle_mult_gate_nchunks: int=1) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        
        chunk_sizes = torch.linspace(0, x.shape[2], steps=triangle_mult_gate_nchunks+1, device=x.device).long()
        x = torch.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3]*2), device=x.device)
        for i in range(triangle_mult_gate_nchunks):
            start = chunk_sizes[i].item()
            end = chunk_sizes[i+1].item()
            x[:,:,start:end,:] = self.p_in(x_in[:,:,start:end,:])*self.g_in(x_in[:,:,start:end,:]).sigmoid()


        #x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        #x = x * mask.unsqueeze(-1)
        x *= mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        # This becomes a bottleneck - can easily be chunked
        x = torch.einsum("bkid,bkjd->bijd", a, b)
        del a, b

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x
