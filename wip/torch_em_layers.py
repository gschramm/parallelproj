# %%

from __future__ import annotations

import array_api_compat.torch as torch
import matplotlib.pyplot as plt
import parallelproj
from array_api_compat import device


# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu
if parallelproj.cuda_present:
    dev = "cuda"
else:
    dev = "cpu"

# %%


class PoissonLogLGradOperator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator, data: torch.Tensor
    ) -> torch.Tensor:

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        y = torch.zeros_like(x)
        ratio2 = torch.zeros_like(data)

        for i in range(x.shape[0]):
            exp = operator(x[i, 0, ...].detach())
            ratio = data[i, ...] / exp
            ratio2[i, ...] = data[i, ...] / (exp**2)
            y[i, 0, ...] = operator.adjoint(1 - ratio)

        ctx.ratio2 = ratio2

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        if grad_output is None:
            return None, None, None
        else:
            operator = ctx.operator
            ratio2 = ctx.ratio2

            x = torch.zeros_like(grad_output)

            for i in range(grad_output.shape[0]):
                exp = operator(grad_output[i, 0, ...].detach())
                x[i, 0, ...] = operator.adjoint(ratio2[i, ...] * exp)

            return x, None, None


# %%


class PoissonEMOperator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        operator: parallelproj.LinearOperator,
        data: torch.Tensor,
        sens_img: torch.Tensor,
    ) -> torch.Tensor:

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        y = torch.zeros_like(x)
        x_sens_ratio = torch.zeros_like(x)
        ratio2 = torch.zeros_like(data)
        mult_update = torch.zeros_like(x)

        for i in range(x.shape[0]):
            exp = operator(x[i, 0, ...].detach())
            ratio = data[i, ...] / exp
            ratio2[i, ...] = data[i, ...] / (exp**2)
            x_sens_ratio[i, 0, ...] = x[i, 0, ...] / sens_img[i, 0, ...]
            mult_update[i, 0, ...] = operator.adjoint(ratio) / sens_img[i, 0, ...]

        ctx.ratio2 = ratio2
        ctx.x_sens_ratio = x_sens_ratio
        ctx.mult_update = mult_update

        return mult_update * x

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        if grad_output is None:
            return None, None, None, None
        else:
            operator = ctx.operator
            ratio2 = ctx.ratio2
            mult_update = ctx.mult_update
            x_sens_ratio = ctx.x_sens_ratio

            x = torch.zeros_like(grad_output)

            for i in range(grad_output.shape[0]):
                exp = operator(
                    x_sens_ratio[i, 0, ...] * grad_output[i, 0, ...].detach()
                )
                x[i, 0, ...] = grad_output[i, 0, ...] * mult_update[
                    i, 0, ...
                ] - operator.adjoint(ratio2[i, ...] * exp)

            return x, None, None, None


# %%
torch.manual_seed(0)

A = torch.tensor(
    [[1.5, 0.5, 0.1], [0.3, 2.1, 0.2], [0.9, 1.2, 2.1], [1.0, 2.0, 0.5]],
    dtype=torch.float64,
)
proj = parallelproj.MatrixOperator(A)

# %%
# Define a mini batch of input and output tensors
# -----------------------------------------------

batch_size = 2

xt = torch.rand(
    (batch_size, 1) + proj.in_shape,
    device=dev,
    dtype=torch.float64,
    requires_grad=True,
)

yt = torch.rand(
    (batch_size,) + proj.out_shape,
    device=dev,
    dtype=torch.float64,
)

sens = torch.zeros_like(xt)
ones_data = torch.ones(proj.out_shape, device=dev, dtype=torch.float64)
for i in range(batch_size):
    sens[i, 0, ...] = proj.adjoint(ones_data)

# %%
# Define the forward and backward projection layers
# -------------------------------------------------

logLgrad_layer = PoissonLogLGradOperator.apply
f2 = logLgrad_layer(xt, proj, yt)

em_layer = PoissonEMOperator.apply
f3 = em_layer(xt, proj, yt, sens)

em_update = torch.zeros_like(xt)
for i in range(batch_size):
    em_update[i, 0, ...] = (xt[i, 0, ...] / sens[i, 0, ...]) * (
        A.T @ (yt[i, ...] / (A @ xt[i, 0, ...]))
    )

# %%
# Check whether the gradients are calculated correctly
# ----------------------------------------------------

test_logLgrad = torch.autograd.gradcheck(
    logLgrad_layer,
    (xt, proj, yt),
)

test_em = torch.autograd.gradcheck(
    em_layer,
    (xt, proj, yt, sens),
)
