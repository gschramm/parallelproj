# %%
from __future__ import annotations

import torch
import parallelproj


# device variable (cpu or cuda) that determines whether calculations
# are performed on the cpu or cuda gpu
if parallelproj.cuda_present:
    dev = "cuda"
else:
    dev = "cpu"

# %%


class PoissonLogLGradLayer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator, data: torch.Tensor, contam: torch.Tensor
    ) -> torch.Tensor:

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        y = torch.zeros_like(x)
        ratio2 = torch.zeros_like(data)

        for i in range(x.shape[0]):
            exp = operator(x[i, 0, ...].detach()) + contam[i, ...]
            ratio = data[i, ...] / exp
            ratio2[i, ...] = data[i, ...] / (exp**2)
            y[i, 0, ...] = operator.adjoint(ratio - 1)

        ctx.ratio2 = ratio2

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        if grad_output is None:
            return None, None, None, None
        else:
            operator = ctx.operator
            ratio2 = ctx.ratio2

            x = torch.zeros_like(grad_output)

            for i in range(grad_output.shape[0]):
                exp = operator(grad_output[i, 0, ...].detach())
                x[i, 0, ...] = -operator.adjoint(ratio2[i, ...] * exp)

            return x, None, None, None


# %%


class PoissonEMOperator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        operator: parallelproj.LinearOperator,
        data: torch.Tensor,
        contam: torch.Tensor,
        sens_img: torch.Tensor,
    ) -> torch.Tensor:

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        x_sens_ratio = torch.zeros_like(x)
        ratio2 = torch.zeros_like(data)
        mult_update = torch.zeros_like(x)

        for i in range(x.shape[0]):
            exp = operator(x[i, 0, ...].detach()) + contam[i, ...]
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
    ) -> tuple[torch.Tensor | None, None, None, None, None]:
        if grad_output is None:
            return None, None, None, None, None
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

            return x, None, None, None, None

# %%

class EMNet(torch.nn.Module):  
    def __init__(self) -> None:
        super().__init__()
        self._data_fid_layer = PoissonLogLGradLayer.apply

    def forward(self, 
        x: torch.Tensor,
        operator: parallelproj.LinearOperator,
        data: torch.Tensor,
        contam: torch.Tensor,
        sens_img: torch.Tensor) -> torch.tensor:

        return x + (x / sens_img) * self._data_fid_layer(x, operator, data, contam)



# %%
torch.manual_seed(0)

A = torch.tensor(
    [[1.5, 0.5, 0.1], [0.3, 2.1, 0.2], [0.9, 1.2, 2.1], [1.0, 2.0, 0.5]],
    dtype=torch.float64, device = dev,
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

data_t = torch.rand(
    (batch_size,) + proj.out_shape,
    device=dev,
    dtype=torch.float64,
    requires_grad=False,
)

contam_t = torch.rand(
    (batch_size,) + proj.out_shape,
    device=dev,
    dtype=torch.float64,
    requires_grad=False,
)

sens_t = torch.zeros_like(xt)
ones_data = torch.ones(proj.out_shape, device=dev, dtype=torch.float64)
for i in range(batch_size):
    sens_t[i, 0, ...] = proj.adjoint(ones_data)

# %%
# Define the forward and backward projection layers
# -------------------------------------------------

logLgrad_layer = PoissonLogLGradLayer.apply
f2 = logLgrad_layer(xt, proj, data_t, contam_t)

em_layer = PoissonEMOperator.apply
em_update_1 = em_layer(xt, proj, data_t, contam_t, sens_t)

em_net = EMNet()
em_update_2 = em_net(xt, proj, data_t, contam_t, sens_t)

manual_em_update = torch.zeros_like(xt)
for i in range(batch_size):
    manual_em_update[i, 0, ...] = (xt[i, 0, ...] / sens_t[i, 0, ...]) * (
        A.T @ (data_t[i, ...] / (A @ xt[i, 0, ...] + contam_t[i, ...]))
    )

# %%
# Check whether the gradients are calculated correctly
# ----------------------------------------------------

test_logLgrad = torch.autograd.gradcheck(
    logLgrad_layer,
    (xt, proj, data_t, contam_t),
)

test_em = torch.autograd.gradcheck(
    em_layer,
    (xt, proj, data_t, contam_t, sens_t),
)

test_em2 = torch.autograd.gradcheck(
    em_net,
    (xt, proj, data_t, contam_t, sens_t),
)