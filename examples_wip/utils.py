import array_api_compat.torch as torch
from array_api_compat import device
import parallelproj

class LMPoissonLogLDescent(torch.autograd.Function):
    """
    Function representing a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, lm_fwd_operator: parallelproj.LinearOperator, contam_list: torch.Tensor, adjoint_ones: torch.Tensor
    ) -> torch.Tensor:
        """forward pass of listmode negative Poisson logL gradient descent layer

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size = 1, 1, num_voxels_x, num_voxels_y, num_voxels_z)
        lm_fwd_operator : parallelproj.LinearOperator in listmode
            linear operator that can act on a single 3D image
        contam_list : torch.Tensor
            listmode contamination (scatter and randoms) with dimension (batch_size = 1, num_events)
        adjoint_ones : torch.Tensor
            minibatch of 3D images of adjoint of the (full not listmode) linear operator applied to ones
            (batch_size = 1, num_voxels_x, num_voxels_y, num_voxels_z)

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size = 1, lm_fwd_operator.out_shape)
        """

        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        ctx.set_materialize_grads(False)
        ctx.lm_fwd_operator = lm_fwd_operator

        batch_size = x.shape[0]

        if batch_size != 1:
            raise ValueError(
                "Because of object dependent LM linear operators, the batch size must be 1 for now"
            )

        g = torch.zeros(
            (batch_size,) + lm_fwd_operator.in_shape, dtype=x.dtype, device=device(x)
        )

        z_lm = torch.zeros(
            (batch_size,) + lm_fwd_operator.out_shape, dtype=x.dtype, device=device(x)
        )

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            z_lm[i, ...] = lm_fwd_operator(x[i, 0, ...].detach()) + contam_list[i, ...]
            g[i, ...] = adjoint_ones[i, ...] - lm_fwd_operator.adjoint(1/z_lm[i, ...])
        
        # save z for the backward pass
        ctx.save_for_backward(z_lm)

        return g

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, operator.out_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size = 1, 1, lm_fwd_operator.in_shape)
        """

        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes four input arguments (x, lm_fwd_operator, contam_list, adjoint_ones)
        # we have to return four arguments (the last three are None)
        if grad_output is None:
            return None, None, None, None
        else:
            lm_fwd_operator = ctx.lm_fwd_operator
            z_lm = ctx.saved_tensors[0]


            batch_size = grad_output.shape[0]

            if batch_size != 1:
                raise ValueError(
                    "Because of object dependent LM linear operators, the batch size must be 1 for now"
                )

            x = torch.zeros(
                (batch_size, 1) + lm_fwd_operator.in_shape,
                dtype=grad_output.dtype,
                device=device(grad_output),
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, 0, ...] = lm_fwd_operator.adjoint(lm_fwd_operator(grad_output[i, ...].detach()) / z_lm[i, ...]**2)

            return x, None, None, None

