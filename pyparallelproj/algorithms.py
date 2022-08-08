import numpy as np


class PDHG_L2_Denoise:
    """
    First-order primal dual image denoising with weighted L2 data fidelity term.
    Solves the problem: argmax_x( \sum_i w_i*(x_i - img_i)**2 + norm(grad_operator x) )

    Arguments
    ---------

    prior ... gradient-based prior

    xp    ... numpy / cupy module to use

    Keyword arguments
    -----------------

    nonneg   ... (bool) whether to clip negative values in solution - default False

    verbose  ... (bool) whether to print some diagnostic output - default False

    Keyword arguments
    -----------------


    niter    ... (int) number of iterations to run - default 200

    cost     ... (1d array) 1d output array for cost calcuation - default None
    """

    def __init__(self, prior, xp, nonneg=False, verbose=False):
        self.prior = prior
        self._xp = xp
        self.nonneg = nonneg
        self.verbose = verbose

    def init(self, img, weights):
        """
        Arguments
        ---------

        img      ... (numpy/cupy array) with the image to be denoised

        weights  ... (scalar or numpy/cupy array) with weights for data fidelity term
        """
        self.img = img
        self.x = self.img.copy()
        self.xbar = self.img.copy()
        self.weights = weights

        self.y = self._xp.zeros((self.x.ndim, ) + self.x.shape,
                                dtype=self._xp.float32)

        if isinstance(weights, self._xp.ndarray):
            self.gam = self.weights.min()
        else:
            self.gam = self.weights

        self.ndim = len([x for x in self.img.shape if x > 1])
        self.grad_op_norm = self._xp.sqrt(self.ndim * 4)

        self.tau = 1.0 / self.gam
        self.sig = 1.0 / (self.tau * self.grad_op_norm**2)

        self.epoch_counter = 0
        self.cost = np.array([], dtype=np.float32)

    def run_update(self):
        # (1) forward model
        self.y += self.sig * self.prior.gradient_operator.forward(self.xbar)

        # (2) proximity operator
        self.y = self.prior.gradient_norm.prox_convex_dual(self.y,
                                                           sigma=self.sig)

        # (3) adjoint model
        xnew = self.x - self.tau * self.prior.gradient_operator.adjoint(self.y)

        # (4) apply proximity of G
        xnew = (xnew + self.weights * self.img * self.tau) / (
            1.0 + self.weights * self.tau)
        if self.nonneg:
            xnew = self._xp.clip(xnew, 0, None)

        # (5) calculate the new stepsizes
        self.theta = 1.0 / np.sqrt(1 + 2 * self.gam * self.tau)
        self.tau *= self.theta
        self.sig /= self.theta

        # (6) update variables
        self.xbar = xnew + self.theta * (xnew - self.x)
        self.x = xnew.copy()

    def run(self, niter, calculate_cost=False):
        cost = np.zeros(niter, dtype=np.float32)

        for it in range(niter):
            if self.verbose:
                print(f"epoch {self.epoch_counter+1}")
            self.run_update()

            if calculate_cost:
                cost[it] = self.calculate_cost()

            self.epoch_counter += 1

        self.cost = np.concatenate((self.cost, cost))

    def calculate_cost(self):
        return 0.5 * (self.weights *
                      (self.x - self.img)**2).sum() + self.prior(self.x)


# -----------------------------------------------------------------------------------------------------------------
class OSEM:

    def __init__(self, em_sino, acq_model, contam_sino, xp, verbose=True):
        self.em_sino = em_sino
        self.acq_model = acq_model
        self.contam_sino = contam_sino
        self.img_shape = tuple(self.acq_model.proj.img_dim)
        self.verbose = True
        self._xp = xp

    def init(self, x=None):
        self.epoch_counter = 0
        self.cost = np.array([], dtype=np.float32)

        if x is None:
            self.x = self._xp.full(self.img_shape, 1.0, dtype=self._xp.float32)
        else:
            self.x = x.copy()

        # calculate the sensitivity images
        self.sens_imgs = self._xp.zeros(
            (self.acq_model.proj.nsubsets, ) + self.img_shape,
            dtype=self._xp.float32)

        for isub in range(self.acq_model.proj.nsubsets):
            if self.verbose:
                print(f"calculating sensitivity image {isub}", end="\r")
            self.sens_imgs[isub, ...] = self.acq_model.adjoint(self._xp.ones(
                self.acq_model.proj.subset_sino_shapes[isub]),
                                                               isub=isub)
        if self.verbose:
            print("")

    def run_EM_update(self, isub):
        ss = self.acq_model.proj.subset_slices[isub]
        exp_sino = self.acq_model.forward(self.x, isub) + self.contam_sino[ss]
        self.x *= (self.acq_model.adjoint(self.em_sino[ss] / exp_sino, isub) /
                   self.sens_imgs[isub, ...])

    def run(self, niter, calculate_cost=False):
        cost = np.zeros(niter, dtype=np.float32)

        for it in range(niter):
            if self.verbose:
                print(f"iteration {self.epoch_counter+1}")
            for isub in range(self.acq_model.proj.nsubsets):
                if self.verbose:
                    print(f"subset {isub+1}", end="\r")
                self.run_EM_update(isub)

            if calculate_cost:
                cost[it] = self.eval_neg_poisson_logL()

            self.epoch_counter += 1

        self.cost = np.concatenate((self.cost, cost))

    def eval_neg_poisson_logL(self):
        cost = 0
        for isub in range(self.acq_model.proj.nsubsets):
            ss = self.acq_model.proj.subset_slices[isub]
            exp_sino = self.acq_model.forward(self.x,
                                              isub) + self.contam_sino[ss]
            cost += float(
                (exp_sino - self.em_sino[ss] * self._xp.log(exp_sino)).sum())

        return cost


# -----------------------------------------------------------------------------------------------------------------
class OSEM_EMTV(OSEM):

    def __init__(self,
                 em_sino,
                 acq_model,
                 contam_sino,
                 prior,
                 xp,
                 verbose=True):
        super().__init__(em_sino, acq_model, contam_sino, xp, verbose=verbose)
        self.prior = prior

        self.denoiser = PDHG_L2_Denoise(self.prior,
                                        self._xp,
                                        verbose=False,
                                        nonneg=True)
        self.tiny = self._xp.finfo(self._xp.float32).tiny
        self.fmax = self._xp.finfo(self._xp.float32).max

    def run_EMTV_update(self, isub, niter_denoise):
        # the denominator for the weights (image*beta) can be 0
        # we clip "tiny" values to make sure that weights.max() stays finite
        denom = self._xp.clip(self.prior.beta * self.x, self.tiny, None)
        weights = self.sens_imgs.sum(axis=0) / denom
        # weights = self.acq_model.proj.nsubsets*self.sens_imgs[isub,...] / denom
        # we also have to make sure that the weights.min() > 0 since gam = weights.min() and tau = 1 / gamma
        weights = self._xp.clip(weights, 10 * self.tiny, self.fmax)

        # OSEM update
        super().run_EM_update(isub)

        # weighted denoising step
        self.denoiser.init(self.x, weights)
        self.denoiser.run(niter_denoise, calculate_cost=False)

        self.x = self.denoiser.x.copy()

    def run(self, niter, niter_denoise=30, calculate_cost=False):
        cost = np.zeros(niter, dtype=np.float32)

        for it in range(niter):
            if self.verbose:
                print(f"iteration {self.epoch_counter+1}")
            for isub in range(self.acq_model.proj.nsubsets):
                if self.verbose:
                    print(f"subset {isub+1}", end="\r")
                self.run_EMTV_update(isub, niter_denoise)

            if calculate_cost:
                cost[it] = self.eval_cost()

            self.epoch_counter += 1

        self.cost = np.concatenate((self.cost, cost))

    def eval_cost(self):
        return super().eval_neg_poisson_logL() + float(self.prior(self.x))


# -----------------------------------------------------------------------------------------------------------------
class SPDHG:

    def __init__(self,
                 em_sino,
                 acq_model,
                 contam_sino,
                 prior,
                 xp,
                 verbose=True):
        self.em_sino = em_sino
        self.acq_model = acq_model
        self.contam_sino = contam_sino
        self.img_shape = tuple(self.acq_model.proj.img_dim)
        self.verbose = True
        self._xp = xp
        self.prior = prior

    def init(self, x=None, y=None, gamma=1.0, rho=0.999, rho_grad=0.999):
        self.epoch_counter = 0
        self.cost = np.array([], dtype=np.float32)
        self.gamma = gamma
        self.rho = rho
        self.rho_grad = rho_grad
        self.nsubsets = self.acq_model.proj.nsubsets

        if x is None:
            self.x = self._xp.full(self.img_shape, 1.0, dtype=self._xp.float32)
        else:
            self.x = x.copy()

        if self.prior.beta == 0:
            self.p_g = 0
        else:
            self.p_g = 0.5
            # norm of the gradient operator = sqrt(ndim*4)
            self.ndim = len([x for x in self.img_shape if x > 1])
            self.grad_op_norm = self._xp.sqrt(self.ndim * 4)

        self.p_p = (1 - self.p_g) / self.nsubsets

        # initialize y for data, z and zbar
        if y is None:
            self.y = self._xp.zeros(self.em_sino.shape, dtype=self._xp.float32)
            self.z = self._xp.zeros(self.img_shape, dtype=self._xp.float32)
        else:
            self.y = y.copy()
            self.z = self.acq_model.adjoint(self.y)

        self.zbar = self.z.copy()

        # calculate S for the gradient operator
        if self.p_g > 0:
            self.S_g = self.gamma * self.rho_grad / self.grad_op_norm
            self.T_g = self.p_g * self.rho_grad / (self.gamma *
                                                   self.grad_op_norm)

        # calculate the "step sizes" S_i for the PET fwd operator
        self.S_i = []
        ones_img = self._xp.ones(self.img_shape, dtype=self._xp.float32)

        for i in range(self.nsubsets):
            ss = self.acq_model.proj.subset_slices[i]
            tmp = self.acq_model.forward(ones_img, isub=i)
            tmp = self._xp.clip(tmp, tmp[tmp > 0].min(),
                                None)  # clip 0 values before division
            self.S_i.append(self.gamma * self.rho / tmp)

        # calculate the step size T
        T_i = self._xp.zeros((self.nsubsets, ) + self.img_shape,
                             dtype=self._xp.float32)

        self.sens_img = self._xp.zeros(self.img_shape, dtype=self._xp.float32)

        for i in range(self.nsubsets):
            # get the slice for the current subset
            ss = self.acq_model.proj.subset_slices[i]
            # generate a subset sinogram full of ones
            ones_sino = self._xp.ones(
                self.acq_model.proj.subset_sino_shapes[i],
                dtype=self._xp.float32)
            tmp = self.acq_model.adjoint(ones_sino, isub=i)

            self.sens_img += tmp
            T_i[i, ...] = self.rho * self.p_p / (self.gamma * tmp)

        # take the element-wise min of the T_i's of all subsets
        self.T = T_i.min(axis=0)

        if self.p_g > 0:
            self.T = self._xp.clip(self.T, None, self.T_g)

        # allocate arrays for gradient operations
        self.y_grad = self._xp.zeros((len(self.img_shape), ) + self.img_shape,
                                     dtype=self._xp.float32)

        # indices where sensitivity is 0 (e.g. outside ring)
        self.zero_sens_inds = self._xp.where(self.sens_img == 0)

        # intitial subset sequence
        self.subset_sequence = np.random.permutation(
            np.arange(int(self.nsubsets / (1 - self.p_g))))

    def run_update(self, isub):
        self.x = self._xp.clip(self.x - self.T * self.zbar, 0, None)

        # select a random subset
        i = self.subset_sequence[isub]

        self.x[self.zero_sens_inds] = 0
        self.zbar[self.zero_sens_inds] = 0
        self.z[self.zero_sens_inds] = 0

        if i < self.nsubsets:
            # PET subset update
            if self.verbose:
                print(f"step {isub} subset {i+1}", end="\r")
            ss = self.acq_model.proj.subset_slices[i]

            y_plus = self.y[ss] + self.S_i[i] * (
                self.acq_model.forward(self.x, isub=i) + self.contam_sino[ss])

            # apply the prox for the dual of the poisson logL
            y_plus = 0.5 * (y_plus + 1 - self._xp.sqrt(
                (y_plus - 1)**2 + 4 * self.S_i[i] * self.em_sino[ss]))
            dz = self.acq_model.adjoint(y_plus - self.y[ss], isub=i)

            # update variables
            self.z += dz
            self.y[ss] = y_plus.copy()
            self.zbar = self.z + dz / self.p_p
        else:
            print(f"step {isub} gradient update", end="\r")
            y_grad_plus = self.y_grad + self.S_g * self.prior.gradient_operator.forward(
                self.x)

            # apply the prox for the gradient norm
            y_grad_plus = self.prior.beta * self.prior.gradient_norm.prox_convex_dual(
                y_grad_plus / self.prior.beta,
                sigma=self.S_g / self.prior.beta)

            dz = self.prior.gradient_operator.adjoint(y_grad_plus -
                                                      self.y_grad)
            dz[self.zero_sens_inds] = 0

            # update variables
            self.z += dz
            self.y_grad = y_grad_plus.copy()
            self.zbar = self.z + dz / self.p_g

    def run(self, niter, calculate_cost=False):
        for it in range(niter):
            self.subset_sequence = np.random.permutation(
                np.arange(int(self.nsubsets / (1 - self.p_g))))
            if self.verbose:
                print(f"\niteration {self.epoch_counter+1}")
            for isub in range(self.subset_sequence.shape[0]):
                self.run_update(isub)

            if calculate_cost:
                self.cost = np.concatenate((self.cost, [self.eval_cost()]))

            self.epoch_counter += 1

    def eval_cost(self):
        cost = 0
        for isub in range(self.acq_model.proj.nsubsets):
            ss = self.acq_model.proj.subset_slices[isub]
            exp_sino = self.acq_model.forward(self.x,
                                              isub) + self.contam_sino[ss]
            cost += float(
                (exp_sino - self.em_sino[ss] * self._xp.log(exp_sino)).sum())

        cost += float(self.prior(self.x))

        return cost


# -----------------------------------------------------------------------------------------------------------------
class LM_OSEM:

    def __init__(self, lm_acq_model, contam_list, xp, verbose=True):
        self.lm_acq_model = lm_acq_model
        self.contam_list = contam_list
        self.img_shape = tuple(self.lm_acq_model.proj.img_dim)
        self.verbose = True
        self._xp = xp

    def init(self, sens_img, nsubsets, x=None):
        self.epoch_counter = 0
        self.sens_img = sens_img
        self.nsubsets = nsubsets

        if x is None:
            self.x = self._xp.full(self.img_shape, 1.0, dtype=self._xp.float32)
        else:
            self.x = x.copy()

    def run_EM_update(self, isub):
        exp_list = (self.lm_acq_model.forward(self.x, isub, self.nsubsets) +
                    self.contam_list[isub::self.nsubsets])
        self.x *= (self.lm_acq_model.adjoint(self.nsubsets / exp_list, isub,
                                             self.nsubsets) / self.sens_img)

    def run(self, niter):
        for it in range(niter):
            if self.verbose:
                print(f"iteration {self.epoch_counter+1}")
            for isub in range(self.nsubsets):
                if self.verbose:
                    print(f"subset {isub+1}", end="\r")
                self.run_EM_update(isub)
            self.epoch_counter += 1


# --------------------------------------------------------------------------------------------
class LM_OSEM_EMTV(LM_OSEM):

    def __init__(self, lm_acq_model, contam_list, prior, xp, verbose=True):
        super().__init__(lm_acq_model, contam_list, xp, verbose=verbose)
        self.prior = prior

        self.denoiser = PDHG_L2_Denoise(self.prior,
                                        self._xp,
                                        verbose=False,
                                        nonneg=True)
        self.fmax = self._xp.finfo(self._xp.float32).max
        self.tiny = self._xp.finfo(self._xp.float32).tiny

    def run_EMTV_update(self, isub, niter_denoise):
        # the denominator for the weights (image*beta) can be 0
        # we clip "tiny" values to make sure that weights.max() stays finite
        denom = self._xp.clip(self.prior.beta * self.x, self.tiny, None)
        weights = self.sens_img / denom
        # we also have to make sure that the weights.min() > 0 since gam = weights.min() and tau = 1 / gamma
        weights = self._xp.clip(weights, 10 * self.tiny, self.fmax)

        # LM_OSEM update
        super().run_EM_update(isub)

        # weighted denoising step
        self.denoiser.init(self.x, weights)
        self.denoiser.run(niter_denoise, calculate_cost=False)

        self.x = self.denoiser.x.copy()

    def run(self, niter, niter_denoise=30, calculate_cost=False):
        for it in range(niter):
            if self.verbose:
                print(f"iteration {self.epoch_counter+1}")
            for isub in range(self.nsubsets):
                if self.verbose:
                    print(f"subset {isub+1}", end="\r")
                self.run_EMTV_update(isub, niter_denoise)
            self.epoch_counter += 1


# --------------------------------------------------------------------------------------------
class LM_SPDHG:

    def __init__(self, lm_acq_model, contam_list, event_counter, prior, xp):
        self.lm_acq_model = lm_acq_model
        self.contam_list = contam_list
        self.event_counter = event_counter
        self.prior = prior
        self._xp = xp

        self.img_shape = tuple(self.lm_acq_model.proj.img_dim)
        self.verbose = True

    def init(self,
             sens_img,
             nsubsets,
             x=None,
             gamma=1.0,
             rho=0.999,
             rho_grad=0.999):
        self.epoch_counter = 0
        self.sens_img = sens_img
        self.gamma = gamma
        self.rho = rho
        self.rho_grad = rho_grad
        self.nsubsets = nsubsets

        if x is None:
            self.x = self._xp.full(self.img_shape, 1.0, dtype=self._xp.float32)
        else:
            self.x = x.copy()

        self.mu = self.event_counter.count(self.lm_acq_model.events)

        if self.prior.beta == 0:
            self.p_g = 0
        else:
            self.p_g = 0.5
            # norm of the gradient operator = sqrt(ndim*4)
            self.ndim = len([x for x in self.img_shape if x > 1])
            self.grad_op_norm = self._xp.sqrt(self.ndim * 4)

        self.p_p = (1 - self.p_g) / self.nsubsets

        # initialize y for data, z and zbar
        self.y = 1 - (self.mu /
                      (self.lm_acq_model.forward(self.x) + self.contam_list))
        self.z = self.sens_img + self.lm_acq_model.adjoint(
            (self.y - 1) / self.mu)
        self.zbar = self.z.copy()

        # calculate S for the gradient operator
        if self.p_g > 0:
            self.S_g = self.gamma * self.rho_grad / self.grad_op_norm
            self.T_g = self.p_g * self.rho_grad / (self.gamma *
                                                   self.grad_op_norm)

        # calculate the "step sizes" S_i for the PET fwd operator
        self.S_i = []
        ones_img = self._xp.ones(self.img_shape, dtype=self._xp.float32)

        for i in range(self.nsubsets):
            ss = slice(i, None, nsubsets)
            tmp = self.lm_acq_model.forward(ones_img, i, self.nsubsets)
            tmp = self._xp.clip(tmp, tmp[tmp > 0].min(),
                                None)  # clip 0 values before division
            self.S_i.append(self.gamma * self.rho / tmp)

        # calculate the step size T
        self.T = self._xp.zeros_like(sens_img)
        inds = self._xp.where(self.sens_img > 0)
        self.T[inds] = (self.nsubsets * self.p_p * self.rho /
                        (self.gamma * self.sens_img[inds]))

        if self.p_g > 0:
            self.T = self._xp.clip(self.T, None, self.T_g)

        # allocate arrays for gradient operations
        self.y_grad = self._xp.zeros((len(self.img_shape), ) + self.img_shape,
                                     dtype=self._xp.float32)

        # indices where sensitivity is 0 (e.g. outside ring)
        self.zero_sens_inds = self._xp.where(self.sens_img == 0)

        # intitial subset sequence
        self.subset_sequence = np.random.permutation(
            np.arange(int(self.nsubsets / (1 - self.p_g))))

    def run_update(self, isub):
        self.x = self._xp.clip(self.x - self.T * self.zbar, 0, None)

        # select a random subset
        i = self.subset_sequence[isub]

        self.x[self.zero_sens_inds] = 0
        self.zbar[self.zero_sens_inds] = 0
        self.z[self.zero_sens_inds] = 0

        if i < self.nsubsets:
            # PET subset update
            if self.verbose:
                print(f"step {isub} subset {i+1}", end="\r")
            ss = slice(i, None, self.nsubsets)

            y_plus = self.y[ss] + self.S_i[i] * (self.lm_acq_model.forward(
                self.x, i, self.nsubsets) + self.contam_list[ss])

            # apply the prox for the dual of the poisson logL
            y_plus = 0.5 * (y_plus + 1 - self._xp.sqrt(
                (y_plus - 1)**2 + 4 * self.S_i[i] * self.mu[ss]))
            dz = self.lm_acq_model.adjoint((y_plus - self.y[ss]) / self.mu[ss],
                                           i, self.nsubsets)

            # update variables
            self.z += dz
            self.y[ss] = y_plus.copy()
            self.zbar = self.z + dz / self.p_p
        else:
            print(f"step {isub} gradient update", end="\r")
            y_grad_plus = self.y_grad + self.S_g * self.prior.gradient_operator.forward(
                self.x)

            # apply the prox for the gradient norm
            y_grad_plus = self.prior.beta * self.prior.gradient_norm.prox_convex_dual(
                y_grad_plus / self.prior.beta,
                sigma=self.S_g / self.prior.beta)

            dz = self.prior.gradient_operator.adjoint(y_grad_plus -
                                                      self.y_grad)
            dz[self.zero_sens_inds] = 0

            # update variables
            self.z += dz
            self.y_grad = y_grad_plus.copy()
            self.zbar = self.z + dz / self.p_g

    def run(self, niter):
        for it in range(niter):
            self.subset_sequence = np.random.permutation(
                np.arange(int(self.nsubsets / (1 - self.p_g))))
            if self.verbose:
                print(f"\niteration {self.epoch_counter+1}")
            for isub in range(self.subset_sequence.shape[0]):
                self.run_update(isub)
            self.epoch_counter += 1
