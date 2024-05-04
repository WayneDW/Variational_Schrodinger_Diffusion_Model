import numpy as np
import abc
from tqdm import tqdm
from functools import partial
import torch

import util
import loss
from ipdb import set_trace as debug


def _assert_increasing(name, ts):
    assert (ts[1:] > ts[:-1]).all(), '{} must be strictly increasing'.format(name)

def build(opt, p, q):
    print(util.magenta("build base sde..."))
    return VPSDE(opt, p, q)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, p, q):
        self.opt = opt
        self.dt = opt.T / opt.interval
        self.p = p # data distribution
        self.q = q # prior distribution

        self.b_min = opt.beta_min
        self.b_max = opt.beta_max
        self.b_r = opt.beta_r

    @abc.abstractmethod
    def _f(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, x, t):
        raise NotImplementedError

    def f(self, x, t, direction):
        sign = 1. if direction=='forward' else -1.
        return sign * self._f(x, t)

    def g(self, t):
        return self._g(t)

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x)*np.sqrt(dt)

    def propagate(self, t, x, z, direction, f=None, dw=None, dt=None):
        g = self.g(t)
        f = self.f(x, t, direction) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(x,dt) if dw is None else dw
        return x + (f + g*z)*dt + g*dw

    def propagate_ode(self, t, x, z, z_f, direction, f=None, dt=None):
        g = self.g(t)
        f = self.f(x, t, direction) if f is None else f
        dt = self.dt if dt is None else dt
        dsm_score = z + z_f # map fb-sde score to dsm score
        return x + (f - g * z_f + 0.5*g*dsm_score)*dt

    def propagate_x0_trick(self, x, policy, direction):
        """ propagate x0 by a tiny step """
        t0  = torch.Tensor([0])
        dt0 = self.opt.t0 - 0
        assert dt0 > 0
        z0  = policy(x,t0)
        return self.propagate(t0, x, z0, direction, dt=dt0)

    def sample_traj(self, ts, policy, corrector=None, apply_trick=True, save_traj=True, adaptive_prior=None, policy_f=None):

        # first we need to know whether we're doing forward or backward sampling
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward', 'backward']

        # set up ts and init_distribution
        #_assert_increasing('ts', ts) # smaller NFE used fake time stamps
        init_dist = self.p if direction=='forward' else self.q
        if direction == 'backward' and adaptive_prior != None:
            print('General prior for backward process.')
            init_dist = adaptive_prior
        ts = ts if direction=='forward' else torch.flip(ts,dims=[0])

        x = init_dist.sample() # [bs, x_dim]

        apply_trick1, apply_trick2, apply_trick3 = compute_tricks_condition(opt, apply_trick, direction)
        # [trick 1] propagate img (x0) by a tiny step
        if apply_trick1: x = self.propagate_x0_trick(x, policy, direction)

        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:])) if save_traj else None
        zs = torch.empty_like(xs) if save_traj else None

        # don't use tqdm for fbsde since it'll resample every itr
        _ts = tqdm(ts,desc=util.yellow("Propagating Dynamics..."))
        for idx, t in enumerate(_ts):
            _t=t if idx == ts.shape[0] - 1 else ts[idx+1]

            f = self.f(x, t, direction)
            z = policy(x, t)
            z_f = policy_f(x, t) if policy_f != None else None

            dw = self.dw(x)

            t_idx = idx if direction == 'forward' else len(ts)-idx-1
            if save_traj:
                xs[:,t_idx,...] = x
                zs[:,t_idx,...] = z

            # [trick 2] zero out dw
            if apply_trick2(t_idx=t_idx): dw = torch.zeros_like(dw)

            if policy_f != None and direction == 'backward':
                x = self.propagate_ode(t, x, z, z_f, direction, f=f)
            else:
                x = self.propagate(t, x, z, direction, f=f, dw=dw)

        x_term = x

        res = [xs, zs, x_term]
        return res


def compute_tricks_condition(opt, apply_trick, direction):
    if not apply_trick:
        return False, lambda t_idx: False,  False

    # [trick 1] source: Song et al ICLR 2021 Appendix C
    # when: (i) image, (ii) p -> q, (iii) t0 > 0,
    # do:   propagate img (x0) by a tiny step.
    apply_trick1 = (util.is_image_dataset(opt) and direction == 'forward' and opt.t0 > 0)

    # [trick 2] Improved DPM
    # when: (i) image, (ii) q -> p, (iii) vp, (iv) last sampling step
    # do:   zero out dw
    trick2_cond123 = (util.is_image_dataset(opt) and direction=='backward')
    def _apply_trick2(trick2_cond123, t_idx):
        return trick2_cond123 and t_idx==0
    apply_trick2 = partial(_apply_trick2, trick2_cond123=trick2_cond123)

    # [trick 3] NCSNv2, Alg 1
    # when: (i) image, (ii) q -> p, (iii) last sampling step
    # do:   additional denoising step
    trick3_cond12 = (util.is_image_dataset(opt) and direction=='backward')
    def _apply_trick3(trick3_cond12, t_idx):
        return trick3_cond12 and t_idx==0
    apply_trick3 = partial(_apply_trick3, trick3_cond12=trick3_cond12)

    return apply_trick1, apply_trick2, apply_trick3


class VPSDE(BaseSDE):
    def __init__(self, opt, p, q):
        super(VPSDE,self).__init__(opt, p, q)

    def _f(self, x, t):
        return compute_vp_drift_coef(t, self.b_min, self.b_max, self.b_r) * x

    def _g(self, t):
        return compute_vp_diffusion(t, self.b_min, self.b_max, self.b_r)

####################################################
##  Implementation of SDE analytic kernel         ##
##  Ref: https://arxiv.org/pdf/2011.13456v2.pdf,  ##
##       page 15-16, Eq (30,32,33)                ##
####################################################


""" Generalized Song, Yang linear schedule to non-linear """
def compute_vp_diffusion(t, b_min, b_max, b_r=1., T=1.):
    return torch.sqrt(b_min+(t/T)**b_r*(b_max-b_min))

def compute_vp_drift_coef(t, b_min, b_max, b_r=1.):
    g = compute_vp_diffusion(t, b_min, b_max, b_r)
    return -0.5 * g**2

def compute_vp_kernel_mean_scale(t, b_min, b_max, b_r=1.):
    return torch.exp(-0.5/(b_r+1)*t**(b_r+1)*(b_max-b_min)-0.5*t*b_min)

def compute_vp_kernel_mean_scale_matrix(t, D, b_min, b_max, b_r=1.):
    time_scalars = -0.5/(b_r+1)*t**(b_r+1)*(b_max-b_min)-0.5*t*b_min
    time_vary_D = torch.einsum('t,ij->tij', time_scalars, D)
    return torch.linalg.matrix_exp(time_vary_D)

def compute_variance(ts, D, b_min, b_max, b_r=1.):
    dim = D.shape[0]
    C_H_power = torch.block_diag(-D, D.t())
    C_H_power[:dim, dim:] = 2. * torch.eye(dim)
    integrate_beta = 0.5 / (b_r + 1)*ts**(b_r + 1) * (b_max - b_min) + 0.5 * ts * b_min
    C_H_pair = torch.linalg.matrix_exp(torch.einsum('t,ij->tij', integrate_beta, C_H_power))

    Initial_Matrix = torch.cat((torch.zeros_like(torch.eye(dim)), torch.eye(dim)), dim=0)
    #Initial_Matrix = torch.cat((torch.eye(dim), torch.eye(dim)), dim=0)

    C_H = torch.einsum('tij,jk->tik', C_H_pair, Initial_Matrix)

    C = C_H[:, : dim, :]
    H = C_H[:, dim: , :]
    Covariance = torch.einsum('tij,tjk->tik', C, torch.linalg.inv(H))

    L = torch.linalg.cholesky(Covariance)
    invL = torch.linalg.inv(L.mH)
    return Covariance, L, invL

def compute_vp_xs_label_matrix(opt, x0, sqrt_betas, mean_scales, D, L, invL, samp_t_idx):
    """ return xs.shape == [batch_x, batch_t, *x_dim]  """
    x_dim = opt.data_dim
    assert x_dim==list(x0.shape[1:])
    batch_x, batch_t = x0.shape[0], len(samp_t_idx)

    # p(x_t|x_0) = N(mean_scale * x_0, std_t^2)
    # x_t = mean_scale * x_0 + std_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim)
    mean_scale_t = mean_scales[samp_t_idx]
    L_t = L[samp_t_idx]
    invL_t = invL[samp_t_idx]

    analytic_xs = torch.einsum('tij,btj->bti', L_t, noise) + torch.einsum('tij,bj->bti', mean_scale_t, x0)
    #""" torch.linalg.inv is OK in [N, 2, 2] when N!= 2 but may be less robust in [2, 2, 2] """
    part_label = - torch.einsum('tij,btj->bti', invL_t, noise)
    sqrt_beta_t = sqrt_betas[samp_t_idx].reshape(1,-1,*([1,]*len(x_dim))) # shape = [1,batch_t,1,1,1]
    label = part_label * sqrt_beta_t
    # change DSM score to SB-FBSDE score
    label -= (torch.einsum('ij,btj->bti', 0.5 * (torch.eye(2) - D), analytic_xs) * sqrt_beta_t)
    return analytic_xs, label

def compute_vp_xs_label(opt, x0, sqrt_betas, mean_scales, samp_t_idx):
    """ return xs.shape == [batch_x, batch_t, *x_dim]  """
    x_dim = opt.data_dim

    assert x_dim==list(x0.shape[1:])
    batch_x, batch_t = x0.shape[0], len(samp_t_idx)

    # p(x_t|x_0) = N(mean_scale * x_0, std_t^2)
    # x_t = mean_scale * x_0 + std_t * noise
    noise = torch.randn(batch_x, batch_t, *x_dim)
    mean_scale_t = mean_scales[samp_t_idx].reshape(1,-1,*([1,]*len(x_dim))) # shape = [1,batch_t,1,1,1]
    std_t = torch.sqrt(1 - mean_scale_t**2)
    analytic_xs = std_t * noise + mean_scale_t * x0[:,None,...]
    # score_of_p = -1/std_t^2 (x_t - mean_scale_t * x_0) = -noise/std_t
    # hence, g * score_of_p = - noise / std_t * sqrt_beta_t
    sqrt_beta_t = sqrt_betas[samp_t_idx].reshape(1,-1,*([1,]*len(x_dim))) # shape = [1,batch_t,1,1,1]
    label = - noise / std_t * sqrt_beta_t

    return analytic_xs, label

def get_xs_label_computer(opt, ts, A): 
    sqrt_betas = compute_vp_diffusion(ts, opt.beta_min, opt.beta_max, opt.beta_r)
    if opt.forward_net == 'Linear' and not opt.DSM_baseline:
        D = torch.eye(A.shape[0]) - 2 * A
        #print(util.green(f'D matrix\n {torch.round(D, decimals=3)}'))
        mean_scales = compute_vp_kernel_mean_scale_matrix(ts, D, opt.beta_min, opt.beta_max, opt.beta_r)
        _, L, invL = compute_variance(ts, D, opt.beta_min, opt.beta_max, opt.beta_r)
        fn = compute_vp_xs_label_matrix
        kwargs = dict(opt=opt, sqrt_betas=sqrt_betas, mean_scales=mean_scales, D=D, L=L, invL=invL)
    else:
        mean_scales = compute_vp_kernel_mean_scale(ts, opt.beta_min, opt.beta_max, opt.beta_r)
        fn = compute_vp_xs_label
        kwargs = dict(opt=opt, sqrt_betas=sqrt_betas, mean_scales=mean_scales)

    return partial(fn, **kwargs)
