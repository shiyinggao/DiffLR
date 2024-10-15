import torch
import torch.nn.functional as F
import numpy as np
import math
from random import random
from tqdm import tqdm

from torch import nn, einsum, repeat_interleave
from functools import partial
from collections import namedtuple

from einops import rearrange, reduce
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Linear(dim, default(dim_out, dim))   
    )

def Downsample(dim, dim_out = None):
    return nn.Linear(dim, default(dim_out, dim))  

class WeightStandardizedLinear(nn.Linear):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> 1', 'mean')
        var = reduce(weight, 'o ... -> 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.linear(x, normalized_weight, self.bias)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusionCondition(nn.Module):
    def __init__(
        self,
        model,
        device,
        *,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0.0,
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        auto_normalize = True 
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type


        self.sampling_timesteps = default(sampling_timesteps, timesteps) 

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta


        register_buffer = lambda name, val:\
            self.register_buffer(name, val.to(torch.float32).to(device))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod',
                        torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev)\
                             / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1',
                        betas * torch.sqrt(alphas_cumprod_prev)\
                            / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2',
                        (1. - alphas_cumprod_prev) * torch.sqrt(alphas)\
                            / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)


        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return(
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return(
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0)\
            /extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, x_cond, t, x_self_cond = None, clip_x_start = False):
        model_output = self.model(x, x_cond, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x,  t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, x_cond, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, x_cond, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, x_cond, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        (model_mean,
         _,
         model_log_variance,
         x_start) = self.p_mean_variance(x=x, x_cond=x_cond,
                                         t=batched_times,
                                         x_self_cond=x_self_cond,
                                         clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. 
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise     # v = log(\sigma^2) = 2 log(\sigma), hence \sigma = exp(0.5*v)
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, cond_shape):
        batch, device = len(cond_shape[0]), self.betas.device
        channels = cond_shape[1]
        x_cond = cond_shape[0]


        sample_size = cond_shape[2]
        img = torch.randn((batch * sample_size, channels), device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc = 'sampling loop time step',
                      total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            x_cond0 = x_cond.repeat_interleave(repeats=sample_size, dim=0)

            img, x_start = self.p_sample(img, x_cond0, t, self_cond)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, cond_x, sample_size = 50):
        batch_size = len(cond_x)
        channels = self.channels
        

        sample_fn = self.p_sample_loop

        return sample_fn((cond_x, channels, sample_size))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)\
            * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None, *args, **kwargs):
        b, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)


        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, args[0], t).pred_x_start
                x_self_cond.detach_()


        model_out = self.model(x, args[0], t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, X, *args, **kwargs):
        n, c, device, = *X.shape, X.device    
        t = torch.randint(0, self.num_timesteps, (n,), device=device).long()

        X = self.normalize(X)
        return self.p_losses(X, t, None, *args, **kwargs)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedLinear(dim, dim_out)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.out = nn.Conv1d(hidden_dim,hidden_dim,1)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c= x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) -> b h c', h = self.heads), qkv)
        sim = einsum('b h i, c h j -> b h c', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('j h i, i h d -> j h d', attn, v)
        out = rearrange(out, 'b h c -> b (h c)', h = self.heads)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        b, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) -> b h c', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h i, c h j -> b h c', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('j h i, i h d -> j h d', attn, v)

        out = rearrange(out, 'b h c -> b (h c)', h = self.heads)
        return self.to_out(out)


class UnetVector(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        aq_dq_dim = 64,
        cond_dim = 9,
        dim_mults=(1, 2, 4, 8),
        channels = 32,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        self.aq_dq_dim = aq_dq_dim
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)   
        self.init_conv = nn.Linear(
                                input_channels + cond_dim + 2 * aq_dq_dim,
                                init_dim
                            )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.cell_pos_emb = SinusoidalPosEmb(self.aq_dq_dim)
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Linear(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Linear(dim_out, dim_in)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Linear(dim, self.out_dim)   #Conv1d(dim, self.out_dim, 1)

    def forward(self, y, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(y))
            x = torch.cat((x_self_cond, x), dim = 1)
        aq = x[:,-2:-1]
        dq = x[:,-1:]
        aq = self.cell_pos_emb(aq).squeeze(1)
        dq = self.cell_pos_emb(dq).squeeze(1)
        x = torch.cat((y, x[:,:-2], aq, dq), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


