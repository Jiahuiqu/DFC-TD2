from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
import math, os, copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from prettytable import PrettyTable
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
from HSIspe import HSIspe
from Lispe import Lispe
from dataset import load_data
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

class LatentEncoder(nn.Module):
    def __init__(self, dim, latent_dim=64):
        super(LatentEncoder, self).__init__()
        self.hsi = HSIspe()
        self.lispe = Lispe(hidden_size=64)
        self.H, self.W = 11, 11
        self.linear_out = nn.Sequential(
            nn.Linear(dim, 121),
            nn.BatchNorm1d(121),
            nn.ReLU()
        )
    def forward(self, x1, x2):
        hsi1 = self.hsi(x1)
        lispe1 = self.lispe(x2)
        return lispe1

class SharedEncoder(nn.Module):
    def __init__(self, n_moments=5):
        super(SharedEncoder, self).__init__()
        self.hsi = HSIspe()
        self.lispe = Lispe(hidden_size=64)
        self.cmd = CMD()
        self.n_moments = n_moments
        self.fc_restore = nn.Conv2d(256, 128, kernel_size=1)

    def forward(self, x1, x2):
        feat1 = self.hsi(x1)
        feat2 = self.lispe(x2)
        cmd_loss = self.cmd(feat1, feat2, self.n_moments)
        shared_features1 = torch.cat((feat1, feat2), dim=1)
        shared_features2 = self.fc_restore(shared_features1)
        return shared_features2, cmd_loss

class CMD(nn.Module):

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

def calculate_sam(target_data, reference_data):
    b, c, h, w = target_data.shape
    target_data = target_data.reshape(b, c, h * w).permute(0, 2, 1)
    reference_data = reference_data.reshape(b, c, h * w).permute(0, 2, 1)
    target_data_norm = torch.nn.functional.normalize(target_data, dim=2)
    reference_data_norm = torch.nn.functional.normalize(reference_data, dim=2)

    dot_product = torch.einsum('bnc,bnc->bn', target_data_norm, reference_data_norm)

    length_product = torch.norm(target_data_norm, dim=2) * torch.norm(reference_data_norm, dim=2)

    sam = torch.acos(dot_product / length_product)
    sam_mean = torch.mean(torch.mean(sam, dim=1))
    return sam_mean


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(SelfAtt, self).__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)
        self.att = att_num

    def forward(self, x):
        x_org = x
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return x_org + self.att * self.proj(out)


class Cross_Att(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(Cross_Att, self).__init__()
        self.att = att_num
        self.groupnorm_1 = nn.GroupNorm(norm_groups, channel_dim)
        self.groupnorm_2 = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv_1 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.qkv_2 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)

        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

        self.downsample = nn.Sequential(nn.Conv2d(channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 2, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),
                                        )

        self.upsample = nn.Sequential(nn.Conv2d(2 * channel_dim, 1 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      )

    def forward(self, x, y, mode):
        b, c, h, w = x.size()
        x_org = x
        if mode == 'spe':
            b, c, h, w = x.size()
            x = self.groupnorm_1(x)
            y = self.groupnorm_1(y)
            qkv_1 = rearrange(self.qkv_1(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads,
                              qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(self.qkv_2(y), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads,
                              qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]
            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)
        else:
            x = self.groupnorm_2(x)
            y = self.groupnorm_2(y)
            if h == 512:
                times = h / 64
            else:
                times = h / 20
            n = np.log(times) / np.log(2)
            for i in range(int(n)):
                x = self.downsample[2 * i](x)
                x = self.downsample[2 * i + 1](x)

            for i in range(int(n)):
                y = self.downsample[2 * i](y)
                y = self.downsample[2 * i + 1](y)

            b, c, h, w = x.size()

            x = x.reshape(b, c, h * w).repeat(1, 1, 3)
            y = y.reshape(b, c, h * w).repeat(1, 1, 3)
            qkv_1 = rearrange(x, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(y, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]

            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads (h w) c -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

            for i in range(int(n)):
                l = int(n) - 1 - i
                out = self.upsample[2 * l](out)
                out = self.upsample[2 * l + 1](out)

        return x_org + self.att * self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=False):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        return x


class ResBlock_skip(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        y = self.block1(x)

        return y + self.res_conv(x)


class SGDC(nn.Module):
    def __init__(self, in_channel=37, out_channel=34, inner_channel=64, norm_groups=32,
                 channel_mults=[1, 2, 4, 8, 8], res_blocks=3, dropout=0, img_size=128):
        super().__init__()

        self_att = []
        cros_att = []
        dim_out = [inner_channel, inner_channel * 2, inner_channel * 2]
        for i in reversed(range(len(dim_out))):
            self_att.append(SelfAtt(dim_out[i], num_heads=1, norm_groups=norm_groups))

        self.self_att = nn.ModuleList(self_att)

        for j in reversed(range(len(dim_out))):
            cros_att.append(Cross_Att(dim_out[j], num_heads=1, norm_groups=norm_groups))

        self.cros_att = nn.ModuleList(cros_att)

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]

        now_res = img_size

        # Downsampling stage of SGDC
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Upsampling stage of SGDC
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]

            for i in range(0, res_blocks + 1):
                ups.append(ResBlock(
                    pre_channel + feat_channels.pop(), channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, noise_level, mode=None):
        t = self.noise_level_mlp(noise_level)
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)
        z = 0

        for i, layer in enumerate(self.ups):
            if isinstance(layer, ResBlock):
                if i == 0:
                    x = layer(torch.cat([x, feats.pop()], dim=1), t)
                elif isinstance(self.ups[i - 1], Upsample):
                    temp_feats = feats.pop()
                    if (x.size()[2:] != temp_feats.size()[2:]):
                        x = F.interpolate(x, size=temp_feats.size()[2:], mode='bilinear')
                    x = layer(torch.cat([x, temp_feats], dim=1), t)
                    z = z + 1
                else:
                    temp_feats = feats.pop()
                    if (x.size()[2:] != temp_feats.size()[2:]):
                        x = F.interpolate(x, size=temp_feats.size()[2:], mode='bilinear')
                    x = layer(torch.cat([x, temp_feats], dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)

class Diffusion(nn.Module):
    def __init__(self, model_spe, device, img_size, channels=3):
        super().__init__()
        self.channels = channels
        self.model_spe = model_spe.to(device)
        self.img_size = img_size
        self.device = device
        # complementary fusion block
        self.fuse = nn.Sequential(
            nn.Conv2d(31 * 2, 31 * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(31 * 2, 31, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(31, 31, kernel_size=3, stride=1, padding=1),
        ).to(device)

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal,
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior

    def p_mean_variance(self, x, t, condition_x=None):
        batch_size, c = x.shape[0], condition_x.shape[1]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        x_start = self.model_spe(torch.cat([condition_x, x], dim=1), noise_level=noise_level)
        posterior_mean = (
                self.posterior_mean_coef1[t] * x_start.clamp(-1, 1) +
                self.posterior_mean_coef2[t] * x
        )

        posterior_variance = self.posterior_log_variance_clipped[t]

        mean, posterior_log_variance = posterior_mean, posterior_variance
        return mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, img_spe, t, condition=None):

        mean1, log_variance1 = self.p_mean_variance(x=img_spe, t=t, condition_x=condition)
        noise = torch.randn_like(img_spe) if t > 0 else torch.zeros_like(img_spe)

        return mean1 + noise * (0.5 * log_variance1).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def reconstructed(self, lidar, condition):
        img = torch.rand_like(lidar, device=lidar.device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition=condition)
        return img

    def net(self, lidar_features, shared_features):

        b, c, h, w = lidar_features.shape   #(B,128,11,11)
        t = torch.randint(1, schedule_opt['n_timestep'], size=(b,))
        sqrt_alpha_cumprod_t = extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t, lidar_features.shape)
        sqrt_alpha = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1).type(torch.float32).to(lidar_features.device)
        noise = torch.randn_like(lidar_features).to(lidar_features.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * lidar_features + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The bilateral model predict actual x0 added at time step t
        pred_x0 = self.model_spe(torch.cat([shared_features, x_noisy], dim=1), noise_level=sqrt_alpha, mode='spe')

        loss_1 = self.loss_func(lidar_features, pred_x0) / int(b * c * h * w)

        return loss_1

    def forward(self, lidar_features, shared_features):
        return self.net(lidar_features, shared_features)


class rec(nn.Module):
    def __init__(self, device, img_size, loss_type, dataloader, testloader,
                 schedule_opt, save_path, load_path=None, load=True,
                 in_channel=62, out_channel=31, inner_channel=64, norm_groups=8,
                 channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-3, distributed=False):
        super(rec, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.shared = SharedEncoder()
        self.shared = self.shared.to(device)
        self.spe =LatentEncoder(latent_dim=64,dim=64)
        self.spe = self.spe.to(device)
        self.fc_restore = nn.Linear(256, 1 * 11 * 11) .to(device)
        self.restore_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=5),
        )

        self.conv = nn.Conv2d(1, 256, kernel_size=11) .to(device)

        # Bilateral Spe-Spa Guided Pyramid Denoing model
        model_spe = SGDC(256, 128, inner_channel, norm_groups, channel_mults, res_blocks, dropout, img_size)
        self.rec = Diffusion(model_spe, device, img_size, out_channel)
        # Apply weight initialization & set loss & set noise schedule
        self.rec.apply(self.weights_init_orthogonal)
        self.rec.set_loss(loss_type)
        self.rec.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.rec = nn.DataParallel(self.rec)

        self.optimizer = torch.optim.Adam(self.rec.parameters(), lr=lr)

        params = sum(p.numel() for p in self.rec.parameters())
        print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train_model(self, epoch, verbose):
        train = True
        for i in range(epoch):
            i = i
            train_loss = 0
            self.rec.train()
            loss_1_epoch = 0
            if train:
                for batch_idx, ((x1, x2), labels) in enumerate(train_dataloader):
                    x1, x2 = x1.to(device), x2.to(device)
                    self.optimizer.zero_grad()
                    x,cmd_loss=self.shared(x1,x2)
                    b, c, h, w = x.shape
                    x2_features=self.spe(x1,x2)
                    loss_1 = self.rec(x2_features, x)
                    loss_1 = loss_1.sum()
                    loss = loss_1 + 0.001 * cmd_loss
                    loss.backward()
                    self.optimizer.step()
                    loss_1_epoch += loss_1.item()
                    train_loss += loss.item()
                print('epoch: {}'.format(i))
                print('损失函数:')
                x = PrettyTable()
                x.add_column("loss", ['value'])
                x.add_column("loss_all", [train_loss / float(len(self.dataloader))])
                x.add_column("loss_1", [loss_1_epoch / float(len(self.dataloader))])
                x.add_column("cmd_loss", [cmd_loss.item() / float(len(self.dataloader))])
                print(x)

            if (i + 1) % verbose == 0:
                self.rec.eval()
                test_data = copy.deepcopy(next(iter(self.testloader)))
                [(x1, x2), labels] = test_data
                x1, x2 = x1.to(device), x2.to(device)
                randn3 = np.random.randint(0, b)
                x2_features=self.spe(x1,x2)
                x,_= self.shared(x1, x2) #(B,128,11,11)
                result = self.test_model(x2_features, x)  #(B,128,11,11)
                result = result[randn3]
                x2_r= x2_features[randn3]


    def test_model(self, x, condition):
        self.rec.eval()
        with torch.no_grad():
            if isinstance(self.rec, nn.DataParallel):
                result = self.rec.module.reconstructed(x, condition)
            else:
                result = self.rec.reconstructed(x, condition)
        self.rec.train()
        return result

    def save(self, save_path, i):
        network = self.rec
        if isinstance(self.rec, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path + 'rec_model_epoch-{}.pt'.format(i))

    def load(self, load_path):
        network = self.rec
        if isinstance(self.rec, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")

    def encode_test_result(self, x2_restored, condition):
        with torch.no_grad():
            result = self.rec.reconstructed(x2_restored, condition)
        return result


