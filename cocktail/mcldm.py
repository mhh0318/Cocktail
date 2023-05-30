import einops
from typing import Callable
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
import math
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

@torch.autocast("cuda")
def inj_forward(self, hidden_states, context=None, mask=None):

    is_dict_format = True
    if context is not None:
        try:
            context_tensor = context["CONTEXT_TENSOR"]
        except:
            context_tensor = context
            is_dict_format = False

    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)

    key = self.to_k(context_tensor)
    value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))

    attention_size_of_img = attention_scores.shape[-2]
    if context is not None:
        if is_dict_format:
            f: Callable = context["WEIGHT_FUNCTION"]
            try:
                w = context[f"CROSS_ATTENTION_WEIGHT_{attention_size_of_img}"]
            except KeyError:
                w = context[f"CROSS_ATTENTION_WEIGHT_ORIG"]
                if not isinstance(w, int):
                    img_h, img_w, nc = w.shape
                    ratio = math.sqrt(img_h * img_w / attention_size_of_img)
                    w = F.interpolate(w.permute(2, 0, 1).unsqueeze(0), scale_factor=1/ratio, mode="bilinear", align_corners=True)
                    w = F.interpolate(w.reshape(1, nc, -1), size=(attention_size_of_img,), mode='nearest').permute(2, 1, 0).squeeze()
                else:
                    w = 0
            sigma = context["SIGMA"]

            cross_attention_weight = f(w, sigma, attention_scores)
        else:
            cross_attention_weight = 0.0
    else:
        cross_attention_weight = 0.0

    attention_scores = (attention_scores + cross_attention_weight) * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)

    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states


def gaussian_filter(image, sigma):
    #TODO Can be optimized by channel numbers
    if sigma == 0:
        return image
    if image.max() ==0:
        return image
    if len(image.size()) == 3:
        image = image.unsqueeze(0)
    # image = image.unsqueeze(0)
    k = int(4 * sigma + 0.5)
    kernel_size = 2 * k + 1
    x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    x = x - k
    y = y - k
    kernel = torch.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(image.size(0), image.size(1), 1, 1).to(image.device)
    filtered_image = torch.nn.functional.conv2d(image, kernel, padding=k)
    return filtered_image[0][0] / filtered_image.max()


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
                    conv_nd(dims, hint_channels, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 16, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 32, 32, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                    nn.SiLU(),
                    conv_nd(dims, 96, 96, 3, padding=1),
                    nn.SiLU(),
                    conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control

    @torch.no_grad()
    def get_black_background(self, num_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.zeros(num_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
        return backgrounds

    def fuse_null_control(self, con, unc, nul, intersections):
        objs = len(intersections)
        cons = []
        uncs = []
        nul_idx = 0
        for j in range(objs):
            if intersections[j] != 'null':
                intersection = [int(_/8) for _ in intersections[j]['bg']]
                cs, us = [], []
                for i in range(len(nul)):
                # for i in range(13):
                    mask_null_i = torch.zeros_like(nul[i][nul_idx])
                    if mask_null_i.shape[-1] == 64:
                        sigma = 4
                        intersection_tmp = intersection
                    elif mask_null_i.shape[-1] == 32:
                        sigma = 2
                        intersection_tmp = [int(_/2) for _ in intersection]
                    elif mask_null_i.shape[-1] == 16:
                        sigma = 1
                        intersection_tmp = [int(_/4) for _ in intersection]
                    elif mask_null_i.shape[-1] == 8:
                        sigma = 0
                        intersection_tmp = [int(_/8) for _ in intersection]
                    mask_null_i[:,intersection_tmp[0]:intersection_tmp[1], intersection_tmp[2]: intersection[3]] = 1
                    guassian_weight = gaussian_filter(mask_null_i, sigma=0)
                    # con_temp = con[i][j] * guassian_weight + nul[i][nul_idx] * (1 - guassian_weight) # 0 input masks
                    # unc_temp = unc[i][j] * guassian_weight + nul[i][nul_idx] * (1 - guassian_weight)
                    con_temp = con[i][j] * guassian_weight + mask_null_i[i][nul_idx] * (1 - guassian_weight) # 0 control signal
                    unc_temp = unc[i][j] * guassian_weight + mask_null_i[i][nul_idx] * (1 - guassian_weight)
                    cs.append(con_temp)
                    us.append(unc_temp)
                nul_idx += 1
                cons.append(cs)
                uncs.append(us)
            else:
                cs, us = [], [] 
                for i in range(len(con)):
                    cs.append(con[i][j])
                    us.append(unc[i][j])
                cons.append(cs)
                uncs.append(us)
        cons = [torch.stack([cons[0][i].squeeze(), cons[1][i].squeeze()]) for i in range(13)]
        uncs = [torch.stack([uncs[0][i].squeeze(), uncs[1][i].squeeze()]) for i in range(13)]
        con_fuse = [torch.sum(k, 0, keepdim=True) for k in cons]
        unc_fuse = [torch.sum(k, 0, keepdim=True) for k in uncs]
        return con_fuse, unc_fuse

    def fuse_null_eps(self, eps, eps_null, intersections):
        mask_null_i = torch.zeros_like(eps_null)
        for i in intersections:
            if i != 'null':
                intersection = [int(_/8) for _ in i['bg']]
                mask_null_i[:,:,intersection[0]:intersection[1], intersection[2]: intersection[3]] = 1
        # guassian_weight = gaussian_filter(mask_null_i, sigma=1)
        eps_temp = eps * mask_null_i + eps_null * (1 - mask_null_i)
        return eps_temp

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt_bg = torch.cat(cond['c_crossattn'], 1)

        i = 0
        cond_hint = []
        cond_text = []
        x_noisy_batch_cond = []
        t_batch_cond = []
        intersections = []
        cuc_flags = []
        cls_flags = []
        while 'object_{}'.format(i) in cond:
        # while cond['object_{}'.format(i)] is not None:
            if cond['area_{}'.format(i)]['intersections'] is not None:
                cond_hint_i = cond['object_{}'.format(i)]
                padding_para_i = cond['area_{}'.format(i)]['padding']
                intersection_i = cond['area_{}'.format(i)]['intersections']
                cond_hint_i = cond_hint_i[...,intersection_i['fg'][0]:intersection_i['fg'][1], intersection_i['fg'][2]:intersection_i['fg'][3]]
                cond_null = self.get_black_background(1)
                text_null = cond['caption_{}'.format(i)][0][:1] # c and uc cat, we use c for black bg
                cond_hint_i = F.pad(cond_hint_i, padding_para_i, mode='constant', value=0)
                # assert cond_hint_i.max() >= 1
                cond_txt_i = torch.cat(cond['caption_{}'.format(i)], 1)
                cond_hint_i = torch.cat([cond_hint_i, cond_null], 0)
                cond_txt_i = torch.cat([cond_txt_i, text_null], 0)
                i+=1
                cond_hint.append(cond_hint_i)
                cond_text.append(cond_txt_i)
                x_noisy_batch_cond.append(x_noisy[0].repeat(3,1,1,1))
                t_batch_cond.append(t[0].repeat(3))
                intersections.append(intersection_i)
                cuc_flags += [0,1,2]
                cls_flags += [0]
            else:
                cond_hint_i = self.get_black_background(2) # cond and unconditional
                padding_para_i = 'null'
                intersection_i = 'null'
                cond_txt_i = torch.cat(cond['caption_{}'.format(i)], 1)
                i+=1
                cond_hint.append(cond_hint_i)
                cond_text.append(cond_txt_i)
                x_noisy_batch_cond.append(x_noisy)
                t_batch_cond.append(t)
                intersections.append(intersection_i)
                cuc_flags += [0,1]
                cls_flags += [1]

        cuc_flags = torch.tensor(cuc_flags)

        cond_hint = torch.cat(cond_hint, 0)
        cond_text = torch.cat(cond_text, 0)

        x_noisy_batch_cond = torch.cat(x_noisy_batch_cond, 0)
        t_batch_cond = torch.cat(t_batch_cond, 0)
        

        control = self.control_model(x=x_noisy_batch_cond, hint=cond_hint, timesteps=t_batch_cond, context=cond_text)
        print('control', control[0].shape)

        # con_control = [torch.sum(i[cuc_flags==0],0,keepdim=True) for i in control]
        # unc_control = [torch.sum(i[cuc_flags==1],0,keepdim=True) for i in control]
        # nul_control = [torch.sum(i[cuc_flags==2],0,keepdim=True) for i in control]
        con_control = [i[cuc_flags==0] for i in control]
        unc_control = [i[cuc_flags==1] for i in control]
        nul_control = [i[cuc_flags==2] for i in control]

        con_control, unc_control = self.fuse_null_control(con_control, unc_control, nul_control, intersections)
        control = [torch.cat([con_control[i],unc_control[i]],0) for i in range(len(con_control))]
        
        # control_null = self.control_model(x=x_noisy, hint=self.get_black_background(2), timesteps= t, context=cond_txt_bg)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt_bg, control=control, only_mid_control=self.only_mid_control)
        # eps_null = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt_bg, control=control_null, only_mid_control=False)

        # eps = self.fuse_null_eps(eps, eps_null, intersections)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    def plugin_cross_attention(self):
        for _module in self.model.diffusion_model.modules():
            if _module.__class__.__name__ == "CrossAttention":
                _module.__class__.__call__ = inj_forward

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
