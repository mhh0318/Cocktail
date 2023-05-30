from ldm.models.diffusion.ddim import DDIMSampler
import torch
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from typing import Callable, Dict, List, Optional, Tuple,Union
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import math
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel



TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def _img_importance_flatten(img: torch.tensor, w: int, h: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        # scale_factor=1 / ratio,
        size=(w, h),
        mode="bilinear",
        align_corners=True,
    ).squeeze()

def always_round(x):
    intx = int(x)
    is_even = intx%2 == 0
    if is_even:
        if x < intx + 0.5:
            return intx
        return intx + 1
    else:
        return round(x)

def _blur_image_mask(seperated_word_contexts, extra_sigmas):
    for k, sigma in extra_sigmas.items():
        blurrer = T.GaussianBlur(kernel_size=(39, 39), sigma=(sigma, sigma))
        v_as_tokens, img_where_color = seperated_word_contexts[k]
        seperated_word_contexts[k] = (v_as_tokens, blurrer(img_where_color[None,None])[0,0])
    return seperated_word_contexts

def _context_seperator(
    color_context: dict, _tokenizer, w= None, h=None
) -> List[Tuple[List[int], torch.Tensor]]:

    ret_lists = []
    bbxs = {}
    w= w if w is not None else 512
    h= h if h is not None else 512
    default_image = np.zeros((w, h), dtype=np.uint8)
    i = 0
    for v, bbx in color_context.items():
        f = bbx.split(",")[-1]
        bbx_ = [int(_) for _ in bbx.split(",")[:-1]]
        f = float(f)
        v_input = _tokenizer(
            v,
            max_length=_tokenizer.model_max_length,
            truncation=True,
        )
        v_as_tokens = v_input["input_ids"][1:-1]

        default_image[bbx_[0] : bbx_[1], bbx_[2] : bbx_[3]] = 1
        bbxs['BOUNDING_BOX{}'.format(i)] = bbx_


        image_attn = torch.tensor(default_image, dtype=torch.float32) * f

        ret_lists.append((v_as_tokens, image_attn))
        i+=1


    if len(ret_lists) == 0:
        ret_lists.append(([-1], torch.zeros((w, h), dtype=torch.float32)))
    return ret_lists, w, h, bbxs

def _tokens_img_attention_weight(
    img_context_seperated, tokenized_texts, ratio: int = 8, original_shape=False
):
    
    token_lis = tokenized_texts["input_ids"][0].tolist()
    w, h = img_context_seperated[0][1].shape

    w_r, h_r = always_round(w/ratio), always_round(h/ratio)
    ret_tensor = torch.zeros((w_r * h_r, len(token_lis)), dtype=torch.float32)
    
    for v_as_tokens, img_where_color in img_context_seperated:
        is_in = 0
        for idx, tok in enumerate(token_lis):
            if token_lis[idx : idx + len(v_as_tokens)] == v_as_tokens:
                is_in = 1

                # print(token_lis[idx : idx + len(v_as_tokens)], v_as_tokens)
                ret_tensor[:, idx : idx + len(v_as_tokens)] += (
                    _img_importance_flatten(img_where_color, w_r, h_r)
                    .reshape(-1, 1)
                    .repeat(1, len(v_as_tokens))
                )

        if not is_in == 1:
            print(f"Warning ratio {ratio} : tokens {v_as_tokens} not found in text")

    if original_shape:
        ret_tensor = ret_tensor.reshape((w_r, h_r, len(token_lis)))

    return ret_tensor



@torch.autocast("cuda")
def local_forward(self, x, context=None):
    h = self.heads

    q = self.to_q(x)
    if context is not None:
        context = context["CONTEXT_TENSOR"]
    else:
        context = x

    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    # force cast to fp32 to avoid overflowing

    with torch.autocast(enabled=False, device_type = 'cuda'):
        q, k = q.float(), k.float()
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    del q, k

    attention_size_of_img = sim.shape[-2]

    if context is not None:
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
        cross_attention_weight = f(w, sigma, sim)

    attention_scores = (sim + cross_attention_weight) 
    attention_probs = attention_scores.softmax(dim=-1)
    # attention, what we cannot get enough of
    # sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attention_probs, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)

def get_views(panorama_height, panorama_width, window_size=64, stride=16):
    # panorama_height /= 8
    # panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views

def pad_pixels(bg_h_start, bg_h_end, bg_w_start, bg_w_end, fg_view):
    h_start, h_end, w_start, w_end = fg_view
    h_pad_left = h_start - bg_h_start*8 if h_start > bg_h_start*8 else 0
    h_pad_right = bg_h_end*8 - h_end if bg_h_end*8 > h_end else 0
    w_pad_left = w_start - bg_w_start*8 if w_start > bg_w_start*8 else 0
    w_pad_right = bg_w_end*8 - w_end if bg_w_end*8 > w_end else 0
    return (w_pad_left, w_pad_right, h_pad_left, h_pad_right)

def get_intersection(bg_h_start, bg_h_end, bg_w_start, bg_w_end, fg_view):

    fg_h_start, fg_h_end, fg_w_start, fg_w_end = fg_view

    h_start = max(bg_h_start*8, fg_h_start)
    h_end = min(bg_h_end*8, fg_h_end)
    w_start = max(bg_w_start*8, fg_w_start)
    w_end = min(bg_w_end*8, fg_w_end)

    if h_start >= h_end or w_start >= w_end:
        return None
    
    h_offset1 = h_start - bg_h_start*8  # has another slice operation at the begin of the loop
    w_offset1 = w_start - bg_w_start*8  # has another slice operation at the begin of the loop
    h_offset2 = h_start - fg_h_start
    w_offset2 = w_start - fg_w_start

    return {'bg':[h_offset1, h_offset1 + h_end - h_start, w_offset1, w_offset1 + w_end - w_start],
            'fg':[h_offset2, h_offset2 + h_end - h_start, w_offset2, w_offset2 + w_end - w_start]} # [bg, fg] 

def extract_control_and_prompt(input):
    new_input = {}
    prompt = input['prompt']
    i = 0
    while 'object{}'.format(i) in input:
        new_input['OBJECT{}'.format(i)] = input['object{}'.format(i)]
        i +=1
    
    return prompt, new_input

class PSampler(DDIMSampler):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        # for _module in self.model.model.diffusion_model.modules():
        #     if _module.__class__.__name__ == "CrossAttention":
        #         _module.__class__.__call__ = local_forward


    def _encode_text_region_inputs(self,
            text_encoder, device, 
            region_context, # dict
            conditional_input, unconditional_input):
        
        # TODO prompt should include control signals.
        # Process input prompt text

        input_prompt, conditional_control = extract_control_and_prompt(conditional_input)
        unconditional_input_prompt, unconditional_control = extract_control_and_prompt(unconditional_input)

        text_input = TOKENIZER(
            input_prompt,
            padding="max_length",
            max_length=TOKENIZER.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Extract seed and sigma from color context
        # color_context, extra_seeds, extra_sigmas = _extract_seed_and_sigma_from_context(color_context)
        # is_extra_sigma = len(extra_sigmas) > 0
        
        # Process color map image and context
        seperated_word_contexts, width, height, bbxs = _context_seperator(
            region_context, TOKENIZER)
        
        # Smooth mask with extra sigma if applicable
        # if is_extra_sigma:
            # print('Use extra sigma to smooth mask', extra_sigmas)

        # Now hard mask
        # seperated_word_contexts = _blur_image_mask(seperated_word_contexts, sigmas)
        
        # Compute cross-attention weights
        cross_attention_weight_1 = _tokens_img_attention_weight(
            seperated_word_contexts, text_input, ratio=1, original_shape=True
        ).to(device)
        cross_attention_weight_8 = _tokens_img_attention_weight(
            seperated_word_contexts, text_input, ratio=8
        ).to(device)
        cross_attention_weight_16 = _tokens_img_attention_weight(
            seperated_word_contexts, text_input, ratio=16
        ).to(device)
        cross_attention_weight_32 = _tokens_img_attention_weight(
            seperated_word_contexts, text_input, ratio=32
        ).to(device)
        cross_attention_weight_64 = _tokens_img_attention_weight(
            seperated_word_contexts, text_input, ratio=64
        ).to(device)

        # Compute conditional and unconditional embeddings
        cond_embeddings = text_encoder.encode_with_transformer(text_input.input_ids.to(device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = TOKENIZER(
            unconditional_input_prompt,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = text_encoder.encode_with_transformer(uncond_input.input_ids.to(device))[0]
        encoder_hidden_states = {
            "CONTEXT_TENSOR": cond_embeddings.unsqueeze(0),
            f"CROSS_ATTENTION_WEIGHT_ORIG": cross_attention_weight_1,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height/8)*always_round(width/8)}": cross_attention_weight_8,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height/16)*always_round(width/16)}": cross_attention_weight_16,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height/32)*always_round(width/32)}": cross_attention_weight_32,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height/64)*always_round(width/64)}": cross_attention_weight_64,
        }

        encoder_hidden_states.update(conditional_control)
        encoder_hidden_states.update(bbxs)

        uncond_encoder_hidden_states = {
            "CONTEXT_TENSOR": uncond_embeddings.unsqueeze(0),
            f"CROSS_ATTENTION_WEIGHT_ORIG": 0,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height/8)*always_round(width/8)}": 0,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height/16)*always_round(width/16)}": 0,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height/32)*always_round(width/32)}": 0,
            f"CROSS_ATTENTION_WEIGHT_{always_round(height/64)*always_round(width/64)}": 0,
        }

        uncond_encoder_hidden_states.update(unconditional_control)
        uncond_encoder_hidden_states.update(bbxs)

        return seperated_word_contexts, encoder_hidden_states, uncond_encoder_hidden_states

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, region_context,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, wf = lambda w, sigma, qk: 1.5 * w * math.log(sigma**2 + 1) * qk.std()):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)


        seperated_word_contexts, encoder_hidden_states, uncond_encoder_hidden_states  = \
            self._encode_text_region_inputs(self.model.cond_stage_model, device, 
            region_context, # dict
            cond, unconditional_conditioning)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      dynamic_threshold=dynamic_threshold,
                                      encoder_hidden_states=encoder_hidden_states,
                                      uncond_encoder_hidden_states=uncond_encoder_hidden_states,
                                      weight_function=wf)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    @torch.no_grad()
    def p_sample_ddim(self, x, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., 
                      dynamic_threshold=None,
                      encoder_hidden_states=None,
                      uncond_encoder_hidden_states=None,
                      weight_function= None):
        b, *_, device = *x.shape, x.device

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
    
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)


        x_in = x
        t_in = t

        encoder_hidden_states.update({
            "SIGMA": a_t, #sigma_t,
            "WEIGHT_FUNCTION": weight_function,
        })
        uncond_encoder_hidden_states.update({
            "SIGMA": a_t, #sigma_t,
            "WEIGHT_FUNCTION": lambda w, sigma, qk: 0.0,
        })
        model_t = self.model.apply_model(x_in, t_in, encoder_hidden_states)
        model_uncond = self.model.apply_model(x_in, t_in, uncond_encoder_hidden_states)
        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output


        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0