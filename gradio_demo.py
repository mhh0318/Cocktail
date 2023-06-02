import sys
from PIL import Image
import numpy as np
from pathlib import Path
import gradio as gr
import torch
import cv2
import einops
import random
from pytorch_lightning import seed_everything

from annotator.hed import HEDdetect
from annotator.openpose import OpenposeDetector
from annotator.SAN import Segmenter
from annotator.util import HWC3, resize_image
from cocktail.model import create_model
from ldm.models.diffusion.ddim import DDIMSampler


apply_hed = HEDdetect()
apply_san = Segmenter()
apply_openpose = OpenposeDetector()

model = create_model('./configs/cocktail_v21.yaml').cpu()
sd = torch.load('./cocktail-laion-512-epoch19.ckpt', map_location='cpu')["state_dict"]
model.load_state_dict(sd)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(hed_image, seg_image, openpose_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed):
    if hed_image is not None:
        conditon_image = hed_image
    elif seg_image is not None:
        conditon_image = seg_image
    elif openpose_image is not None:
        conditon_image = openpose_image
    else:
        conditon_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
    H, W, C = resize_image(HWC3(conditon_image), image_resolution).shape

    with torch.no_grad():
        if hed_image is not None:
            hed_image = cv2.resize(hed_image, (W, H))
            hed_map = HWC3(apply_hed(HWC3(hed_image)))
        else:
            hed_map = np.zeros((H, W, C)).astype(np.uint8)
        if seg_image is not None:
            seg_image = cv2.resize(seg_image, (W, H))
            seg_map = HWC3(apply_san(HWC3(seg_image), type='np'))
        else:
            seg_map = np.zeros((H, W, C)).astype(np.uint8)
        if openpose_image is not None:
            openpose_image = cv2.resize(openpose_image, (W, H))
            openpose_map = HWC3(apply_openpose(HWC3(openpose_image), hand_and_face=True))
        else:
            openpose_map = np.zeros((H, W, C)).astype(np.uint8)

        detected_map_list = [hed_map, seg_map, openpose_map]
        controls = []
        for detected_map in detected_map_list:
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            controls.append(control)

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": controls, "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": controls, "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                         shape, cond, verbose=False, eta=0.0,
                         unconditional_guidance_scale=scale,
                         unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    return [detected_map_list, results]

block = gr.Blocks().queue()
with block:
    gr.Markdown("# Cocktail")
    with gr.Row():
        hed_image = gr.Image(source='upload', type="numpy", label='HED')
        seg_image = gr.Image(source='upload', type="numpy", label='Segmentation')
        openpose_image = gr.Image(source='upload', type="numpy", label='OpenPose')
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
    with gr.Row():
        with gr.Accordion("Advanced options", open=False):
            a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
            n_prompt = gr.Textbox(label="Negative Prompt",
                                  value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
            scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    with gr.Row():
        run_button = gr.Button(label="Run")
    with gr.Row():
        cond_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=3, height='auto')
    with gr.Column():
        result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')

    ips = [hed_image, seg_image, openpose_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed]
    run_button.click(fn=process, inputs=ips, outputs=[cond_gallery, result_gallery])


share = False
if "--share" in sys.argv:
    share = True
block.launch(server_name='0.0.0.0', share=share)
