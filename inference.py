import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything

from cocktail.utils import resize_image, HWC3
from cocktail.model import create_model
from ldm.models.diffusion.ddim import DDIMSampler



model = create_model('./configs/cocktail_v21.yaml').cpu()
sd = torch.load('./cocktail-laion-512-epoch19.ckpt', map_location='cpu')["state_dict"]
model.load_state_dict(sd)
model.eval()

prompts_set = {
                0: "An astronaut standing on the mountain.",
                1: "A little boy walks down the road, followed by a cat.",
                2: "James Bond and cocktail.",
                3: "James Bond is drinking cocktail."
                }

idx = sys.argv[1] # 1, 5, 7

seed_everything(42)
n_samples = 2
batch_size = n_samples
ddim_steps = 50
strength = 1

from PIL import Image
import numpy as np

input_image_1  = np.array(Image.open(f'./samples/conditions/{idx}_sketch.png'))
input_image_2  = np.array(Image.open(f'./samples/conditions/{idx}_seg.png'))
input_image_3  = np.array(Image.open(f'./samples/conditions/{idx}_keypoints.png'))

def get_input_modality(image_array, size):
        img = resize_image(HWC3(image_array), size)
        modality = torch.from_numpy(img.copy()).float().cuda() / 255.0
        modality = torch.stack([modality for _ in range(n_samples)], dim=0)
        modality = einops.rearrange(modality, 'b h w c -> b c h w').clone()
        return modality

control1 = get_input_modality(input_image_1, 512)
control2 = get_input_modality(input_image_2, 512)
control3 = get_input_modality(input_image_3, 512)
control = [control1, control2, control3]

H, W = control[0].shape[2:]

model.cuda()

# prompt = 'A girl stands on a mountain ground, looking at a sheep'
prompt = prompts_set[int(idx)]
a_prompt = 'best quality, extremely detailed, cyberpunk.'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality.'

c = model.get_learned_conditioning([prompt + ', ' + a_prompt] * n_samples)
cond={"c_concat": control, "c_crossattn": [c]}

uc = model.get_learned_conditioning([n_prompt] * n_samples)
uc_cat = control
uc_cond = {"c_concat": uc_cat, "c_crossattn": [uc]}

shape = (4, H//8, W//8)

model.control_scales = [strength] * 13

from torchvision.utils import save_image
ddim_sampler = DDIMSampler(model)
samples, _ = ddim_sampler.sample(ddim_steps, batch_size,
                         shape, cond, verbose=False, eta=0.0,
                         unconditional_guidance_scale=9.0,
                         unconditional_conditioning=uc_cond)

imgs = model.decode_first_stage(samples)
imgs = torch.clamp(imgs/2+0.5, 0, 1)

for i, img in enumerate(imgs):
        save_image(img, './samples/results/{}_sample_{}.png'.format(idx, i))

control = control1[0] + control2[0] + control3[0]
save_image(control, f"./samples/conditions/{idx}_all.png")

print('Rendering Done!')