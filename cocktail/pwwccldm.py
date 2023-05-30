import torch
import torch.nn.functional as F
from cocktail.ccldm import ControlLDM as CCNet


class ControlLDM(CCNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_controls = len(self.control_keys)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        num_samples = x_noisy.shape[0]
        cond_txt = cond['CONTEXT_TENSOR']

        i = 0
        cond_hint = [list() for _ in range(self.num_controls)]
        while 'OBJECT{}'.format(i) in cond:
            cond_hint_tmp = cond['OBJECT{}'.format(i)]
            bbx_hint_i = cond['BOUNDING_BOX{}'.format(i)]
            for control_i in range(self.num_controls):
                cond_hint_i =  torch.zeros(num_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
                cond_hint_i[..., bbx_hint_i[0]:bbx_hint_i[1], bbx_hint_i[2]:bbx_hint_i[3]] = cond_hint_tmp[control_i]
                cond_hint[control_i].append(cond_hint_i)
            i += 1

        cond_hint = [torch.cat(cond_hint[i], 0) for i in range(self.num_controls)]
        x_batch_noisy = x_noisy.repeat(len(cond_hint), 1, 1, 1)
        t_batch = t.repeat(len(cond_hint))
        cond_txt_batch = cond_txt.repeat(len(cond_hint), 1,1)

        control = self.control_model(x=x_batch_noisy, hint=cond_hint, timesteps=t_batch, context=cond_txt_batch)
        # control = [c * scale for c, scale in zip(control, self.control_scales)]
        fuse_control = [torch.sum(i[torch.arange(0, t_batch.size(0))],0,keepdim=True) for i in control]

        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=fuse_control, only_mid_control=self.only_mid_control)

        return eps
