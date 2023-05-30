import torch
import torch.nn.functional as F
from cocktail.ccldm import ControlLDM as CCNet
from cocktail.mcldm import gaussian_filter


class ControlLDM(CCNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_controls = len(self.control_keys)

    @torch.no_grad()
    def get_black_background(self, num_samples):
        # sample random background with a constant rgb value
        backgrounds = torch.zeros(num_samples, 3, device=self.device)[:, :, None, None].repeat(1, 1, 512, 512)
        return [backgrounds for _ in range(self.num_controls)]

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
                    guassian_weight = gaussian_filter(mask_null_i, sigma=sigma)
                    con_temp = con[i][j] * guassian_weight + nul[i][nul_idx] * (1 - guassian_weight)
                    unc_temp = unc[i][j] * guassian_weight + nul[i][nul_idx] * (1 - guassian_weight)
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

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt_bg = torch.cat(cond['c_crossattn'], 1)

        i = 0
        cond_hint = [list() for _ in range(self.num_controls)]
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
                cond_null = self.get_black_background(1)
                text_null = cond['caption_{}'.format(i)][0][:1] # c and uc cat, we use c for black bg
                for control_i in range(self.num_controls):
                    cond_hint_i[control_i] = F.pad(cond_hint_i[control_i], padding_para_i, mode='constant', value=0)
                    # assert cond_hint_i[control_i].max() >= 1
                    cond_hint_i[control_i] = torch.cat([cond_hint_i[control_i], cond_null[control_i]], 0)
                    cond_hint[control_i].append(cond_hint_i[control_i])

                cond_txt_i = torch.cat(cond['caption_{}'.format(i)], 1)
                cond_txt_i = torch.cat([cond_txt_i, text_null], 0)
                i+=1
                # cond_hint.append(cond_hint_i)
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
                for control_i in range(self.num_controls):
                    cond_hint[control_i].append(cond_hint_i[control_i])
                cond_text.append(cond_txt_i)
                x_noisy_batch_cond.append(x_noisy)
                t_batch_cond.append(t)
                intersections.append(intersection_i)
                cuc_flags += [0,1]
                cls_flags += [1]

        cuc_flags = torch.tensor(cuc_flags)

        cond_hint = [torch.cat(cond_hint[i], 0) for i in range(self.num_controls)]
        cond_text = torch.cat(cond_text, 0)

        x_noisy_batch_cond = torch.cat(x_noisy_batch_cond, 0)
        t_batch_cond = torch.cat(t_batch_cond, 0)
        

        control = self.control_model(x=x_noisy_batch_cond, hint=cond_hint, timesteps=t_batch_cond, context=cond_text)
        print('control', control[0].shape)

        con_control = [i[cuc_flags==0] for i in control]
        unc_control = [i[cuc_flags==1] for i in control]
        nul_control = [i[cuc_flags==2] for i in control]

        con_control, unc_control = self.fuse_null_control(con_control, unc_control, nul_control, intersections)
        control = [torch.cat([con_control[i],unc_control[i]],0) for i in range(len(con_control))]
        
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt_bg, control=control, only_mid_control=self.only_mid_control)

        return eps
