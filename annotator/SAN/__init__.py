import os
import numpy as np
from PIL import Image
from .predict import Predictor

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')
san_ckpt_path = os.path.join(annotator_ckpts_path, "san_vit_large_14.pth")

if not os.path.exists(san_ckpt_path):
    from basicsr.utils.download_util import load_file_from_url
    load_file_from_url("https://huggingface.co/Mendel192/san/resolve/main/san_vit_large_14.pth", model_dir=annotator_ckpts_path)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i + 1  # let's give 0 a color
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] =  r
        cmap[i, 1] =  g
        cmap[i, 2] =  b
     
    return cmap


class Colorize(object):
    def __init__(self, n=182):
        self.cmap = labelcolormap(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1])) 
     
        for label in range(0, len(self.cmap)):
            mask = (label == gray_image ) 
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        
        color_image = np.transpose(color_image , (1,2,0)).astype(np.uint8)

        return color_image


class Segmenter(object):
    def __init__(self, config_file="annotator/SAN/configs/san_clip_vit_large_res4_coco.yaml", model_path=san_ckpt_path, n_labels=171):
        self.predictor = Predictor(config_file=config_file, model_path=model_path)
        self.colorizer = Colorize(n_labels)
    
    def __call__(self, img, seg_path=None, type='filepath'):
        try:
            if type == 'np':
                img = Image.fromarray(img)
            result = self.predictor.predict(img)
            gray_seg = result["sem_seg"].astype(np.uint8)
            if type == 'filepath':
                self.gary2rgb(gray_seg, seg_path, type)
            else:
                return self.gary2rgb(gray_seg, seg_path, type)
        except:
            print(f'{img} is wrong.')
        
    def gary2rgb(self, gray_seg, output_file, type):
        color_seg = self.colorizer(gray_seg)
        if type == 'filepath':
            Image.fromarray(color_seg).save(output_file)
        else:
            return color_seg
