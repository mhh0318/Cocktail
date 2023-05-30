import random
import json
from pathlib import Path
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset


LAION_Aesthetics_PATH_STRUCTURE = {
    'train': {
        'top_level': '',
        'data': 'train/data',
        'sketch': 'train/annotations/hed',
        'segmentation': 'train/annotations/san/color_segmentation',
        'gray_seg': 'train/annotations/san/mask'
    },
    'val': {
        'top_level': '',
        'data': 'val/data',
        'sketch': 'val/annotations/hed',
        'segmentation': 'val/annotations/san/color_segmentation',
        'gray_seg': 'val/annotations/san/mask'
    }
}


class ComposedAnnotations(Dataset):
    def __init__(self, data_root, json_path, split, size=None, random_crop=False, 
                 interpolation="bicubic", composed_type="latent_composed"):
        self.data_root=data_root
        self.split=split
        self.size = size
        self.random_crop = random_crop
        self.composed_type = composed_type

        self.paths = self.build_paths(data_root)
        with open(json_path, 'r') as f:
            self.image_paths = json.load(f)
        self._length = len(self.image_paths)

        if self.size is not None and self.size > 0:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size, 
                                                           interpolation=self.interpolation)

    def get_path_structure(self):
        if self.split not in LAION_Aesthetics_PATH_STRUCTURE:
            raise ValueError(f'Split [{self.split} does not exist for LAION_Aesthetics data.]')
        return LAION_Aesthetics_PATH_STRUCTURE[self.split]

    def build_paths(self, top_level):
        top_level = Path(top_level)
        sub_paths = {name: top_level.joinpath(sub_path) for name, sub_path in self.get_path_structure().items()}
        for path in sub_paths.values():
            if not path.exists():
                raise FileNotFoundError(f'{type(self).__name__} data structure error: [{path}] does not exist.')
        return sub_paths

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.rescaler(image=image)["image"]
        return image

    def preprocess_gray_seg(self, image_path):
        segmentation = Image.open(image_path)
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.size is not None:
            segmentation = self.rescaler(image=segmentation)["image"]
        return segmentation 

    def paired_cropper(self, image, segmentation, sketch, gray_seg):
        width, height=image.shape[0],image.shape[1]

        if not self.random_crop:
            x=(width-self.size)//2
            y=(height-self.size)//2
        else:
            x=random.randint(0, width-self.size)
            y=random.randint(0, height-self.size)
        
        image=image[x:x+self.size, y:y+self.size:, :]
        segmentation=segmentation[x:x+self.size, y:y+self.size, :]
        sketch=sketch[x:x+self.size, y:y+self.size, :]
        gray_seg=gray_seg[x:x+self.size, y:y+self.size]
        
        return {"image": image, "segmentation": segmentation, "sketch": sketch, "gray_seg": gray_seg}
    
    def get_incompleted_controls(self, sketch, seg, gray_seg):
        all_labels = np.unique(gray_seg)
        random_labels = np.random.choice(all_labels, np.random.randint(1, len(all_labels)+1), replace=False)
        mask = np.isin(gray_seg, random_labels)

        p = random.random()
        if p < 0.7:
            if p < 0.1 and len(random_labels)!=len(all_labels):
                sketch[mask] = 0
                seg[mask] = 0
            elif 0.1 <= p < 0.4:
                sketch[mask] = 0
            else:
                seg[mask] = 0
        return sketch, seg

    def __getitem__(self, i):
        sample = dict()

        relative_file_path = self.image_paths[i]
        image = self.preprocess_image(self.paths["data"].joinpath(relative_file_path))
        sketch= self.preprocess_image(self.paths["sketch"].joinpath(relative_file_path))
        segmentation =self.preprocess_image(self.paths["segmentation"].joinpath(relative_file_path))
        gray_segmentation =self.preprocess_gray_seg(self.paths["gray_seg"].joinpath(relative_file_path))

        if self.size is not None:
            cropper = self.paired_cropper(image=image,
                                          segmentation=segmentation,
                                          sketch=sketch,
                                          gray_seg=gray_segmentation)
        else:
            cropper = {"image": image,
                       "segmentation": segmentation,
                       "sketch": sketch,
                       "gray_segmentation": gray_segmentation}

        sample["image"] = (cropper["image"]/127.5 - 1.0).astype(np.float32)

        if self.composed_type == "latent_composed":
            sketch, segmentation = self.get_incompleted_controls(cropper["sketch"], cropper["segmentation"], cropper["gray_seg"])
            sample["sketch"] = (sketch/255.0).astype(np.float32)
            sample["segmentation"] = (segmentation/255.0).astype(np.float32)
        
        sample["caption"] = self.paths["data"].joinpath(relative_file_path).with_suffix('.txt').read_text().strip()

        return sample

if __name__ == "__main__":
    dataset = ComposedAnnotations(data_root="data/LAION-Aesthetics-6.5", json_path="data/LAION-Aesthetics-6.5/train.json", split="train", size=512)
    print(len(dataset))