import os, json, random
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict


class AnnotationsWithCaptionSketch(Dataset):
    def __init__(self, data_root, json_path, index_path=None,
                 size=None, random_crop=False, 
                 interpolation="bicubic", use_simplified_sketch=False):
        self.data_root=data_root
        self.json_path=json_path
        self.index_path=index_path
        self.size = size
        self.random_crop = random_crop
        self.use_simplified_sketch=use_simplified_sketch
        random.seed(1)

        self.labels = dict()
        self.file_to_caption=self._get_file_to_caption(self.json_path)
        if self.index_path is not None:
            with open(self.index_path, "r") as f:
                self.image_paths = f.read().splitlines()
        else:
            self.image_paths=list(self.file_to_caption.keys())
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths]}

        if self.size is not None and self.size > 0:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size = self.size, 
                                                           interpolation=self.interpolation)
            self.sketch_rescaler = albumentations.SmallestMaxSize(max_size = self.size, 
                                                           interpolation=self.interpolation)

    def _get_file_to_caption(self, json_path):
        file_to_caption=defaultdict(list)
        id_to_caption=defaultdict(list)
        with open(json_path, "r") as f:
            json_data=json.load(f)
            for caption in tqdm(json_data["annotations"], desc="IdtoCaption"):
                id_to_caption[caption["image_id"]].append(caption["caption"])
            for image_dir in tqdm(json_data["images"], desc="FilenameToCaption"):
                file_to_caption[image_dir["file_name"]]=id_to_caption[image_dir['id']]
        return file_to_caption

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        return image

    def preprocess_sketch(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.sketch_rescaler(image=image)["image"]
        return image

    def preprocessor(self, image, ske):
        if not self.use_simplified_sketch:
            width, height=image.shape[0],image.shape[1]
        else:
            width, height=min(ske.shape[0],image.shape[0]),min(ske.shape[1],image.shape[1])
        if not self.random_crop:
            x=(width-self.size)//2
            y=(height-self.size)//2
        else:
            x=random.randint(0, width-self.size)
            y=random.randint(0, height-self.size)
        
        image=image[x:x+self.size, y:y+self.size:, :]
        ske=ske[x:x+self.size, y:y+self.size, :]
        
        return {"image": image, "ske": ske}

    def __getitem__(self, i):
        example = dict()
        image = self.preprocess_image(self.labels["file_path_"][i])
        if not self.use_simplified_sketch:
            sketch= self.preprocess_sketch(self.labels["file_path_"][i].replace("image", "sketch"))
        else:
            sketch= self.preprocess_sketch(self.labels["file_path_"][i].replace("image", "sketch_simplification"))
        if self.size is not None:
            processed = self.preprocessor(image=image,
                                          ske=sketch
                                          )
        else:
            processed = {"image": image,
                         "ske": sketch
                         }
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        example["sketch"] = (processed["ske"]/255.0).astype(np.float32)

        captions=self.file_to_caption[self.labels["relative_file_path_"][i]]
        if len(captions)>1:
            example["caption"]= captions[random.randint(0, len(captions)-1)]
        else:
            example["caption"]= captions 

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class AnnotationsWithCaptionSeg(Dataset):
    def __init__(self, data_root, json_path,
                 size=None, random_crop=False, 
                 n_labels=183, interpolation="bicubic"):
        self.data_root=data_root
        self.json_path=json_path
        self.size = size
        self.random_crop = random_crop
        self.n_labels=n_labels
        random.seed(1)

        self.labels = dict()
        self.file_to_caption=self._get_file_to_caption(self.json_path)
        self.image_paths=list(self.file_to_caption.keys())
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths]}

        if self.size is not None and self.size > 0:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size = self.size, 
                                                            interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                            interpolation=self.interpolation)

    def _get_file_to_caption(self, json_path):
        file_to_caption=defaultdict(list)
        id_to_caption=defaultdict(list)
        with open(json_path, "r") as f:
            json_data=json.load(f)
            for caption in tqdm(json_data["annotations"], desc="IdtoCaption"):
                id_to_caption[caption["image_id"]].append(caption["caption"])
            for image_dir in tqdm(json_data["images"], desc="FilenameToCaption"):
                file_to_caption[image_dir["file_name"]]=id_to_caption[image_dir['id']]
        return file_to_caption

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        return image

    def preprocess_segmentation(self, image_path):
        segmentation = Image.open(image_path).convert("RGB")
        # assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        return segmentation
    
    def preprocessor(self, image, seg):
        width, height=image.shape[0],image.shape[1]
        if not self.random_crop:
            x=(width-self.size)//2
            y=(height-self.size)//2
        else:
            x=random.randint(0, width-self.size)
            y=random.randint(0, height-self.size)
        
        image=image[x:x+self.size, y:y+self.size:, :]
        seg=seg[x:x+self.size, y:y+self.size, :]
        
        return {"image": image, "seg": seg}

    def __getitem__(self, i):
        example = dict()
        image = self.preprocess_image(self.labels["file_path_"][i])
        segmentation =self.preprocess_segmentation(self.labels["file_path_"][i].replace("jpg", "png").replace("image", "segmentation"))
        if self.size is not None:
            processed = self.preprocessor(image=image,
                                          seg=segmentation,
                                          )
        else:
            processed = {"image": image,
                         "seg": segmentation,
                         }
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        example["segmentation"] = (processed["seg"]/255.0).astype(np.float32)

        captions=self.file_to_caption[self.labels["relative_file_path_"][i]]
        if len(captions)>1:
            example["caption"]= captions[random.randint(0, len(captions)-1)]
        else:
            example["caption"]= captions 

        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
