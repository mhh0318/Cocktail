import numpy as np
from PIL import Image
import cv2

def multi_labels_dilation(gray_img: np.array, background=0, kernel_size=2):
    labels=np.unique(gray_img)
    
    gray_erosion_img=gray_img.copy()
    for label in labels:
        if not np.isin(label, background):
            continue

        binary_img=np.zeros_like(gray_img)
        binary_img[gray_img!=label]=0
        binary_img[gray_img==label]=255
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation = cv2.dilate(binary_img, kernel, iterations=kernel_size)
        gray_erosion_img[dilation==255]=label
    
    return gray_erosion_img

def preprocess_sketch(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    return image

def preprocess_segmentation(image_path):
    segmentation = Image.open(image_path)
    if not segmentation.mode == "RGB":
        segmentation = segmentation.convert("RGB")
    segmentation = np.array(segmentation).astype(np.uint8)
    return segmentation

def preprocess_gray_seg(image_path):
    segmentation = Image.open(image_path)
    assert segmentation.mode == "L", segmentation.mode
    segmentation = np.array(segmentation).astype(np.uint8)+1
    return segmentation

sketch = preprocess_sketch("./sketch.jpg")
seg = preprocess_segmentation("./seg_color.png")
gray_seg = preprocess_gray_seg("./seg.png")

all_labels = np.unique(gray_seg)
random_labels = np.random.choice(all_labels, np.random.randint(0, len(all_labels)+1), replace=False)
mask = np.isin(gray_seg, random_labels)
composd = sketch
composd[mask] = seg[mask]
Image.fromarray(composd).save("./composed.jpg")

print("done!")
