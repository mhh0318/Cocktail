import sys
import json
from tqdm import tqdm
from pathlib import Path

from ccldm.data.laion_aesthetics import LAION_Aesthetics_PATH_STRUCTURE

save_path = sys.argv[1]
split = sys.argv[2]

top_level = "data/LAION-Aesthetics-6.5"
top_level = Path(top_level)
sub_paths = {name: top_level.joinpath(sub_path) for name, sub_path in LAION_Aesthetics_PATH_STRUCTURE[split].items()}

image_paths = list()
for image_dir in tqdm(list([f for f in sub_paths["data"].iterdir() if f.is_dir()])):
    dir_name = image_dir.name
    for image_file in image_dir.glob("*.jpg"):
        file_name = dir_name + '/' + image_file.name
        if sub_paths["sketch"].joinpath(file_name).exists() and sub_paths["segmentation"].joinpath(file_name).exists() and sub_paths["gray_seg"].joinpath(file_name).exists():
            # image_size = Image.open(image_file).size
            # if image_size[0] <= max_resolution and image_size[1] <= max_resolution:
            image_paths.append(file_name)

print(len(image_paths))
with open(save_path,'w') as f:
    json.dump(image_paths, f)
