import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import numpy as np
from openpose import OpenposeDetector
from openpose.util import HWC3


if __name__ == '__main__':
    input_path = sys.argv[1]
    out_path = sys.argv[2]
    hand_and_face = True if len(sys.argv)<4 else sys.argv[3]
    
    model = OpenposeDetector()
    
    image = HWC3(np.array(Image.open(input_path)).astype(np.uint8))
    if hand_and_face == "True":
        openpose_map = model(image, hand_and_face=True)
    else:
        openpose_map = model(image, hand_and_face=False)
        
    Image.fromarray(openpose_map).save(out_path)
