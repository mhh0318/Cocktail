import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SAN import Segmenter

if __name__ == "__main__":
    import sys

    input_path = sys.argv[1]
    save_path = sys.argv[2]
    
    segmenter = Segmenter()
    segmenter(input_path, save_path)
    