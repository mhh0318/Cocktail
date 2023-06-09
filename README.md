# *Cocktail*🍸: Mixing Multi-Modality Controls for Text-Conditional Image Generation


<!-- <a href=""><img src="https://img.shields.io/badge/arXiv-2203.10821-b31b1b.svg" height=22.5></a>  -->
<a href="https://mhh0318.github.io/cocktail/"><img src="https://img.shields.io/badge/Web-Project Page-brightgreen.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" height=22.5></a> 
<a href="https://huggingface.co/MichaelHu/cocktail"><img src="https://img.shields.io/badge/HuggingFace-Checkpoint-yellow.svg" height=22.5></a> 

![img](samples/results/3_sample_0_mark.png)

*James Bond is drinking Cocktail🍸.*



https://github.com/mhh0318/Cocktail/assets/42776955/e2a93a6d-3e36-4e54-8462-b359fa8946fa


  
![img](readme/cktl.png)

Our approach requires only **[one generalized model]**, unlike previous that needed multiple models for mixing multiple modalities. 

Different from currently existing schemes, our scheme does not require modifications to the modal prior of the base model <strong>Fig.(a)</strong>, which results in a significant reduction in cost. Also in the face of multiple modalities we do not need multiple models demonstrated in <strong>Fig.(b)</strong>. Cocktail🍸 fuse the information from multiple modalities like <strong>Fig.(c)</strong> shown.

## Abstract 

![img](readme/teaser.jpg)

We propose Cocktail, a pipeline to mix various modalities into one embedding, amalgamated with a generalized ControlNet (gControlNet), a controllable normalisation (ControlNorm), and a spatial guidance sampling method, to actualize multi-modal and spatially-refined control for text-conditional diffusion models.
## Pipeline

![img](readme/ppl.png)

The parameters indicated by the yellow sections are sourced from the pre-trained model and stay constant, while only those in the blue sections are updated during training, with the gradient back-propagated along the blue arrows. The light grey dashed sections signify additional operations that occur solely during the inference process, specifically, the process of storing attention maps derived from the gControlNet for the sampling stage.

## Results

### [Examples] Cocktail for Multi-modality

![img](readme/fig1.png)

### [Examples] Cocktail for free-modality

![img](readme/free.png)

### [Comparisons] single-modality

![img](readme/fig3.png)

### [Comparisons] multi-modality

Here, the "cross" symbol ❌ and the checkmark symbol ✅ denote the unmatched and matched modalities, respectively. It is important to note that our model accurately captures all modalities.

![img](readme/fig5.png)
![img](readme/fig4.png)





## TODO

- [x] Release Gradio Demo
- [ ] Release sampling codes
- [x] Release inference codes
- [x] Release pre-trained models

## Setup

### Installation Requirmenets

You can create an anaconda environment called `cocktail` with the required dependencies by running:

```
git clone https://github.com/mhh0318/cocktail.git
cd cocktail
conda env create -f environment.yaml
```

### Download Pretrained Weights

Download the pretrained models from [here](https://huggingface.co/MichaelHu/cocktail), and save it to the root dir.

### Gradio Demo
Gradio demo can be launched by:
```bash
python gradio_demo.py [--share]
```
![img](readme/gradio_demo.png) 

### Annotations
We use HED, SAN, and OpenPose to extract the sketch map, segmentation map, and human pose map from the image.
- Extract sketch map:
```python
python annotator/hed.py {/path/to/image.png} {/path/to/sketch.png}
```
- Extract segmentation map:
```python
python annotator/SAN/run.py {/path/to/image.png} {/path/to/seg.png}
```
- Extract human pose map:
```python
python annotator/openpose/run.py {/path/to/image.png} {/path/to/openpose.png}
```

### Quick Inference

For the simultaneous vision-language generation, please run:

```bash
python ./inference {args}
```
args here can be int 0 or 1, as the provided two example conditions.


If the environment is setup correctly, this command should function properly and generate some results in the folder `./samples/results/{args}_sample_{batch}.png`.

![img](samples/results/0_sample_0.png)
![img](samples/results/0_sample_1.png)
![img](samples/results/1_sample_0.png)
![img](samples/results/1_sample_1.png)

## Comments 

Our codebase for the diffusion models builds heavily on [ControlNet](https://github.com/lllyasviel/ControlNet) and  [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

Thanks for the opensourcing!

## Citation
If you use this code for your research, please cite our paper.
```
@article{hu2023cocktail,
  title = {Cocktail: Mixing Multi-Modality Controls for Text-Conditional Image Generation},
  author = {Hu, Minghui and Zheng, Jianbin and Liu, Daqing and Zheng, Chuanxia and Wang, Chaoyue and Tao, Dacheng and Cham, Tat-Jen},
  journal = {arXiv},
  year = {2023},
}
```

