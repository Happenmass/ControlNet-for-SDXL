# ControlNet

Modified from original repo of controlnet https://github.com/lllyasviel/ControlNet

ControlNet is a neural network structure to control diffusion models by adding extra conditions.

![img](github_page/he.png)

It copys the weights of neural network blocks into a "locked" copy and a "trainable" copy. (actually the UNet part in SD network)

The "trainable" one learns your condition. The "locked" one preserves your model. 

No constructure change has been made in controleNet

## Device Requirements
All params need to be trained is about 2.6B, it is tested that at least 32GB GPU memory needed to train the model.


# Stable Diffusion + ControlNet

In stable diffusion 2.1 and 1.5 ,by repeating the above simple structure 13 times, we can control stable diffusion in this way:

![img](github_page/sd.png)

In Stable diffusion XL, there are only 3 Encoder blocks, so  the above simple structure only need to be repeated 10 times




# Production-Ready Pretrained Models

First create a new conda environment

    conda env create -f environment.yaml
    conda activate control


## Generate SDXL + ControlNet

    python tool_add_controlnet.py --model sd_xl --controlnet --output sd_xl_controlnet



# Annotate Your Own Data

We provide simple python scripts to process images.

[See a gradio example here](docs/annotator.md).

# Train with Your Own Data

Training a ControlNet is as easy as (or even easier than) training a simple pix2pix. 

[See the steps here](docs/train.md).


