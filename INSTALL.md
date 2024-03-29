# Installation

This document contains detailed instructions for installing the necessary dependencies for LLB. The instrustions have been tested on an Ubuntu 16.04 system.

### Requirements  
* Conda installation with Python 3.7. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU.

## Step-by-step instructions  
#### Create and activate a conda environment
```bash
conda create --name LLB python=3.7
conda activate LLB
```


#### Install PyTorch  
Install PyTorch with cuda10.  
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

**Note:**  
- It is possible to use any PyTorch supported version of CUDA (not necessarily v10).   
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install matplotlib, pandas, tqdm, opencv, scikit-image, visdom, tikzplotlib, gdown, and tensorboad  
```bash
conda install matplotlib pandas tqdm
pip install opencv-python visdom tb-nightly scikit-image tikzplotlib
```


#### Install the coco and lvis toolkits  
```bash
conda install cython
pip install pycocotools
pip install lvis
```


#### Install jpeg4py  
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip install jpeg4py 
```

**Note:** The first step (```sudo apt-get install libturbojpeg```) can be optionally ignored, in which case OpenCV's imread() will be used to read the images. However the second step is a must.  

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py.  


#### Setup the environment  
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.
