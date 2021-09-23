# open_ipcl
 PyTorch implementation and pretrained models for IPCL (instance-prototype contrastive learning). For details see **Beyond category-supervision: instance-level contrastive learning models predict human visual system responses to objects** [[bioRxiv]](https://www.biorxiv.org/content/10.1101/2021.05.28.446118v1.full)

<p align="center">
  <img src="images/ipcl.png" width="60%" title="ipcl">
</p>

## Load Pretrained Models

Models are numbered to align with Supplementary Table 1 in our paper [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.05.28.446118v1.full). For example, to load a self-supervised model, trained with IPCL:

```
import models
model, transform = models.__dict__['ipcl1']()

```

The transform returned here should be used when getting activations for test images, which in our case were
stimuli from a neuroimaging experiment. For these test images, standard validation transforms (e.g., those used
in knn_eval.py or main_lincls_onecycle.py) would crop out details of the object depicted. The transform returned above
resizes to 224 pixels, then center crops (as opposed to resizing to 256 pixels followed by a center crop).

```
Compose(
    Resize(size=224, interpolation=PIL.Image.BILINEAR)
    CenterCrop(size=(224, 224))
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
```

## Get Activations

To get the activations for any model layer, you can use the FeatureExtractor class. 

```
from PIL import Image
from lib.feature_extractor import FeatureExtractor

# load an image
img = Image.open('./images/cheetah.jpg')

# transform and add batch dimension
img = transform(img).unsqueeze(0)

# get features from fc7
model.eval()
with FeatureExtractor(model, 'fc7') as extractor:
    features = extractor(img)
    for name,val in features.items():
        print(name, val.shape)
        
# get features from fc7, fc8, and l2norm layers
model.eval()
with FeatureExtractor(model, ['fc7','fc8','l2norm']) as extractor:
    features = extractor(img)
    for name,val in features.items():
        print(name, val.shape)
        
# get features from conv_block1.0, conv_block1.1, conv_block1.2
model.eval()
with FeatureExtractor(model, ['conv_block_1.0','conv_block_1.1','conv_block_1.2']) as extractor:
    features = extractor(img)
    for name,val in features.items():
        print(name, val.shape)        
        
```        

## Train Models
Our original training code was based on https://github.com/zhirongw/lemniscate.pytorch, but the IPCL models were slow to train (~21 days on a single Titan X Pascal). The same code runs faster on newer gpus (e.g., ~7 days on a Tesla V100). 

***replicate original ipcl_alexnet_gn model (warning could be slow, unless you have a Tesla V100)***
```
python train_original.py --data /path/to/imagenet
```

We found the primary bottleneck for training these models was the fact that IPCL augments each image N times (N=5 in our experiments), so we implemented custom transforms that perform augmentations on the GPU, which required a change to the colorspace conversion for color_jitter. These models train almost twice as fast (~11 days on a single Titan X Pascal gpu; ~4 days on a Tesla V100), and fit neural data equally well, but perform slightly less accurately on Imagenet Classification (e.g., knn accurcy = 39.3% with torchvision transforms, and 32.8% with custom transforms).  

***train ipcl_alexnet_gn with faster augmentations (warning could be slow)***
```
python train_orginal_fastaug.py --data /path/to/imagenet
```