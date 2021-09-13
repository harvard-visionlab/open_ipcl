import os
import torch
from torch.hub import load_state_dict_from_url
from torchvision.datasets.folder import default_loader
from pathlib import Path 
from torchvision import transforms

from .alexnet_gn import *
from .resnet import *

url_root = os.path.join(
    "https://visionlab-pretrainedmodels.s3.amazonaws.com",
    "project_instancenet",
    "ipcl_alpha"
)

def build_alexnet_model(weights_url, mean, std):
    
    model = alexnet_gn()
    
    print(f"=> loading checkpoint: {Path(weights_url).name}")
    checkpoint = load_state_dict_from_url(weights_url, model_dir=None, map_location=torch.device('cpu'))
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    print("=> state loaded.")
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return model, transform

def ipcl1():
    model_name = 'ipcl_alpha_alexnet_gn_u128_stack'
    filename = '06_instance_imagenet_AlexNet_n5_lr03_pct40_t07_div1000_e100_bs128_bm20_gn_stack_final_weights_only.pth.tar'
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 1,
        "type": "ipcl",
        "details": "primary model",
        "aug": "Set 1",
        "top1_knn": 38.4,
        "top1_linear": 39.5
    }
    print(model.info)
    return model, transform

def ipcl2():
    model_name = 'ipcl_alpha_alexnet_gn_u128_rep2'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 2,
        "type": "ipcl",
        "details": "variation: new code base",
        "aug": "Set 1",
        "top1_knn": 38.4,
        "top1_linear": 39.7
    }
    print(model.info)
    return model, transform

def ipcl3():
    model_name = 'ipcl_alpha_alexnet_gn_u128_redux'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 3,
        "type": "ipcl",
        "details": "variation: one cycle lr & momentum (73 epochs)",
        "aug": "Set 1",
        "top1_knn": 35.4,
        "top1_linear": 35.7
    }
    print(model.info)
    return model, transform

def ipcl4():
    model_name = 'ipcl_alpha_alexnet_gn_u128_ranger'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 4,
        "type": "ipcl",
        "details": "variation: explore ranger (82 epochs)",
        "aug": "Set 1",
        "top1_knn": 37.5,
        "top1_linear": 32.2
    }
    print(model.info)
    return model, transform

def ipcl5():
    model_name = 'ipcl_alpha_alexnet_gn_u128_transforms'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 5,
        "type": "ipcl",
        "details": "variation: custom transforms (82 epochs)",
        "aug": "Set 1",
        "top1_knn": 36.9,
        "top1_linear": 38.5
    }
    print(model.info)
    return model, transform

def ipcl6():
    model_name = 'ipcl_alexnet_gn_u128_imagenet'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 6,
        "type": "ipcl",
        "details": "ImageNet baseline with new augmentations",
        "aug": "Set 2",
        "top1_knn": 35.1,
        "top1_linear": None,
    }
    print(model.info)
    return model, transform

def ipcl7():
    model_name = 'ipcl_alexnet_gn_u128_openimagesv6'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 7,
        "type": "ipcl",
        "details": "train on independent object dataset, OpenImagesV6",
        "aug": "Set 2",
        "top1_knn": 33.3,
        "top1_linear": None
    }
    print(model.info)
    return model, transform

def ipcl8():
    model_name = 'ipcl_alexnet_gn_u128_places2'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 8,
        "type": "ipcl",
        "details": "train on scene dataset, Places2",
        "aug": "Set 2",
        "top1_knn": 30.9,
        "top1_linear": None
    }
    print(model.info)
    return model, transform

def ipcl9():
    model_name = 'ipcl_alexnet_gn_u128_vggface2'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 9,
        "type": "ipcl",
        "details": "train on face dataset, VggFace2",
        "aug": "Set 2",
        "top1_knn": 12.4,
        "top1_linear": None
    }
    print(model.info)
    return model, transform

def ipcl10():
    model_name = 'ipcl_alexnet_gn_u128_FacesPlacesObjects1281167'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 10,
        "type": "ipcl",
        "details": "train on faces-places-objects-1x-ImageNet",
        "aug": "Set 2",
        "top1_knn": 31.6,
        "top1_linear": None
    }
    print(model.info)
    return model, transform

def ipcl11():
    model_name = 'ipcl_alexnet_gn_u128_FacesPlacesObjects1281167x3'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 11,
        "type": "ipcl",
        "details": "train on faces-places-objects-3x-ImageNet",
        "aug": "Set 2",
        "top1_knn": 33.9,
        "top1_linear": None
    }
    print(model.info)
    return model, transform

def ipcl12():
    model_name = 'ipcl_alpha_alexnet_gn_s1000_imagenet_wus_aug'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 12,
        "type": "category supervised",
        "details": "trained with 5 augmentations per image to match IPCL",
        "aug": "Set 1",
        "top1_knn": 58.8,
        "top1_linear": 55.7
    }
    print(model.info)
    return model, transform

def ipcl13():
    model_name = 'wusnet_alexnet_gn_s1000'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 13,
        "type": "category supervised",
        "details": "trained with single augmentation per image",
        "aug": "Set 1",
        "top1_knn": 55.5,
        "top1_linear": 54.5
    }
    print(model.info)
    return model, transform

def ipcl14():
    model_name = 'ipcl_alexnet_gn_s1000_imagenet'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 14,
        "type": "category supervised",
        "details": "ImageNet baseline with new augmentations",
        "aug": "Set 2",
        "top1_knn": 56.0,
        "top1_linear": None
    }
    print(model.info)
    return model, transform

def ipcl15():
    model_name = 'ipcl_alexnet_gn_s1000_imagenet_rep1'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 15,
        "type": "category supervised",
        "details": "primary model",
        "aug": "Set 2",
        "top1_knn": 56.0,
        "top1_linear": None
    }
    print(model.info)
    return model, transform

def ipcl16():
    model_name = 'ipcl_alpha_alexnet_gn_u128_random'
    filename = ''
    weights_url = os.path.join(url_root, filename)    
    model, transform = build_alexnet_model(weights_url, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.info = {
        "ref#": 16,
        "type": "untrained",
        "details": "untrained model with random weights and biases",
        "aug": "-",
        "top1_knn": 3.5,
        "top1_linear": 7.2
    }
    print(model.info)
    return model, transform
