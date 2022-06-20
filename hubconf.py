import os
import torch
import torchvision

from models.alexnet_gn import alexnet_gn as _alexnet_gn
import models.resnet as _resnet

dependencies = ['torch', 'torchvision']

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform
  
def alexnetgn_ipcl_ref1(pretrained=True, **kwargs):
    """Official ipcl imagenet model from the paper `A self-supervised domain-general learning framework for human ventral stream representation <https://github.com/grez72/publications/blob/master/pdfs/Konkle_et_al-2022-Nature_Communications.pdf>`.    
    
    This model instance corresponds to Supplementary Table Ref#1 (alexnet_gn trained on imagenet with instance-prototype-contrastive learning; IPCL).
    
    Args:
        pretrained (bool): whether to load pre-trained weights
    returns: model, transform
        model: the requested model
        transform: the validation transforms needed to pre-process images
        
    """
    
    model = _alexnet_gn(out_dim=128, l2norm=True)
          
    if pretrained:
        checkpoint_name = "06_instance_imagenet_AlexNet_n5_lr03_pct40_t07_div1000_e100_bs128_bm20_gn_stack_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_ref1-51bcc71556.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            filename=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '51bcc71556'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model, transform
  
