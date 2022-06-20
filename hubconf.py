import os
import torch
import torchvision

from models.alexnet_gn import alexnet_gn as _alexnet_gn
import models.resnet as _resnet

dependencies = ['torch', 'torchvision']

_doc ="""Official ipcl imagenet model from the paper `A self-supervised domain-general learning framework for human ventral stream representation <https://github.com/grez72/publications/blob/master/pdfs/Konkle_et_al-2022-Nature_Communications.pdf>`.    
    
    This model instance corresponds to Supplementary Table {}, {}. 
        - {}
    
    Args:
        pretrained (bool): whether to load pre-trained weights
        
    returns: model, transform
        model: the requested model
        transform: the validation transforms needed to pre-process images
        
    """

def _docstring_parameter(*sub):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform

# -------------------------------------------------
#  Official models from published paper
# -------------------------------------------------

@_docstring_parameter(_doc.format("Ref#1","(alexnet_gn trained on imagenet with instance-prototype-contrastive learning; IPCL)"))
def alexnetgn_ipcl_ref1(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True)
          
    if pretrained:
        checkpoint_name = "06_instance_imagenet_AlexNet_n5_lr03_pct40_t07_div1000_e100_bs128_bm20_gn_stack_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_ref1-51bcc71556.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '51bcc71556'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#2","(alexnet_gn trained on imagenet with instance-prototype-contrastive learning; IPCL)"))
def alexnetgn_ipcl_ref2(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "06_instance_imagenet_AlexNet_n5_lr03_pct40_t07_div1000_e100_bs128_bm20_gn_rep2_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_ref2-a4b3151e28.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = 'a4b3151e28'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#3","(alexnet_gn trained on imagenet with instance-prototype-contrastive learning; IPCL)"))
def alexnetgn_ipcl_ref3(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_dim128_unsupervised_redux_checkpoint_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_ref3-543e478735.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '543e478735'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#4","(alexnet_gn trained on imagenet with instance-prototype-contrastive learning; IPCL)"))
def alexnetgn_ipcl_ref4(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_dim128_unsupervised_ranger_checkpoint_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_ref4-e8d0736300.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = 'e8d0736300'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#5","(alexnet_gn trained on imagenet with instance-prototype-contrastive learning; IPCL)"))
def alexnetgn_ipcl_ref5(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_dim128_unsupervised_transforms_checkpoint_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_ref5-d06507981d.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = 'd06507981d'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#6","(alexnet_gn trained on imagenet with IPCL; visual-diet variation set)","trained with reduced range of random-resize-crop range"))
def alexnetgn_ipcl_diet_imagenet_ref6(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_u128_imagenet_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_diet_imagenet_ref6-fc6efe7bec.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = 'fc6efe7bec'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#7","(alexnet_gn trained on OpenImagesV6 with IPCL; visual-diet variation set)","trained with reduced range of random-resize-crop range"))
def alexnetgn_ipcl_diet_openimagesv6_ref7(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_u128_openimagesv6_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_diet_openimagesv6_ref7-79a120b915.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '79a120b915'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#8","(alexnet_gn trained on Places2 with IPCL; visual-diet variation set)","trained with reduced range of random-resize-crop range"))
def alexnetgn_ipcl_diet_places2_ref8(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_u128_places2_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_diet_places2_ref8-343c15875d.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '343c15875d'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#9","(alexnet_gn trained on VGGFace2 with IPCL; visual-diet variation set)","trained with reduced range of random-resize-crop range"))
def alexnetgn_ipcl_diet_vggface2_ref9(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_u128_vggface2_lr001_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_diet_vggface2_ref9-23ef0936c4.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '23ef0936c4'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#10","(alexnet_gn trained on FacesPlacesObjects1281167 with IPCL; visual-diet variation set)","trained with reduced range of random-resize-crop range"))
def alexnetgn_ipcl_diet_FacesPlacesObjects1281167_ref10(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_u128_FacesPlacesObjects1281167_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_diet_FacesPlacesObjects1281167_ref10-25e539c255.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '25e539c255'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#11","(alexnet_gn trained on FacesPlacesObjects1281167 with IPCL; visual-diet variation set)","trained with reduced range of random-resize-crop range"))
def alexnetgn_ipcl_diet_FacesPlacesObjects1281167x3_ref11(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=128, l2norm=True, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_u128_FacesPlacesObjects1281167x3_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_ipcl_diet_FacesPlacesObjects1281167x3_ref11-af6119b74a.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = 'af6119b74a'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#12","(alexnet_gn trained on imagenet with category-supervision)","trained with 5 augmentations per image to match IPCL, Aug Set 1"))
def alexnetgn_supervised_augset1_5x_ref12(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=1000, l2norm=False, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_s1000_imagenet_wus_aug_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_supervised_augset1_5x_ref12-3a41dd57b4.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '3a41dd57b4'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#13","(alexnet_gn trained on imagenet with category-supervision)","trained with standard single augmentation per image, Aug Set 1"))
def alexnetgn_supervised_augset1_1x_ref13(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=1000, l2norm=False, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_supervised_final.pth.tar"
        cache_file_name = "alexnetgn_supervised_augset1_1x_ref13-1ea33cba82.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '1ea33cba82'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#14","(alexnet_gn trained on imagenet with category-supervision)","ImageNet baseline with new augmentations, Set 2"))
def alexnetgn_supervised_augset2_ref14(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=1000, l2norm=False, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_s1000_imagenet_base_aug3_blur0_rot0_w0_d0_std02_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_supervised_augset2_ref14-f9c1603f16.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = 'f9c1603f16'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.5, 0.5, 0.5]], std=[0.2, 0.2, 0.2])
    
    return model, transform

@_docstring_parameter(_doc.format("Ref#15","(alexnet_gn trained on imagenet with category-supervision)","ImageNet baseline with new augmentations, Set 2"))
def alexnetgn_supervised_augset2_rep1_ref15(pretrained=True, **kwargs):
    """{0}"""
    
    model = _alexnet_gn(out_dim=1000, l2norm=False, **kwargs)
          
    if pretrained:
        checkpoint_name = "alexnet_gn_s1000_imagenet_base_aug3_blur0_rot0_w0_d0_std02_rep1_final_weights_only.pth.tar"
        cache_file_name = "alexnetgn_supervised_augset2_rep1_ref15-5b19b17783.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(
            url=f'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/{checkpoint_name}', 
            map_location='cpu',
            file_name=cache_file_name,
            check_hash=True
        )
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.hashid = '5b19b17783'
        model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
    
    transform = _transform(mean=[0.5, 0.5, 0.5]], std=[0.2, 0.2, 0.2])
    
    return model, transform


# -------------------------------------------------
#  additional models
# -------------------------------------------------
  
