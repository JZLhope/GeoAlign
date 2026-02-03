from torchvision import transforms

TRANSFORM_DICT = {
    'Resize256': transforms.Resize((256,256), interpolation=3),
    'Resize384': transforms.Resize((384,384), interpolation=3),
    'ToTensor': transforms.ToTensor(),
    'normalize': transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]),
}

def build_transform(t_name_list):
    t_list = []
    for t_name in t_name_list:
        t_list.append(TRANSFORM_DICT[t_name])
    return transforms.Compose(t_list)

