import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer)
from PIL import Image
import os
import torch
import random
import shutil
torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from torchvision import transforms, models

img_dir = os.listdir('jpg/')
path = 'jpg/'

df_master = pd.read_csv('Image_data.csv')
device = 'cpu'

model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning').to(device)
processor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

def image_text_feat(image):
    image_path = f'{path}/{image}'
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert("RGB")

    pixel_values = processor(image,return_tensors = 'pt').pixel_values.to(device)
    output = model.generate(pixel_values,max_length = 40, num_beams = 10)
    caption = tokenizer.decode(output[0],skip_special_tokens=True)

    flower_name = df_master[df_master['Image_Name'] == image]['Name']

    if 'flower' in caption.lower():
        final_text = caption.replace('flower',flower_name)
    else:
        final_text = caption

    # image features extraction using densesnet121 model
    model_densenet121 = models.densenet121(weights='DEFAULT')
    model_densenet121.eval()
    features = model_densenet121.features

    image_preprocess = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.428,0.456,0.406],
                                                              std = [0.229,0.224,0.225])])
    image_tensor = image_preprocess(image).unsqueeze(0)
    image_feat = features(image_tensor)
    image_feat = torch.nn.functional.adaptive_avg_pool2d(image_feat,(1,1))
    image_feat = image_feat.view(image_feat.size(0),-1)
    image_feat = image_feat.detach().numpy()

    return final_text, image_feat

# Dataframe to store image text and features
# Dataframe to store image text and features
d,image_name, image_desc, img_f, n = [],[],[],[], 0
for i in img_dir:
    text, feat = image_text_feat(i)
    d.append(n)
    image_name.append(i)
    image_desc.append(text)
    img_f.append(feat)
    n += 1

df_master = pd.DataFrame({'ID':d, 'Image_Name':image_name, 'Image_Description':image_desc, 'Image_Features': img_f})

image_features1 = img_f.copy()

