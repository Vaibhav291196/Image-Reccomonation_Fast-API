import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import Response
import uvicorn
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
from pathlib import Path
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import uuid
from natsort import natsorted
import cv2 as cv

app = FastAPI()

@app.middleware("http")
async def disable_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


imagedir = 'images/'
images__ = 'Images_1/'

images_list = natsorted(os.listdir(images__))

df = pd.read_excel('dataset.xlsx')

image_text_list=df['image_name'].str.lower().str.replace('-',' ').tolist()

model = models.densenet121(weights='DEFAULT')
model.eval()

feature_extractor=model.features

image_features=[]

for i in images_list:


  preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.428,0.456,0.406],
                                                      std = [0.229,0.224,0.225])
                                ])

  input_tensor = preprocess(Image.open(f'Images_1/{i}').convert('RGB')).unsqueeze(0)

  features = feature_extractor(input_tensor)
  features = torch.nn.functional.adaptive_avg_pool2d(features,(1,1))
  features = features.view(features.size(0),-1)

  image_features.append(features.detach().numpy())

image_features=np.vstack(image_features)

@app.get('/image type')
async def text_recommondation(text:str,number_of_images:int=Form(...)):

  vec = TfidfVectorizer(use_idf=False,
                        norm=None)

  text=text.lower()
  combined_text = image_text_list+[text]
  result = vec.fit_transform(combined_text)

  data_tfidf = pd.DataFrame(result.toarray(),columns=sorted(vec.vocabulary_))

  idx=data_tfidf[[i for i in text.split() if i in vec.vocabulary_.keys()]].sum(axis=1).sort_values(ascending=False).index.to_list()[:number_of_images]


  image_path = f'{images__}{images_list[idx[0]]}'
  return FileResponse(f'Images_1/{images_list[idx[0]]}')

@app.post("/add_image")
async def img_reccomondation(upload_image: UploadFile = File(...),Number_of_images:int = Form(...)):
    upload_image.filename = f'{uuid.uuid4()}.jpg'
    contents = await upload_image.read()

    with open(f'{imagedir}{upload_image.filename}', 'wb') as f:
        f.write(contents)
    image_path = Path('images/' + upload_image.filename)
    im_=FileResponse(image_path)

    preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.428,0.456,0.406],
                                                      std = [0.229,0.224,0.225])
                                ])

    input_tensor = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)

    features = feature_extractor(input_tensor)
    features = torch.nn.functional.adaptive_avg_pool2d(features,(1,1))
    features = features.view(features.size(0),-1)
    features = features.detach().numpy()

    similarity = cosine_similarity(features,image_features)

    similarity = similarity.flatten()

    idx_=pd.DataFrame(similarity).sort_values(by=0,ascending=False).index.to_list()[:Number_of_images]

    img=[]
    for k in idx_:
        a=Image.open(f'Images_1/{images_list[k]}')
        a=a.resize([300,300])
        a=np.array(a)
        a=cv.cvtColor(a,cv.COLOR_BGR2RGB)
        img.append(a)
    combined_img = cv.hconcat(img)
    img__, combined_img = cv.imencode('.jpg', combined_img)

    return Response(content=combined_img.tobytes(),media_type='image/jpeg')

if __name__ == "__main__":
    uvicorn.run(app)
