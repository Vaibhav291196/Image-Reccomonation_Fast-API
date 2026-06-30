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
import cv2 as cv
from natsort import natsorted

df_master = pd.read_csv('Image_data.csv')
image_text_list = df_master['Image_Description'].str.lower().tolist()
images_name = df_master['Image_Name'].tolist()
imagedir = 'uploaded_images/'

img_f = torch.load("image_features1",
                     map_location="cpu",
                     weights_only=False)

app = FastAPI()

@app.middleware("http")
async def disable_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response

@app.get('/image type')
async def text_recommondation(Image_Description:str, Number_Of_Images:int):

  vec = TfidfVectorizer(use_idf=False,
                        norm=None)

  text=Image_Description.lower()
  combined_text = image_text_list+[text]
  result = vec.fit_transform(combined_text)

  data_tfidf = pd.DataFrame(result.toarray(),columns=sorted(vec.vocabulary_))

  idx=data_tfidf[[i for i in text.split() if i in vec.vocabulary_.keys()]].sum(axis=1).sort_values(ascending=False).index.to_list()[:Number_Of_Images]

  img = []
  for k in idx:
      a = Image.open(f'jpg/{images_name[k]}')
      a = a.resize([300, 300])
      a = np.array(a)
      a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
      img.append(a)
  combined_img = cv.hconcat(img)
  img__, combined_img = cv.imencode('.jpg', combined_img)

  return Response(content=combined_img.tobytes(), media_type='image/jpeg')

model_densenet121 = models.densenet121(weights='DEFAULT')
model_densenet121.eval()
feature_extractor = model_densenet121.features
img_feat = np.vstack(img_f)
print(img_feat)
print(df_master[['Image_Name','Image_Features']])

@app.post("/add_image")
async def img_reccomondation(upload_image: UploadFile = File(...),Number_of_images:int = Form(...)):
    upload_image.filename = f'{uuid.uuid4()}.jpg'
    contents = await upload_image.read()

    with open(f'{imagedir}{upload_image.filename}', 'wb') as f:
        f.write(contents)
    image_path = Path('uploaded_images/' + upload_image.filename)
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

    similarity = cosine_similarity(features,img_feat)

    similarity = similarity.flatten()

    idx_=pd.DataFrame(similarity).sort_values(by=0,ascending=False).index.to_list()[:Number_of_images]

    img=[]
    for k in idx_:
        a=Image.open(f'jpg/{images_name[k]}')
        a=a.resize([300,300])
        a=np.array(a)
        a=cv.cvtColor(a,cv.COLOR_BGR2RGB)
        img.append(a)
    combined_img = cv.hconcat(img)
    img__, combined_img = cv.imencode('.jpg', combined_img)

    if os.path.exists(image_path):
        os.remove(image_path)

    return Response(content=combined_img.tobytes(), media_type='image/jpeg')

if __name__ == "__main__":
    uvicorn.run(app)
