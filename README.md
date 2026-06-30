# 🌸 AI-Based Flower Image Recommendation System

An AI-powered recommendation system that recommends flower images based on **text descriptions** or **uploaded flower images**. The project combines **Computer Vision**, **Natural Language Processing**, and **Deep Learning** techniques to build a multimodal content-based image retrieval system deployed using **FastAPI**.

---

# 📖 Project Overview

This project consists of two major modules:

1. **Feature Generation Pipeline (`image_desc_feat.py`)**
   - Generates image captions using a pretrained image captioning model.
   - Extracts deep visual features using DenseNet121.
   - Stores captions and image embeddings for fast inference.

2. **Recommendation API (`app.py`)**
   - Provides REST APIs for text-based and image-based recommendations.
   - Uses precomputed image features to perform efficient image retrieval.

---

# ✨ Features

- 🌼 Automatic Image Caption Generation
- 🧠 Deep Feature Extraction using DenseNet121
- 🔍 Text-Based Flower Recommendation
- 🖼 Image Similarity Recommendation
- 🚀 FastAPI REST API
- ⚡ Fast inference using precomputed embeddings
- 💻 CPU compatible
- 📦 Easy deployment

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Backend | FastAPI |
| Deep Learning | PyTorch |
| Computer Vision | TorchVision |
| NLP | Hugging Face Transformers |
| Image Processing | OpenCV, Pillow |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-Learn |

---

# 📂 Project Structure

```text
Flower-Recommendation-System/

│── app.py
│── image_desc_feat.py
│── Image_data.csv
│── image_features1
│── requirements.txt
│── README.md
│
├── jpg/
│     ├── image1.jpg
│     ├── image2.jpg
│     └── ...
│
└── uploaded_images/
```

---

# 🔄 Project Workflow

```text
Flower Dataset
      │
      ▼
Image Caption Generation
      │
      ▼
DenseNet121 Feature Extraction
      │
      ▼
Store Captions + Image Features
      │
      ▼
FastAPI Recommendation Engine
      │
      ▼
Recommended Flower Images
```

---

# 📄 image_desc_feat.py

This script prepares the flower dataset before deployment.

## Step 1: Load Dataset

All flower images are loaded from the `jpg/` directory together with metadata stored in `Image_data.csv`.

```python
img_dir = os.listdir("jpg/")
```

---

## Step 2: Image Caption Generation

A pretrained **VisionEncoderDecoderModel** (`nlpconnect/vit-gpt2-image-captioning`) generates natural language descriptions for every flower image.

Example

```text
Input Image

↓

"A yellow sunflower blooming in a garden."
```

These captions provide semantic information that is later used for text-based recommendation.

---

## Step 3: Flower Name Replacement

Sometimes the caption contains the generic word **flower**.

The script replaces it with the actual flower name from the dataset.

Example

```text
Generated Caption

"A beautiful flower"

↓

"A beautiful sunflower"
```

This improves recommendation accuracy.

---

## Step 4: Image Feature Extraction

A pretrained **DenseNet121** model is used as a feature extractor.

The classification layer is ignored, and only convolutional features are extracted.

Processing pipeline

```text
Image

↓

Resize

↓

Center Crop

↓

Normalize

↓

DenseNet121

↓

1024-Dimensional Feature Vector
```

These embeddings capture the visual characteristics of every flower image.

---

## Step 5: Store Features

For every image, the following information is stored:

- Image ID
- Image Name
- Generated Caption
- DenseNet Feature Vector

The image embeddings are stored using

```python
torch.save()
```

This avoids recomputing image features during inference, significantly improving API response time.

---

# 📄 app.py

This file implements the FastAPI backend that exposes recommendation APIs.

At startup, the application loads:

- Image descriptions
- Image names
- Precomputed DenseNet feature vectors

This ensures fast recommendations without running the feature extraction pipeline repeatedly.

---

# 🔍 Text-Based Recommendation

**Endpoint**

```http
GET /image type
```

### Input Parameters

| Parameter | Description |
|----------|-------------|
| Image_Description | Text query |
| Number_Of_Images | Number of recommendations |

Example

```text
yellow flower with green leaves
```

### Workflow

```text
User Query

↓

Convert to Lowercase

↓

TF-IDF Vectorization

↓

Keyword Matching

↓

Rank Similar Captions

↓

Return Top-N Images
```

The API converts the user query into TF-IDF vectors and compares it with all generated image captions.

Images with the highest matching scores are combined into a single image and returned.

---

# 🖼 Image-Based Recommendation

**Endpoint**

```http
POST /add_image
```

### Input Parameters

| Parameter | Description |
|----------|-------------|
| upload_image | Flower image |
| Number_of_images | Number of recommendations |

### Workflow

```text
Upload Image

↓

Resize & Normalize

↓

DenseNet121

↓

Feature Vector

↓

Cosine Similarity

↓

Top-N Similar Images
```

The uploaded image is converted into a DenseNet feature vector and compared against all stored feature vectors using cosine similarity.

The most visually similar flower images are returned.

---

# 🤖 AI Models Used

## 1. VisionEncoderDecoder (ViT-GPT2)

Model

```text
nlpconnect/vit-gpt2-image-captioning
```

Purpose

- Automatic image caption generation
- Semantic understanding of flower images

---

## 2. DenseNet121

Purpose

- Deep visual feature extraction
- Represent images as numerical embeddings

Output

```text
1024-Dimensional Feature Vector
```

---

## 3. TF-IDF Vectorizer

Used for

- Text representation
- Keyword matching
- Text similarity

---

## 4. Cosine Similarity

Used for

- Image similarity search

Similarity values closer to **1** indicate visually similar images.

---

# 🚀 Installation

Clone the repository

```bash
git clone <repository-url>

cd Flower-Recommendation-System
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run FastAPI

```bash
uvicorn app:app --reload
```

Open Swagger documentation

```text
http://127.0.0.1:8000/docs
```

---

# 📌 API Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/image type` | GET | Recommend images using text |
| `/add_image` | POST | Recommend images using uploaded image |

---

# 📈 Future Improvements

- Replace TF-IDF with Sentence Transformers or CLIP embeddings.
- Integrate FAISS for large-scale vector search.
- Support batch image recommendations.
- Deploy using Docker.
- Cloud deployment using Azure, GCP, or AWS.
- Add user authentication.
- Improve recommendations using multimodal embeddings.

---

# 🎯 Applications

- Botanical image retrieval
- Plant identification
- Flower recommendation systems
- Educational applications
- Image search engines
- AI-powered e-commerce recommendation systems

---

# 📄 License

This project is developed for educational and research purposes. Feel free to modify and extend it according to your requirements.

---

# 👨‍💻 Author

Developed using **FastAPI**, **PyTorch**, **Hugging Face Transformers**, and **DenseNet121** to demonstrate multimodal AI for image captioning, feature extraction, and intelligent flower image recommendation.
