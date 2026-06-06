# 🖼️ Image Recommendation System using DenseNet121 and FastAPI

A content-based image recommendation system built with **FastAPI**, **PyTorch**, and **DenseNet121**. The application supports both:

- 🔍 Text-to-Image Retrieval
- 📷 Image-to-Image Retrieval

The system extracts deep visual features using a pretrained DenseNet121 model and returns the most relevant images based on visual similarity or textual descriptions.

---

## 🚀 Features

### 1. Text-Based Image Retrieval
Users provide a text query, and the system retrieves the most relevant image from the dataset using TF-IDF based keyword matching.

### 2. Image-Based Recommendation
Users upload an image, and the system returns visually similar images using deep feature embeddings extracted from DenseNet121.

### 3. REST API Support
Built using FastAPI for fast and scalable deployment.

### 4. Deep Learning Powered
Uses pretrained DenseNet121 for feature extraction without requiring additional model training.

---

## 🏗️ Project Architecture

```text
User Query
    │
    ├── Text Query
    │       │
    │       └── TF-IDF Matching
    │               │
    │               └── Most Relevant Image
    │
    └── Uploaded Image
            │
            └── DenseNet121 Feature Extraction
                    │
                    └── Cosine Similarity
                            │
                            └── Similar Images
```

---

## 📂 Project Structure

```text
project/
│
├── Images_1/                 # Dataset Images
├── images/                   # Uploaded Images
├── dataset.xlsx              # Image Metadata
├── app.py                    # FastAPI Application
├── requirements.txt
└── README.md
```

---

## 🧠 Technologies Used

- Python
- FastAPI
- PyTorch
- TorchVision
- DenseNet121
- OpenCV
- Pandas
- NumPy
- Scikit-Learn
- TF-IDF Vectorization
- Cosine Similarity

---

## ⚙️ How It Works

### Feature Extraction

The system uses a pretrained DenseNet121 model:

```python
model = models.densenet121(weights='DEFAULT')
feature_extractor = model.features
```

For every image:

1. Resize image
2. Normalize image
3. Extract feature maps
4. Apply Global Average Pooling
5. Store feature vectors

---

### Text Retrieval Pipeline

1. Read image names from dataset.
2. Convert image names into text descriptions.
3. Apply TF-IDF vectorization.
4. Match user query against dataset descriptions.
5. Return the highest-ranked image.

---

### Image Retrieval Pipeline

1. Upload image through API.
2. Extract DenseNet features.
3. Compute cosine similarity against all dataset images.
4. Retrieve top-N most similar images.
5. Return combined recommendation image.

---

## 📊 Similarity Metric

Cosine Similarity is used for image matching:

\[
Similarity(A,B)=\frac{A \cdot B}{||A|| ||B||}
\]

Higher values indicate greater visual similarity.

---

## 🔌 API Endpoints

### Text-to-Image Search

```http
GET /image type
```

#### Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| text | string | Search query |
| number_of_images | int | Number of images to retrieve |

#### Example

```http
GET /image type?text=red flower
```

---

### Image Recommendation

```http
POST /add_image
```

#### Form Data

| Parameter | Type |
|------------|------|
| upload_image | Image File |
| Number_of_images | Integer |

#### Example

```bash
curl -X POST \
-F "upload_image=@test.jpg" \
-F "Number_of_images=5" \
http://localhost:8000/add_image
```

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/image-recommendation-system.git

cd image-recommendation-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

Application will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 📈 Future Improvements

- Multi-image recommendations
- Hybrid text + image retrieval
- CLIP-based multimodal search
- FAISS indexing for large-scale datasets
- User feedback and relevance ranking
- Docker deployment

---

## 🎯 Applications

- E-commerce product recommendation
- Fashion image retrieval
- Artwork search
- Visual search engines
- Medical image similarity search
- Digital asset management

---

## 👨‍💻 Author

**Vaibhav Solanke**

- Data Science & Machine Learning Enthusiast
- Computer Vision Researcher
- FastAPI & Deep Learning Developer

---

## 📜 License

This project is licensed under the MIT License.
