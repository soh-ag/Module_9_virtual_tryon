# 👗 Virtual Try-On AI Backend

This is the **backend service** for the Virtual Try-On AI project.  
It powers the frontend (built with Lovable) by providing an API to detect clothing, segment garments, and generate try-on results using **Roboflow, SAM (Segment Anything), and Segmind Inpainting**.

---

## 🚀 Features
- Upload a person’s photo and **detect clothing items** (upper/lower body).
- **Segment clothing regions** using [Meta’s SAM](https://segment-anything.com/).
- Perform **virtual try-on** using [Segmind Inpainting API](https://segmind.com/).
- Expose APIs via [FastAPI](https://fastapi.tiangolo.com/).
- Supports **GPU acceleration** if available.

---

## 📂 Project Structure
```

backend/
├── virtual\_tryon.py      # Core class with detection, segmentation, and try-on logic
├── main.py               # FastAPI app exposing API endpoints
├── requirements.txt      # Python dependencies
├── uploads/              # Temporary uploaded images
├── outputs/              # Generated output images
└── .env                  # Environment variables (API keys)

````


## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Virtual_tryon.git
cd virtual-tryon-backend
````

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup environment variables

Create a `.env` file in the project root:

```
ROBOFLOW_API_KEY=your_roboflow_api_key
SEGMIND_API_KEY=your_segmind_api_key
```

### 5️⃣ Run the FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
👉 `http://localhost:8000`

Interactive API docs (Swagger UI):
👉 `http://localhost:8000/docs`

---

## 📡 API Endpoints

### 🔹 Health Check

```http
GET /
```

Returns a simple health status.

---

### 🔹 Detect Clothing

```http
POST /detect-clothing
```

**Request:**

* `file`: Image file

**Response:**

* Detected clothing items
* Available regions (`upper` / `lower`)

---

### 🔹 Virtual Try-On

```http
POST /virtual-tryon
```

**Request:**

* `file`: Image file
* `prompt`: Text prompt describing desired clothing (e.g., `"red leather jacket"`)
* `region`: `"upper"` or `"lower"`

**Response:**

* Original image (base64)
* Masked clothing region (base64)
* Generated try-on result (base64)

---

## 📦 Dependencies

Main libraries and models used:

* **FastAPI** – Web API framework
* **Torch** – Deep learning framework
* **Ultralytics SAM** – Segment Anything model
* **Roboflow Inference SDK** – Clothing detection
* **Segmind API** – Inpainting for virtual try-on
* **OpenCV / Pillow / NumPy** – Image processing
* **Uvicorn** – ASGI server

See `requirements.txt` for full details.

---

## 🖼️ Example Workflow

1. Upload an image of a person.
2. API detects **clothing regions** (upper/lower).
3. User selects a region and provides a **text prompt** (e.g., "blue denim jacket").
4. System segments the selected region and **generates try-on output** using Segmind.

---

## 🌐 Frontend

The **frontend** for this project was built using [Lovable](https://lovable.dev) and connects directly with this backend API.
🔗 [Frontend Repository](https://github.com/your-username/virtual-tryon-frontend) (replace with your actual repo link)

---

## 📜 License

MIT License © 2025 \[Your Name]

---

## 🙌 Acknowledgements

* [Roboflow](https://roboflow.com/) – Clothing detection
* [Meta’s SAM](https://segment-anything.com/) – Image segmentation
* [Segmind](https://segmind.com/) – Inpainting
* [FastAPI](https://fastapi.tiangolo.com/) – API framework
