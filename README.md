# 🛡️ Sentinel Backend

**Sentinel Backend** is a **FastAPI-based backend service** that powers the Sentinel security platform.

It provides APIs for:

- 🎭 **Deepfake Detection**
- 🔗 **Malicious URL Detection**

The backend processes requests from the frontend and uses detection logic or ML models to determine whether the input is **safe or malicious**.

---

## 🚀 Features

- ⚡ High-performance API using **FastAPI**
- 🎭 Deepfake detection endpoint
- 🔗 Malicious URL detection endpoint
- 📦 Lightweight and easy to deploy
- 🔌 REST API ready for frontend integration

---

## 🧑‍💻 Tech Stack

**Backend**

- Python
- FastAPI

**Libraries**

- Uvicorn
- Pydantic
- Requests / ML libraries (if used)

---

## 📁 Project Structure

```
Sentinel-backend
│
├── Fast.py        # Main FastAPI server
├── Url.py         # URL detection logic
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/vedant05bhombe-netizen/sentinel-backend.git
```

Move into the project folder

```bash
cd sentinel-backend
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Server

Start the FastAPI server using **uvicorn**

```bash
uvicorn Fast:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

---

## 📡 API Endpoints

### Deepfake Detection

```
POST /deepfake
```

**Description**

Upload media to check whether it is a **deepfake or real**.

---

### URL Detection

```
POST /url-check
```

**Description**

Checks whether a given URL is **malicious or safe**.

---

## 📘 API Documentation

FastAPI automatically generates documentation.

Swagger UI:

```
http://127.0.0.1:8000/docs
```

Redoc:

```
http://127.0.0.1:8000/redoc
```

---

## 🔮 Future Improvements

- 🧠 Improved ML detection models
- 🔐 Authentication for API usage
- 📊 Logging & monitoring
- 🚀 Docker deployment

---

## 👨‍💻 Author

**Vedant Bhombe**

3rd Year IT Student  
Java • Python • React • Spring Boot • PostgreSQL

---
