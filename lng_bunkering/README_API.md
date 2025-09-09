# LNG Bunkering Application REST API

A FastAPI‑based REST service for LNG bunkering calculations, **without any database dependency**.  
It uses ship/tank CSV data files for interpolation and corrections, and returns results in JSON.

---

## ✨ Features

- **Bunkering Calculations** → Corrected tank levels, volumes, loaded quantities, and energy values.
- **Ship & Tank Data** → Endpoints to explore available vessels and their tank configurations.
- **CSV‑Driven** → All calculations sourced from pre‑supplied CSV data files.
- **Interactive Docs** → Swagger UI (`/docs`) and ReDoc (`/redoc`) auto‑generated.
- **Lightweight Deployment** → No CouchDB; portable via Docker or direct `uvicorn`.

---

## 📦 Installation (Local)

1. **Install dependencies**
   ```bash
   pip install --no-cache-dir -r requirements_api.txt

Run the API server
python api.py

Access the API
Base URL → http://localhost:8000

Swagger Docs → http://localhost:8000/docs

ReDoc → http://localhost:8000/redoc

Docker Deployment
Dockerfile:
FROM python:3.9.23-slim

WORKDIR /lng_bunker_api

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements_api.txt

COPY api.py ./
COPY DATA ./DATA

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


Build & Run:
docker build -t lng-bunker-api .
docker run -p 8000:8000 lng-bunker-api


API Endpoints
1. Ship Information
GET /ships → List all available ships.

GET /ships/{ship_id} → Get tank details and CSV status for a ship.

2. LNG Bunkering Calculations
POST /bunkering/calculate → Perform calculation from supplied measurements.

Example Request:
{
  "ship_id": "MOUNT TOURMALINE",
  "opening_tank1": {
    "level": 1000.0,
    "vapor_temp": -150.0,
    "liquid_temp": -160.0,
    "pressure": 0.1
  },
  "closing_tank1": {
    "level": 1500.0,
    "vapor_temp": -149.0,
    "liquid_temp": -159.0,
    "pressure": 0.15
  },
  "opening_trim": 0.5,
  "opening_list": 1.2,
  "closing_trim": 0.3,
  "closing_list": 0.8,
  "opening_time": "12/01/2024 08:00",
  "closing_time": "12/01/2024 16:00",
  "density": 0.425,
  "bdn_quantity": 500.0,
  "bog": 200.0,
  "gross_energy": 2500.0,
  "unreckoned_qty": 10.0,
  "net_energy": 2400.0
}

Usage Examples
Python

import requests

BASE_URL = "http://localhost:8000"
ships = requests.get(f"{BASE_URL}/ships").json()["ships"]
print("Ships:", ships)

payload = {...}  # See example above
res = requests.post(f"{BASE_URL}/bunkering/calculate", json=payload).json()
print("Loaded qty:", res["loaded_quantity"])


cURL
curl -X GET "http://localhost:8000/ships"

curl -X POST "http://localhost:8000/bunkering/calculate" \
  -H "Content-Type: application/json" \
  -d @payload.json


JavaScript

fetch('http://localhost:8000/ships')
  .then(r => r.json())
  .then(d => console.log(d.ships));


⚠ Error Handling
400 → Invalid input or missing fields

404 → Ship ID or CSV file not found

500 → Internal calculation error

Error format:
{ "detail": "Invalid ship ID: UNKNOWN_SHIP" }

📂 Project Structure
.
├── api.py                 # FastAPI app (no CouchDB)
├── DATA/                  # Tank/ship CSV files
├── requirements_api.txt   # Dependencies
├── test_bunkering_api.py  # Optional test script
└── README.md              # This file


Production Tips
Use gunicorn with uvicorn workers:
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker

Bind‑mount DATA/ if you want to refresh CSVs without rebuilding the image.

Apply proper CORS/auth in production.