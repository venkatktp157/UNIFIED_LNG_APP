from fastapi import FastAPI
from lng_bunkering.api import app as lng_app
from mncal.app import app as mncal_app

app = FastAPI(title="Unified Selector App")

# Mount each app under a path
app.mount("/lng", lng_app)
app.mount("/mncal", mncal_app)

@app.get("/")
def root():
    return {
        "message": "Choose your app",
        "options": {
            "LNG Bunkering": "/lng",
            "MNCal": "/mncal"
        }
    }
