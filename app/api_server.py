from fastapi import FastAPI
from typing import Any
from server import verify_disaster_text, verify_disaster_batch

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "api server is running"
    }

@app.post("/verify")
def verify(data: dict[str, Any]) -> dict[str, Any]:
    return verify_disaster_text(data["text"])


@app.post("/verify_batch")
def verify_batch(data: dict[str, Any]) -> dict[str, Any]:
    return verify_disaster_batch(data["texts"])
