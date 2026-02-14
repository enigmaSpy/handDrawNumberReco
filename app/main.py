from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch

app = FastAPI()

# CORS (ważne!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = torch.load("model.pt")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        output = model(img)
        digit = int(torch.argmax(output))

    return {"digit": digit}