from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import cv2
import tensorflow as tf
import os
from scipy import ndimage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("debug", exist_ok=True)

try:
    model = tf.keras.models.load_model("model/mnist_better.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model not found: {e}")
    print("Run train_model.py first to create the model")
    model = None

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

def preprocess_for_mnist(img):
    """Preprocessuj obraz tak jak MNIST - centrowanie i skalowanie"""
 
    cv2.imwrite("debug/01_original.png", img)
    
    img = 255 - img
    cv2.imwrite("debug/02_inverted.png", img)
    
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite("debug/03_threshold.png", img)
    
    coords = cv2.findNonZero(img)
    
    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2*padding)
    h = min(img.shape[0] - y, h + 2*padding)
    
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite("debug/04_cropped.png", cropped)
    
    rows, cols = cropped.shape
    
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
    
    cropped = cv2.resize(cropped, (cols, rows), interpolation=cv2.INTER_AREA)
    
    result = np.zeros((28, 28), dtype=np.uint8)
    
    col_padding = (28 - cols) // 2
    row_padding = (28 - rows) // 2
    
    result[row_padding:row_padding+rows, col_padding:col_padding+cols] = cropped
    
    cv2.imwrite("debug/05_centered.png", result)
    
    result = result.astype(np.float32) / 255.0
    
    cv2.imwrite("debug/06_final_to_model.png", (result * 255).astype(np.uint8))
    
    return result

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}
    
    try:
        image_bytes = await file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        
        processed = preprocess_for_mnist(img)
        
        img_input = processed.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img_input, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        print(f"Predicted: {digit} with {confidence*100:.1f}% confidence")
        print(f"All probabilities: {prediction[0]}")
        
        return {
            "digit": digit,
            "confidence": confidence,
            "probabilities": prediction[0].tolist()
        }
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}