import os
import torch
from transformers import AutoModel, AutoTokenizer
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gdown

app = FastAPI()

# معرفات الملفات من Google Drive (عدلها حسب ملفاتك)
MODEL_ID = "YOUR_MODEL_FILE_ID"
SCALER_ID = "YOUR_SCALER_FILE_ID"

# تحميل النماذج من Google Drive
if not os.path.exists("model.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", "model.pkl", quiet=False)
if not os.path.exists("scaler.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={SCALER_ID}", "scaler.pkl", quiet=False)

classifier = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# تحميل نموذج MARBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
model = AutoModel.from_pretrained("UBC-NLP/MARBERT").to(device)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Algerian Dialect Sentiment Analysis API"}

@app.post("/predict")
def predict(request: TextRequest):
    try:
        inputs = tokenizer(request.text, return_tensors="pt", 
                          max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embedding_scaled = scaler.transform(embedding)
        prediction = classifier.predict(embedding_scaled)[0]
        
        sentiment_map = {0: "negative", 1: "positive", 2: "neutral"}
        
        return {
            "text": request.text,
            "sentiment": sentiment_map[prediction],
            "code": int(prediction),
            "arabic_sentiment": {0: "سلبي", 1: "إيجابي", 2: "محايد"}[prediction]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
