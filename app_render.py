
import os
import torch
from transformers import AutoModel, AutoTokenizer
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import urllib.request
import gdown

app = FastAPI()

# تحميل النماذج من Google Drive عند بدء التشغيل
print("🔄 جاري تحميل النماذج من Google Drive...")

# استخدم هذا الرابط المباشر (بدل YOUR_FILE_ID بمعرف ملفك)
# ملاحظة: يجب أن يكون الملف عاماً (Anyone with link can download)

# رابط تحميل مباشر من Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
SCALER_URL = "https://drive.google.com/uc?export=download&id=YOUR_SCALER_ID"

# تحميل النموذج
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
urllib.request.urlretrieve(SCALER_URL, "scaler.pkl")

classifier = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

print("✅ تم تحميل النماذج بنجاح")

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
