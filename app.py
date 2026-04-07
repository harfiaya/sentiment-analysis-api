
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
import joblib
import numpy as np

app = FastAPI(title="Sentiment Analysis API - Algerian Dialect")

# تحميل النموذج والتوكينايزر
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
model = AutoModel.from_pretrained("UBC-NLP/MARBERT").to(device)
classifier = joblib.load('sentiment_model_improved.pkl')
scaler = joblib.load('scaler.pkl')

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Algerian Dialect Sentiment Analysis API"}

@app.post("/predict")
def predict(request: TextRequest):
    try:
        # استخراج embedding
        inputs = tokenizer(request.text, return_tensors="pt", 
                          max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embedding_scaled = scaler.transform(embedding)
        
        # تصنيف
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
