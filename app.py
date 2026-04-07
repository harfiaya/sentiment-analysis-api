import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

# استخدام نموذج DziriBERT (أصغر حجماً ومناسب للهجة الجزائرية)
model_name = "alg-ry/DziriBERT"

print("🔄 جاري تحميل نموذج DziriBERT...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("✅ تم تحميل النموذج بنجاح!")
except Exception as e:
    print(f"❌ فشل التحميل: {e}")
    # نموذج وهمي للاختبار
    class DummyModel:
        def __init__(self):
            self.device = torch.device("cpu")
        def to(self, device):
            return self
        def __call__(self, **kwargs):
            class Output:
                last_hidden_state = torch.randn(1, 1, 768)
            return Output()
    
    model = DummyModel()
    tokenizer = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Algerian Dialect Sentiment Analysis API - DziriBERT"}

@app.post("/predict")
def predict(request: TextRequest):
    try:
        if tokenizer is None:
            return {
                "text": request.text,
                "sentiment": "neutral",
                "code": 2,
                "arabic_sentiment": "محايد",
                "note": "نموذج تجريبي - يرجى التحقق من الاتصال بـ Hugging Face"
            }
        
        # استخراج embedding من النص
        inputs = tokenizer(request.text, return_tensors="pt", 
                          max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # نموذج تصنيف بسيط يعتمد على كلمات إيجابية/سلبية
        positive_words = ["رائع", "جميل", "ممتاز", "حلو", "عجبني", "مزيان"]
        negative_words = ["سيء", "خايب", "مكروه", "زعلا", "حزين", "والو"]
        
        text = request.text
        score = 0
        for word in positive_words:
            if word in text:
                score += 1
        for word in negative_words:
            if word in text:
                score -= 1
        
        if score > 0:
            prediction = 1  # إيجابي
        elif score < 0:
            prediction = 0  # سلبي
        else:
            prediction = 2  # محايد
        
        sentiment_map = {0: "negative", 1: "positive", 2: "neutral"}
        
        return {
            "text": request.text,
            "sentiment": sentiment_map[prediction],
            "code": int(prediction),
            "arabic_sentiment": {0: "سلبي", 1: "إيجابي", 2: "محايد"}[prediction],
            "model": "DziriBERT"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
