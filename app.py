import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

# تعيين متغير البيئة للعمل في وضع عدم الاتصال (لن يتم الاتصال بـ Hugging Face)
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # مؤقتاً، سنحاول الاتصال مرة أخرى

# قائمة بالمرايا المحتملة لنموذج MARBERT (اختر أول واحد يعمل)
model_sources = [
    "UBC-NLP/MARBERT",  # المصدر الأصلي
    "marefa-nlp/marbert",  # مرآة بديلة
    "bert-base-arabic",  # بديل أخف
]

model_source = None
tokenizer = None
model = None

for source in model_sources:
    try:
        print(f"محاولة تحميل النموذج من: {source}")
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModel.from_pretrained(source)
        model_source = source
        print(f"✅ تم تحميل النموذج بنجاح من: {source}")
        break
    except Exception as e:
        print(f"❌ فشل التحميل من {source}: {e}")

if model is None:
    # إذا فشل كل شيء، استخدم نموذجاً وهمياً للاختبار
    print("⚠️ استخدام نموذج وهمي للاختبار")
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
    return {"message": "Algerian Dialect Sentiment Analysis API (Test Version)"}

@app.post("/predict")
def predict(request: TextRequest):
    try:
        if tokenizer is None:
            # نموذج وهمي للاختبار
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
        
        # نموذج تصنيف بسيط للاختبار (يمكن استبداله بنموذج حقيقي لاحقاً)
        # هنا نستخدم قاعدة بسيطة: طول النص مؤشر على المشاعر (للتجربة فقط)
        text_length = len(request.text)
        if text_length < 20:
            prediction = 1  # إيجابي
        elif text_length > 50:
            prediction = 0  # سلبي
        else:
            prediction = 2  # محايد
        
        sentiment_map = {0: "negative", 1: "positive", 2: "neutral"}
        
        return {
            "text": request.text,
            "sentiment": sentiment_map[prediction],
            "code": int(prediction),
            "arabic_sentiment": {0: "سلبي", 1: "إيجابي", 2: "محايد"}[prediction],
            "model_used": model_source if model_source else "dummy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
