import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle

app = FastAPI()

# تحميل نموذج MARBERT للهجة الجزائرية
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
model = AutoModel.from_pretrained("UBC-NLP/MARBERT").to(device)

# نموذج تصنيف بسيط للاختبار (سيتم تحسينه لاحقاً)
# هذا نموذج افتراضي، ننصح بتدريب نموذج حقيقي وتضمينه
class SimpleClassifier:
    def predict(self, embedding):
        # نموذج مبني على قواعد بسيطة للاختبار
        # سيتم استبداله بنموذج حقيقي لاحقاً
        return 2  # محايد

classifier = SimpleClassifier()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Algerian Dialect Sentiment Analysis API (Test Version)"}

@app.post("/predict")
def predict(request: TextRequest):
    try:
        # استخراج embedding من النص
        inputs = tokenizer(request.text, return_tensors="pt", 
                          max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # تصنيف النص
        prediction = classifier.predict(embedding)
        
        # خريطة المشاعر
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
    uvicorn.run(app, host="0.0.0.0", port=10000)
