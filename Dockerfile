FROM python:3.11-slim

WORKDIR /app

# تثبيت dependencies النظام
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات
COPY . .

# تشغيل الخادم
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
