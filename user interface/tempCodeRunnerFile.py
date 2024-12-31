from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Tải mô hình và tokenizer đã lưu
MODEL_DIR = "final_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # Đặt mô hình ở chế độ đánh giá

# Route trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Route nhận dữ liệu từ người dùng và dự đoán kết quả
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ người dùng (ví dụ, mã học sinh và tình trạng)
        student_id = request.form['student-id']
        text = request.form['text']

        # Chuẩn bị dữ liệu cho mô hình
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Lấy kết quả dự đoán (logits -> index có giá trị cao nhất)
        prediction = outputs.logits.argmax(-1).item()

        # Phân tích xu hướng học sinh (Ví dụ đơn giản)
        trend_analysis = "Chưa đủ dữ liệu để phân tích."  # Bạn có thể viết logic phân tích xu hướng theo cách riêng

        # Trả về kết quả dưới dạng JSON
        return jsonify({
            'prediction': prediction,
            'trend_analysis': trend_analysis
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
