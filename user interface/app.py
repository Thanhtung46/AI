import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

# Khởi tạo Flask app
app = Flask(__name__)

# Tải mô hình đã huấn luyện
with open('final_model', 'rb') as f:
    model = pickle.load(f)

# Route trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Route nhận dữ liệu từ người dùng và dự đoán kết quả
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ người dùng (ví dụ, 4 tham số cho Iris dataset)
        data = request.form.to_dict()
        features = np.array([list(map(float, data.values()))]).reshape(1, -1)

        # Dự đoán với mô hình đã huấn luyện
        prediction = model.predict(features)
        
        # Trả về kết quả dưới dạng JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
