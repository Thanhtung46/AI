import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

class StudentHistoryAnalyzer:
    def __init__(self):
        self.history = {}

    def add_observation(self, student_id, text, prediction):
        if student_id not in self.history:
            self.history[student_id] = []
        self.history[student_id].append((text, prediction))

    def analyze_progress(self, student_id):
        if student_id not in self.history or len(self.history[student_id]) < 2:
            return "Chưa đủ dữ liệu để phân tích."

        recent_predictions = [pred for _, pred in self.history[student_id][-5:]]
        trend = "Không có thay đổi"
        
        if all(p == 0 for p in recent_predictions):
            trend = "Học sinh ổn định, có vẻ bình thường."
        elif all(p == 1 for p in recent_predictions):
            trend = "Học sinh đang gặp căng thẳng."
        elif all(p == 2 for p in recent_predictions):
            trend = "Học sinh có dấu hiệu thất vọng, cần hỗ trợ thêm."
        else:
            trend = "Có sự thay đổi trong trạng thái tâm lý học sinh."
        return trend

class PsychologyPredictor:
    def __init__(self, model_path):
        # Tải BertTokenizer và BertForSequenceClassification
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.analyzer = StudentHistoryAnalyzer()  # Đảm bảo lớp được khởi tạo ở đây

    def predict(self, text, student_id):
        # Chuẩn bị input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Lấy kết quả
        prediction = outputs.logits.argmax(-1).item()
        
        # Thêm vào lịch sử và phân tích
        self.analyzer.add_observation(student_id, text, prediction)
        trend_analysis = self.analyzer.analyze_progress(student_id)
        
        return {
            'prediction': prediction,
            'trend_analysis': trend_analysis
        }

    def save_history(self, filename):
        # Lưu lịch sử học sinh vào tệp JSON
        with open(filename, 'w') as f:
            json.dump(self.analyzer.history, f)

    def load_history(self, filename):
        # Tải lịch sử học sinh từ tệp JSON
        try:
            with open(filename, 'r') as f:
                self.analyzer.history = json.load(f)
        except FileNotFoundError:
            print(f"Không tìm thấy tệp lịch sử: {filename}")

# Sử dụng mô hình
predictor = PsychologyPredictor('./final_model')

# Ví dụ sử dụng
text = "Lan ngủ trong tiết toán"
student_id = "HS001"
result = predictor.predict(text, student_id)
print(f"Kết quả dự đoán: {result['prediction']}")
print(f"Phân tích xu hướng: {result['trend_analysis']}")

# Lưu lịch sử
predictor.save_history('student_history.json')

# Tải lịch sử
predictor.load_history('student_history.json')
