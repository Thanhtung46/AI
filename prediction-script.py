from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class PsychologyPredictor:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.analyzer = StudentHistoryAnalyzer()

    def predict(self, text, student_id):
        # Chuẩn bị input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
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

# Sử dụng mô hình
predictor = PsychologyPredictor('./final_model')

# Ví dụ sử dụng
text = "Lan ngủ trong tiết toán"
student_id = "HS001"
result = predictor.predict(text, student_id)
print(f"Kết quả dự đoán: {result['prediction']}")
print(f"Phân tích xu hướng: {result['trend_analysis']}")
