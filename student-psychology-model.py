from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from huggingface_hub import HfApi, HfFolder
import pandas as pd
import numpy as np
from datetime import datetime

# Sử dụng mô hình PhoBERT cho tiếng Việt
MODEL_NAME = "vinai/phobert-base"

# Tải dữ liệu từ tệp CSV
dataset = load_dataset('csv', data_files={
    'train': 'train.csv', 
    'validation': 'valid.csv'
}, delimiter=';')  # Thêm delimiter để xử lý đúng định dạng CSV

# Tokenizer và model phù hợp với tiếng Việt
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(set(dataset['train']['label']))  # Số nhãn dựa trên dữ liệu thực tế
)

# Hàm xử lý dữ liệu
def preprocess_function(examples):
    # Xử lý text tiếng Việt
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256  # Tăng độ dài tối đa cho văn bản tiếng Việt
    )

# Xử lý dữ liệu
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Thêm trường timestamp để theo dõi thời gian
def add_timestamp(example):
    example['timestamp'] = datetime.now().isoformat()
    return example

tokenized_datasets = tokenized_datasets.map(add_timestamp)

# Thiết lập tham số huấn luyện
training_args = TrainingArguments(
    output_dir='./model_results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,  # Tăng số epoch để học tốt hơn
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,  # Lưu mô hình tốt nhất
    metric_for_best_model='f1',  # Sử dụng F1 làm metric chính
    remove_unused_columns=False
)

# Định nghĩa hàm tính metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Thiết lập Trainer với compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics
)

# Class để lưu trữ và phân tích lịch sử
class StudentHistoryAnalyzer:
    def __init__(self):
        self.history = {}
    
    def add_observation(self, student_id, observation, prediction):
        if student_id not in self.history:
            self.history[student_id] = []
        self.history[student_id].append({
            'timestamp': datetime.now(),
            'observation': observation,
            'prediction': prediction
        })
    
    def analyze_progress(self, student_id):
        if student_id not in self.history:
            return "Chưa có dữ liệu về học sinh này"
        
        observations = self.history[student_id]
        if len(observations) < 2:
            return "Cần thêm dữ liệu để phân tích xu hướng"
            
        # Phân tích xu hướng và đưa ra đề xuất
        recent_issues = [obs['prediction'] for obs in observations[-3:]]
        return self._generate_recommendations(recent_issues)
    
    def _generate_recommendations(self, recent_issues):
        # Logic đưa ra đề xuất dựa trên lịch sử quan sát
        # Có thể mở rộng phần này với các quy tắc phức tạp hơn
        recommendations = []
        if len(set(recent_issues)) == 1:
            recommendations.append("Vấn đề vẫn đang tiếp diễn, cần tăng cường biện pháp can thiệp")
        elif len(recent_issues) >= 3 and recent_issues[-1] != recent_issues[0]:
            recommendations.append("Có sự thay đổi trong hành vi, cần điều chỉnh phương pháp phù hợp")
        return "\n".join(recommendations)

# Huấn luyện mô hình
trainer.train()

# Lưu mô hình
model_path = './final_model'
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# Đăng tải lên Hugging Face Hub
api = HfApi()
api.upload_folder(
    folder_path=model_path,
    repo_id="tchun3879/student-psychology-vietnam",  # Cập nhật repo_id
    repo_type="model"
)

# Hàm dự đoán với phân tích lịch sử
def predict_and_track(text, student_id, analyzer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    
    # Thêm vào lịch sử
    analyzer.add_observation(student_id, text, prediction)
    
    # Lấy phân tích xu hướng
    trend_analysis = analyzer.analyze_progress(student_id)
    
    return {
        'current_prediction': prediction,
        'trend_analysis': trend_analysis
    }

