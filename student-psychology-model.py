import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pytorch_lightning as pl
from sklearn.metrics import classification_report

# Thêm các import còn thiếu
from PIL import Image
import pytesseract
import speech_recognition as sr

class StudentPsychologyModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        # Tải mô hình BERT đã được huấn luyện trước và bộ tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels, max_length=512):
        # Tiền xử lý dữ liệu văn bản và chuẩn bị nhãn cho mô hình BERT
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })

    def train_model(self, train_texts, train_labels, val_texts, val_labels, output_dir='./final_model'):
        # Tiền xử lý dữ liệu huấn luyện và kiểm tra
        train_data = self.preprocess_data(train_texts, train_labels)
        val_data = self.preprocess_data(val_texts, val_labels)

        # Định nghĩa các tham số huấn luyện
        training_args = TrainingArguments(
            output_dir=output_dir,  # Đường dẫn để lưu mô hình
            evaluation_strategy="epoch",  # Đánh giá mô hình sau mỗi epoch
            learning_rate=2e-5,  # Tốc độ học
            per_device_train_batch_size=16,  # Kích thước batch khi huấn luyện
            per_device_eval_batch_size=16,  # Kích thước batch khi kiểm tra
            num_train_epochs=3,  # Số epoch huấn luyện
            weight_decay=0.01,  # Giảm trọng số (weight decay)
            save_steps=10_000,  # Số bước huấn luyện để lưu mô hình
            save_total_limit=2,  # Giới hạn số lượng mô hình được lưu
            logging_dir='./logs',  # Đường dẫn lưu log
            logging_steps=500,  # Số bước để ghi log
            load_best_model_at_end=True,  # Tải mô hình tốt nhất khi huấn luyện kết thúc
            metric_for_best_model="accuracy"  # Đánh giá mô hình tốt nhất dựa trên độ chính xác
        )

        # Khởi tạo Trainer để huấn luyện mô hình
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=self.compute_metrics  # Hàm tính toán các chỉ số đánh giá
        )

        # Huấn luyện mô hình
        trainer.train()

        # Lưu mô hình và tokenizer đã huấn luyện
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def compute_metrics(self, p):
        # Tính toán độ chính xác và các chỉ số đánh giá khác
        preds = p.predictions.argmax(axis=1)
        labels = p.label_ids
        report = classification_report(labels, preds, output_dict=True)
        accuracy = report['accuracy']
        return {'accuracy': accuracy}

    def predict(self, text):
        # Tiền xử lý văn bản và thực hiện dự đoán với mô hình đã huấn luyện
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()  # Lấy lớp phân loại có xác suất cao nhất
        return predicted_class

    def _generate_recommendations(self, prediction, last_text):
        # Tạo ra các gợi ý dựa trên dự đoán và văn bản nhập vào
        recommendations = []
        
        if prediction == 0:
            recommendations.append("Học sinh bình thường, cần khuyến khích động viên.")
        elif prediction == 1:
            recommendations.append("Học sinh có dấu hiệu căng thẳng. Nên trao đổi trực tiếp và tạo không gian để chia sẻ.")
        elif prediction == 2:
            recommendations.append("Học sinh thất vọng, cần đề xuất hỗ trợ tâm lý chuyên sâu hoặc kết nối với chuyên gia.")
        
        # Gợi ý dựa trên các từ khóa trong văn bản nhập vào
        if "khóc" in last_text:
            recommendations.append("Học sinh có dấu hiệu cảm xúc tiêu cực. Nên hỏi thăm và động viên.")
        elif "ngủ" in last_text:
            recommendations.append("Học sinh có dấu hiệu mệt mỏi. Hỏi về lịch sinh hoạt và tình trạng sức khỏe.")
        
        return "\n".join(recommendations)

    def extract_text_from_image(self, image_path):
        # Quét văn bản từ hình ảnh bằng OCR
        text = pytesseract.image_to_string(Image.open(image_path), lang='vie')
        return text

    def transcribe_speech(self):
        # Nhận diện giọng nói và chuyển thành văn bản
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="vi-VN")
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return ""

# Ví dụ huấn luyện và đánh giá mô hình
if __name__ == "__main__":
    # Dữ liệu ví dụ (texts và labels cần được cung cấp)
    texts = ["Học sinh có dấu hiệu căng thẳng", "Học sinh hoạt động bình thường"]
    labels = [1, 0]  # 0: bình thường, 1: căng thẳng, 2: thất vọng

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

    # Khởi tạo và huấn luyện mô hình
    model = StudentPsychologyModel()
    model.train_model(train_texts, train_labels, val_texts, val_labels)

    # Dự đoán và tạo gợi ý
    sample_text = "Học sinh có dấu hiệu căng thẳng"
    prediction = model.predict(sample_text)
    recommendations = model._generate_recommendations(prediction, sample_text)
    print(f"Dự đoán: {prediction}")
    print(f"Gợi ý: {recommendations}")
