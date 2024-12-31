import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import os

# Đường dẫn dữ liệu
TRAIN_PATH = './AI/train.csv'
VALID_PATH = './AI/valid.csv'
OUTPUT_DIR = "final_model"

# 1. Load dữ liệu
def load_data(train_path, valid_path):
    train_data = pd.read_csv(train_path, sep=';')
    valid_data = pd.read_csv(valid_path, sep=';')
    return train_data, valid_data

# 2. Chuẩn bị dữ liệu cho Hugging Face Datasets
def prepare_dataset(data, tokenizer):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data['label'].tolist())  # Chuyển đổi labels thành số
    texts = data['text'].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels  # Đảm bảo rằng bạn truyền đúng labels
    })
    return dataset

# 3. Huấn luyện mô hình
def train_model(train_dataset, valid_dataset, model, tokenizer):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_strategy="steps",
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data, valid_data = load_data(TRAIN_PATH, VALID_PATH)
    train_dataset = prepare_dataset(train_data, tokenizer)
    valid_dataset = prepare_dataset(valid_data, tokenizer)

    # Khởi tạo mô hình
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(train_data['label'].unique()))

    # Huấn luyện mô hình
    train_model(train_dataset, valid_dataset, model, tokenizer)
