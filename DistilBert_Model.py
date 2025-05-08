'''
This script is to finetune the DistilBert model on the csv from 50testforeachbehavior.py
'''


import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Load your CSV
df = pd.read_csv("behavior_eval_log50_150.csv")

# Drop rows with missing labels
df = df.dropna(subset=["actual_behavior"])

# Combine teacher_input and model_output as the input text
df["text"] = "Teacher: " + df["teacher_input"] + " Student: " + df["model_output"]

# Encode labels
label_map = {"Focused": 0, "Confused": 1, "Distracted": 2}
df["label"] = df["actual_behavior"].map(label_map)

# Split into train and validation
train_df, val_df = train_test_split(df[["text", "label"]], test_size=0.2, random_state=42)
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])


from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./distilbert_behavior_classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()
trainer.save_model("./final_distilbert_behavior_classifier")
tokenizer.save_pretrained("./final_distilbert_behavior_classifier")