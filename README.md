# NLPFinal – Behavior Classification with TinyLlama + DistilBERT

This project explores behavioral classification of student responses using a two-model NLP pipeline:

- **TinyLlama (Fine-Tuned):** Generates realistic student responses to teacher questions based on three behavioral prompts — Focused, Confused, and Distracted.
- **DistilBERT (Fine-Tuned):** Classifies generated responses into one of the three behaviors based on content and tone.

---

## 🎯 Project Goal

The goal of this project is to simulate student behaviors in classroom settings and automatically classify the quality or intent of AI-generated answers. This can be used to study engagement patterns in educational tools, classroom simulations, and AI-assisted tutoring systems.

---


## 💾 Model Downloads (Google Drive)

Due to GitHub's file size limits, the models must be downloaded from these public links:

- 🔹 [TinyLlama (Fine-Tuned)](https://drive.google.com/file/d/12BMCA5gZgEd7QsnhKKSy5t6io6fbArlB/view?usp=share_link)
- 🔹 [DistilBERT Classifier](https://drive.google.com/file/d/1TG7aCZho4_-0Pfi6_PnZS8YIib2xV0L5/view?usp=sharing)
- 🔹 [Backup Model Folder](https://drive.google.com/file/d/1FtmR42ZH5EgDJyn18vFN43Pd9PPYd0kp/view?usp=share_link)

> ⚠️ After downloading, extract the model folders and place them in the project directory.

---

## 🚀 Quick Start

### 1. Set Up Environment

```bash
pip install transformers datasets scikit-learn torch

```



How to Run the Classifier on Your Input
 -  python TestClassifier.py
You'll be prompted to enter a teacher question and student response. The script will predict the behavior class (Focused / Confused / Distracted).

How to Run Full Evaluation
To run a full test across all behaviors and automatically generate responses:

 -  python FullTestBert.py
Results will be saved to behavior_eval_20x3.csv.

📊 Dataset Summary

300 labeled examples (100 per behavior)
Teacher + student text pairs formatted as:
"Teacher: <input> Student: <response>"
🧠 Model Details

TinyLlama
Fine-tuned with behavior-conditioning prompts (e.g., ### Behavior: Distracted)
Generates realistic short-form student answers
DistilBERT Classifier
Fine-tuned on the labeled dataset with 3 classes
Uses DistilBertForSequenceClassification
🔍 Research Paper

See ShortResearchPaper.docx for a full write-up of background, methodology, and evaluation results.
