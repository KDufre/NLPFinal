'''
This is to test the finetuned DistilBert model.
Enter a behavior (Focused, Distracted, Confused)
Enter a teacher input (Usually a school related question)
Output: a student output from tinyLlama + behavior from DistilBert
'''


import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.nn.functional import softmax

# === Load TinyLlama Model for Student Response Generation ===
llama_dir = "./tinyllama_Complete_3Gen"
llama_model = AutoModelForCausalLM.from_pretrained(llama_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_dir)
llama_model.to("cuda" if torch.cuda.is_available() else "cpu")
device = llama_model.device

# === Load DistilBERT Classifier ===
clf_dir = "final_distilbert_behavior_classifier"
clf_model = DistilBertForSequenceClassification.from_pretrained(clf_dir)
clf_tokenizer = DistilBertTokenizerFast.from_pretrained(clf_dir)
clf_model.eval()

# === Labels for classification output ===
labels = ["Focused", "Confused", "Distracted"]

behavior = input("Behavior: ")

# === Generate a student response from TinyLlama ===
def generate_response(teacher_input):
    teacher_input = teacher_input.lower().rstrip("?")
    prompt = f"### Behavior: {behavior}\n### Teacher Input: {teacher_input}\n### Student Response:"
    model_input = llama_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = llama_model.generate(
            **model_input,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.8,
            repetition_penalty=1.0
        )
    full_response = llama_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_student_response(full_response)

# === Extract only the model's student response ===
def extract_student_response(full_response):
    if "### Student Response:" in full_response:
        return full_response.split("### Student Response:")[1].split("###")[0].strip()
    return full_response.strip().split("#")[0].strip()

# === Run both generation and classification ===
teacher_input = input("Teacher Input: ")

# Generate model output using TinyLlama
model_output = generate_response(teacher_input)
print(f"\nModel Output: {model_output}")

# Classify using DistilBERT
text = f"Teacher: {teacher_input} Student: {model_output}"
inputs = clf_tokenizer(text, return_tensors="pt", truncation=True)
with torch.no_grad():
    outputs = clf_model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    pred_label = labels[probs.argmax().item()]
    confidence = probs.max().item()

# Show classification result
print(f"\nPredicted Behavior: {pred_label} (Confidence: {confidence:.2f})")