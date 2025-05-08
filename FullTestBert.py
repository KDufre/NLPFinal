'''
This is to test 50 different teacher questions on each behavior on the finetuned DistilBert model.
Given behavior (Focused, Distracted, Confused)
Given teacher input (Usually a school related question)
Output: a student output from tinyLlama + behavior from DistilBert
'''


import torch
import csv
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.nn.functional import softmax

# === Load TinyLlama Generator ===
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

# === Classifier label mapping (must match training) ===
labels = ["Focused", "Confused", "Distracted"]

# === Diverse 20 Teacher Inputs ===
teacher_inputs = [
    "Why do leaves change color in the fall?",
    "What years did World War 2 happen?",
    "What does the heart do?",
    "Who was Steve Jobs and why is he important?",
    "What is the difference between a simile and a metaphor?",
    "Why do some objects float while others sink?",
    "How does the moon affect animal behavior?",
    "What happens during a chemical reaction?",
    "How is a bar graph different from a line graph?",
    "What are the main sides of the American Civil War?",
    "What is the function of a preposition in a sentence?",
    "Name an element from the periodic table?",
    "What is the importance of reading comprehension?",
    "What is the difference between weather and climate?",
    "How does multiplication relate to addition?",
    "What are renewable and nonrenewable resources?",
    "What speech is Martin Luther King Jr. remembered for today?",
    "How do you calculate the area of a rectangle?",
    "What are the planets in our solar system?",
    "What makes a good persuasive argument?"
]

# === Behaviors to prompt ===
behaviors = ["Focused", "Confused", "Distracted"]

# === Extract student response from TinyLlama output ===
def extract_student_response(full_response):
    if "### Student Response:" in full_response:
        return full_response.split("### Student Response:")[1].split("###")[0].strip()
    return full_response.strip().split("#")[0].strip()

# === Generate response from TinyLlama ===
def generate_response(teacher_input, behavior):
    prompt = f"### Behavior: {behavior}\n### Teacher Input: {teacher_input.lower().rstrip('?')}\n### Student Response:"
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = llama_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.6,
            top_k=50,
            top_p=0.8,
            repetition_penalty=1.0
        )
    full_response = llama_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return extract_student_response(full_response)

# === Classify response using DistilBERT ===
def classify_behavior(teacher_input, model_output):
    text = f"Teacher: {teacher_input} Student: {model_output}"
    inputs = clf_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = clf_model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        label = labels[probs.argmax().item()]
        conf = probs.max().item()
    return label, conf

# === Run evaluation ===
results = []

print("\n===== Behavior Classification Evaluation =====\n")

for behavior in behaviors:
    print(f"\n--- Prompted Behavior: {behavior} ---\n")
    for i, question in enumerate(teacher_inputs, 1):
        model_output = generate_response(question, behavior)
        predicted_behavior, confidence = classify_behavior(question, model_output)

        print(f"{i}. Q: {question}")
        print(f"   📝 Model Output: {model_output}")
        print(f"   🧠 Predicted: {predicted_behavior} (Confidence: {confidence:.2f})\n")

        results.append({
            "prompted_behavior": behavior,
            "teacher_input": question,
            "model_output": model_output,
            "predicted_behavior": predicted_behavior,
            "confidence": round(confidence, 2)
        })

# === Save results to CSV ===
csv_filename = "behavior_eval_20x3.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Results saved to {csv_filename}")