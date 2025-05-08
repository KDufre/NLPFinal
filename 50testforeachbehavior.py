'''
This script is for testing 50 different teacher inputs for the finetuned tinyLlama model to get the outputs.
It is then saved to a csv file with behavior, teacher input, model output, and (empty) actual behavior.
This is for testing the finetuned model and for training Distilbert.

'''

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
import csv

# === Load Fine-Tuned TinyLlama Model ===
model_dir = "./tinyllama_Complete_3Gen"
ft_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
ft_tokenizer = LlamaTokenizer.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_model.to(device)

# === Generate a Response from the Model ===
def generate_response(prompt):
    model_input = ft_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = ft_model.generate(
            **model_input,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.6,
            top_k=50,
            top_p=0.8,
            repetition_penalty=1.0,
            num_return_sequences=1
        )
        generated_text = ft_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# === Extract only the student response portion ===
def extract_student_response(full_response):
    if "### Student Response:" in full_response:
        reply = full_response.split("### Student Response:")[1].split("### Teacher Input:")[0].strip()
    else:
        reply = full_response.strip()
    return reply.split('#')[0].strip()

# === 50 Diverse Prompts ===
teacher_inputs = [
    "What is two plus two?",
    "What causes seasons to change?",
    "What is the role of the mitochondria in cells?",
    "How does a bill become a law?",
    "What is the function of red blood cells?",
    "What are tectonic plates?",
    "How does photosynthesis produce glucose?",
    "Why is the moon important for Earth?",
    "How does gravity affect tides?",
    "What is the function of the circulatory system?",
    "What are Newton's three laws of motion?",
    "How do plants absorb water?",
    "What causes lightning during storms?",
    "What is the purpose of the immune system?",
    "How do earthquakes occur?",
    "How does digestion work in the human body?",
    "What is the greenhouse effect?",
    "How does the nervous system function?",
    "What is the water table?",
    "Why are bees important to ecosystems?",
    "How does deforestation affect the environment?",
    "What is the process of cellular respiration?",
    "What is DNA and its function?",
    "How do vaccines work?",
    "How does the human brain process information?",
    "What is the function of the skeletal system?",
    "What are the stages of the rock cycle?",
    "How do volcanoes form?",
    "What is the function of the heart valves?",
    "Why is biodiversity important?",
    "How do magnets work?",
    "What causes ocean currents?",
    "What is the carbon cycle?",
    "How do animals adapt to their environments?",
    "How is energy transferred in a food web?",
    "What causes the phases of the moon?",
    "How do sound waves travel through air?",
    "How does climate change impact weather patterns?",
    "What is the role of the ozone layer?",
    "How do plants make oxygen?",
    "What is the process of metamorphosis in insects?",
    "How do fish breathe underwater?",
    "What are the properties of matter?",
    "What causes wind to form?",
    "How does a rainbow form?",
    "What is the purpose of roots in a plant?",
    "What is the difference between mass and weight?",
    "How do simple machines make work easier?",
    "What is static electricity?",
    "How do seeds grow into plants?",
    "Why do we need sleep?",
    "What are the functions of the liver?",
    "What is the role of chlorophyll in plants?",
    "How does pollution affect marine life?",
    "What is the structure of an atom?",
    "How do birds migrate?",
    "What causes eclipses?",
    "What is a habitat?",
    "How do camels survive in the desert?",
    "What causes a tsunami?",
    "What are fossils and how are they formed?",
    "What is the purpose of camouflage in animals?",
    "How do spiders make webs?",
    "How does the digestive system process food?",
    "What causes metal to rust?",
    "What is a black hole?",
    "How does the eye perceive light?",
    "What is the function of white blood cells?",
    "Why is the sky blue?",
    "How does photosynthesis help the food chain?",
    "What are producers and consumers in an ecosystem?",
    "What causes the northern lights?",
    "What is a lunar eclipse?",
    "How does sound differ from light?",
    "What causes magnets to attract?",
    "What is the function of the pancreas?",
    "Why is oxygen important for survival?",
    "How do plants reproduce asexually?",
    "What are the layers of the ocean?",
    "How does pressure affect weather systems?",
    "What is a hurricane and how does it form?",
    "How does a battery store energy?",
    "What is the circulatory system's main role?",
    "How do bacteria help in digestion?",
    "What are the types of clouds?",
    "How do vaccines protect us?",
    "What is the life span of a butterfly?",
    "How does the skeletal system support the body?",
    "What are the different states of matter?",
    "What causes seasons to change on Earth?",
    "How does water travel in a plant?",
    "What is the function of the brainstem?",
    "Why is carbon important in biology?",
    "How do geysers erupt?",
    "What causes droughts?",
    "What is the purpose of stomata in leaves?",
    "How do dolphins communicate?",
    "What is erosion and how does it happen?",
    "How do astronauts survive in space?",
    "What is the pH scale used for?",
    "How do amphibians breathe?",
    "What causes iron to oxidize?",
    "How does friction affect motion?",
    "What is an ecosystem made of?",
    "How does the human body regulate temperature?"
]

# === Behavior Modes to Test ===
behaviors = ["Distracted", "Focused", "Confused"]
results = []

# === Run Evaluation ===
print("\n================ TEST OUTPUT ================\n")
for behavior in behaviors:
    print(f"\n--- BEHAVIOR: {behavior} ---\n")
    for i, teacher_input in enumerate(teacher_inputs, 1):
        clean_input = teacher_input.lower().rstrip("?")
        prompt = f"### Behavior: {behavior}\n### Teacher Input: {clean_input}\n### Student Response: "
        raw_response = generate_response(prompt)
        student_response = extract_student_response(raw_response)
        print(f"{i}. Teacher: {teacher_input}\n   Model: {student_response}\n")
        results.append({
            "prompted_behavior": behavior,
            "teacher_input": teacher_input,
            "model_output": student_response,
            "actual_behavior": ""  # You will fill this in manually
        })

# === Export Results to CSV ===
csv_filename = "behavior_eval_log50_150.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["prompted_behavior", "teacher_input", "model_output", "actual_behavior"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ CSV export complete: {csv_filename}")
print("📝 You can now open the file and manually fill in the 'actual_behavior' column.")