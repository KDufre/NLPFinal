import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the model directory
model_dir = "./tinyllama_Complete_3Gen"

# Load the model and tokenizer
ft_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
ft_tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_model.to(device)

# Function to generate responses
def generate_response(prompt):
    # Tokenize the input and move to the same device as the model
    model_input = ft_tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the response
    with torch.no_grad():
        generated_ids = ft_model.generate(
            **model_input,
            max_new_tokens=20,          # Adjust this based on desired response length
            do_sample=True,             # Enable sampling for faster generation
            temperature=1,            # Adjust temperature for a balance between randomness and coherence
            top_p=0.7,                  # Nucleus sampling to improve generation quality
            repetition_penalty=1.0,     # Penalty to reduce repetitive text
            num_return_sequences=1      # Ensure only one output is generated
        )
        generated_text = ft_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# Chat loop
print("Start chatting with the model (type 'goodbye' to exit):")
behavior = input("Enter the behavior (Distracted, Confused, Focused): ").strip()
while True:
    user_input = input("You: ")
    if user_input.lower() == "goodbye":
        print("Goodbye!")
        break
    eval_prompt = f"### Behavior: {behavior}\n### Teacher Input: {user_input}\n### Student Response: "
    response = generate_response(eval_prompt)
    
    # Extract only the student's response
    if "### Student Response:" in response:
        student_response = response.split("### Student Response:")[1].split("### Teacher Input:")[0].strip()
    else:
        student_response = response.strip()
    
    student_response = student_response.split('#')[0]
    
    print(f"Model: {student_response}")
