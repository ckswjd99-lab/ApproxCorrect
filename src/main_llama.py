import torch
# Import the specific model class LlamaForCausalLM
from transformers import AutoTokenizer, LlamaForCausalLM

# Login to Hugging Face Hub is required.
# Run `huggingface-cli login` in your terminal and enter your token.

# Model ID for Llama 3.2 1B
model_id = "meta-llama/Llama-3.2-1B"

# 1. Load tokenizer and model
# The 1B model is small enough to run on a typical GPU.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load the model using the specific LlamaForCausalLM class
model = LlamaForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,  # Use 16-bit floating point to reduce memory usage
    device_map="auto"
)

# 2. Set the prompt (the starting text for generation)
prompt = "Llama 3.2 is a new model from Meta AI that is"

# 3. Tokenize the prompt into the model's input format
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 4. Generate text
# max_new_tokens: The maximum number of new tokens to generate.
# temperature: Controls the randomness of the output. Lower values make it more deterministic.
output = model.generate(
    **inputs,
    max_new_tokens=60,
    temperature=0.6,
    eos_token_id=tokenizer.eos_token_id
)

# 5. Decode the generated tokens and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)