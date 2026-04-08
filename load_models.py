from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Load the Qwen model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Save the model and tokenizer to the models directory
model.save_pretrained("./models/qwen_model")
tokenizer.save_pretrained("./models/qwen_tokenizer")

text = "Your thoughts for the world are: "
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
out_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0])

print("Model Saved")
print("Sample Model Output: " + out_text)