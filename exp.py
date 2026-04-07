from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Load the model from saved
loaded_model = AutoModelForCausalLM.from_pretrained("./models/qwen_model")
loaded_tokenizer = AutoTokenizer.from_pretrained("./models/qwen_tokenizer")

text = "The quick brown fox "
inputs = loaded_tokenizer(text, return_tensors="pt")

outputs = loaded_model(**inputs)
out_text = loaded_tokenizer.decode(outputs.logits.argmax(dim=-1)[0])

print("Qwen Model: " + out_text)

