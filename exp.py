from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

text = "The quick brown fox "
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
out_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0])

print("Qwen Model: " + out_text)
