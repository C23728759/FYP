from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

M = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(M)
mdl = AutoModelForSeq2SeqLM.from_pretrained(M)

ctx = "Photosynthesis takes place in the chloroplasts of plant cells."
p = f"Write a question whose answer is 'chloroplasts'. Text: {ctx}"

x = tok(p, return_tensors="pt")
y = mdl.generate(**x, max_new_tokens=64, num_beams=4)
print(tok.decode(y[0], skip_special_tokens=True))
