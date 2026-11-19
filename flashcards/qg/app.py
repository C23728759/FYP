import spacy, torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nlp = spacy.load("en_core_web_sm")
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("qg-model")
mdl = AutoModelForSeq2SeqLM.from_pretrained("qg-model").to(dev).eval()

def answers(txt, n=8):
    d = nlp(txt)
    c = [e.text for e in d.ents] + [nc.text for nc in d.noun_chunks
                                    if len(nc) > 1]
    seen, out = set(), []
    for a in c:
        k = a.lower()
        if k not in seen:
            seen.add(k); out.append(a)
    return out[:n]

def ask(a, ctx):
    x = tok(f"generate question: answer: {a} context: {ctx}",
            return_tensors="pt", truncation=True, max_length=512).to(dev)
    with torch.no_grad():
        y = mdl.generate(**x, max_new_tokens=64, num_beams=4)
    return tok.decode(y[0], skip_special_tokens=True)

def cards(txt):
    return [[ask(a, txt), a] for a in answers(txt)]

gr.Interface(
    fn=cards,
    inputs=gr.Textbox(lines=8, label="Study text"),
    outputs=gr.Dataframe(headers=["Question", "Answer"]),
    title="Flashcard Generator (prototype)",
).launch(share=True)
