import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)

M = "google/flan-t5-small"
tok = AutoTokenizer.from_pretrained(M)
mdl = AutoModelForSeq2SeqLM.from_pretrained(M)

ds = load_dataset("rajpurkar/squad")
tr = ds["train"].shuffle(seed=42).select(range(5000))
va = ds["validation"].select(range(500))

def prep(b):
    src = [f"generate question: answer: {a['text'][0]} context: {c}"
           for a, c in zip(b["answers"], b["context"])]
    x = tok(src, max_length=512, truncation=True)
    x["labels"] = tok(text_target=b["question"], max_length=64,
                      truncation=True)["input_ids"]
    return x

cols = tr.column_names
tr = tr.map(prep, batched=True, remove_columns=cols)
va = va.map(prep, batched=True, remove_columns=cols)

args = Seq2SeqTrainingArguments(
    output_dir="ckpt",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=50,
    report_to="none",
)

t = Seq2SeqTrainer(
    model=mdl, args=args,
    train_dataset=tr, eval_dataset=va,
    processing_class=tok,
    data_collator=DataCollatorForSeq2Seq(tok, model=mdl),
)
t.train()
t.save_model("qg-model")
tok.save_pretrained("qg-model")
