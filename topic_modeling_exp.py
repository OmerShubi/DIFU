from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

mname = "cristian-popa/bart-tl-ng"
tokenizer = AutoTokenizer.from_pretrained(mname)
model = AutoModelForSeq2SeqLM.from_pretrained(mname)

# Load the data
from datasets import load_dataset
import seaborn as sns

dataset = load_dataset("reddit_tifu", 'long')

input=dataset['train']
for i in range(25,30):
    input_text=input['tldr'][i]
    print('current text:')
    print(input_text)
    enc = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_length=15,
        min_length=1,
        do_sample=False,
        num_beams=25,
        length_penalty=1.0,
        repetition_penalty=1.5
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f' topic: {decoded}') # windows live messenger