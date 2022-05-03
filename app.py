import torch

import gradio as gr

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("aditi2222/automatic_title_generation")

model = AutoModelForSeq2SeqLM.from_pretrained("aditi2222/automatic_title_generation")


def tokenize_data(text):
    # Tokenize the review body
    input_ = str(text) + ' </s>'
    max_len = 120
    # tokenize inputs
    tokenized_inputs = tokenizer(input_, padding='max_length', truncation=True, max_length=max_len,
                                 return_attention_mask=True, return_tensors='pt')

    inputs = {"input_ids": tokenized_inputs['input_ids'],
              "attention_mask": tokenized_inputs['attention_mask']}
    return inputs


def generate_answers(text):
    inputs = tokenize_data(text)
    results = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], do_sample=True,
                             max_length=120,
                             top_k=120,
                             top_p=0.98,
                             early_stopping=True,
                             num_return_sequences=1)
    answer = tokenizer.decode(results[0], skip_special_tokens=True)
    return answer


iface = gr.Interface(fn=generate_answers, inputs=['text'], outputs=["text"])
iface.launch(inline=False, share=True, server_name='0.0.0.0')