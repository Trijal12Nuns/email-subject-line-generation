
import subprocess
import torch

subprocess.call(['pip', 'install', 'uvicorn==0.17.6','fastapi==0.99.1','pydantic==1.10.10', 'requests==2.23.0','jinja2==3.1.2','python-multipart','numpy','pandas','setuptools-rust','accelerate'])
subprocess.call(['pip', 'install', '--upgrade', 'transformers[torch]'])
import os

import streamlit as st
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer ,GPT2Model




current_path = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(current_path, "gpt_tokenizer")
model_path = os.path.join(current_path, "gpt2_3epoch")
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path) # also try gpt2-medium
model = GPT2LMHeadModel.from_pretrained(model_path)


def generate_text(model, tokenizer, prompt_text):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=25, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def subject_gen_func(email):
    device = "cpu"
    prompt = email
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id
    output_ids = model.generate(input_ids, max_length=1024, num_return_sequences=1,attention_mask=attention_mask,
            pad_token_id=pad_token_id)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text.split("<subject>")[1].replace("<eos>","")
    


def main():
    st.title("Email Subject Line Generator")

    # Load the GPT-2 model and tokenizer

    # User input
    prompt_text = st.text_area("Enter Prompt Text")

    # Generate button
    if st.button("Generate Email Text"):
        if prompt_text:
            # Generate text using the loaded model
            generated_text = subject_gen_func(prompt_text)
            st.success(f"Generated Email Text: {generated_text}")
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()
