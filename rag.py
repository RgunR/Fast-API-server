from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Loading the language model.

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

articles = [
    {"title": "Managing Stress", "content": "Content about managing stress..."},
    {"title": "Coping with Anxiety", "content": "Content about coping with anxiety..."},
    {"title": "Dealing with Depression", "content": "Content about dealing with depression..."}
]


def retrieve(prompt):
    return articles


def generate(prompt, context):
    input_text = f"Context: {context} Prompt: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def rag_response(prompt):
    retrieved_articles = retrieve()
    generated_responses = [generate(prompt, article['content'])for article in retrieved_articles]
    return [{"titles": article['title'], "content": response} for article, response in zip(retrieved_articles, generated_responses)]
