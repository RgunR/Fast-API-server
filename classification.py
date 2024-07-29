from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your classification model here. For example:
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ["Category 1", "Category 2"]  # Replace with actual categories


def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    category = labels[torch.argmax(probs, dim=-1).item()]
    return category
