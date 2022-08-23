from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import *
num_labels = 1
num_labels = 1
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
