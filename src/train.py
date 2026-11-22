
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from src.preprocessing import tokenize, stem
from src.neural_network import NeuralNetwork


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "intents.json"
MODEL_PATH = BASE_DIR / "model" / "chatbot_model.pth"

HIDDEN_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 1000


with open(DATASET_PATH, "r", encoding="utf-8") as file:
    intents = json.load(file)


all_words = []
tags = []
xy = []


for intent in intents["intents"]:
    tag = intent["tag"]

    if tag not in tags:
        tags.append(tag)

    for pattern in intent["patterns"]:
        words = tokenize(pattern)

        all_words.extend(
            stem(word) for word in words if word.isalnum()
        )

        xy.append((words, tag))


all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_train = []
y_train = []


for pattern_sentence, tag in xy:

    bag = [
        1.0 if word in [
            stem(w) for w in pattern_sentence if w.isalnum()
        ] else 0.0
        for word in all_words
    ]

    X_train.append(bag)
    y_train.append(tags.index(tag))


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)


input_size = len(X_train[0])
output_size = len(tags)

model = NeuralNetwork(input_size, HIDDEN_SIZE, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("Starting Chatbot Training...")
print("Total Training Samples:", len(X_train))
print("Total Vocabulary Words:", input_size)
print("Total Intents:", output_size)


for epoch in range(EPOCHS):

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Loss: {loss.item():.4f}"
        )


with torch.no_grad():
    predictions = torch.argmax(model(X_train), dim=1)
    accuracy = (
        (predictions == y_train).sum().item()
        / len(y_train)
    ) * 100


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    },
    MODEL_PATH
)


print("\nTraining Completed Successfully!")
print(f"Training Accuracy: {accuracy:.2f}%")
print("Model saved at:", MODEL_PATH)
