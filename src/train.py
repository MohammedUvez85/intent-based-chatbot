
import json
import torch
import numpy as np

from pathlib import Path
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from src.preprocessing import preprocess, build_vocabulary, bag_of_words
from src.neural_network import NeuralNetwork


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "intents.json"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "chatbot_model.pth"


# Training configuration
HIDDEN_SIZE = 128
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 1000
RANDOM_SEED = 42


def load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        intents = json.load(file)

    all_words, tags = build_vocabulary(intents)

    X = []
    y = []

    for intent in intents["intents"]:

        tag_index = tags.index(intent["tag"])

        for pattern in intent["patterns"]:

            words = preprocess(pattern)
            bag = bag_of_words(words, all_words)

            X.append(bag)
            y.append(tag_index)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, all_words, tags


def create_model(input_size, output_size):

    return NeuralNetwork(
        input_size,
        HIDDEN_SIZE,
        output_size
    )


def train_model(X, y, input_size, output_size):

    torch.manual_seed(RANDOM_SEED)

    model = create_model(input_size, output_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    features = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(features, labels)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for batch_features, batch_labels in loader:

            outputs = model(batch_features)

            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 200 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], "
                f"Loss: {total_loss / len(loader):.4f}"
            )

    return model


def evaluate_model(model, X_test, y_test, tags):

    model.eval()

    test_features = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = model(test_features)

        predictions = torch.argmax(outputs, dim=1).numpy()

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(tags)))
    )

    print("\n========== MODEL EVALUATION ==========")

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nIntent Labels:")
    print(tags)

    return accuracy, precision, recall


def save_model(model, all_words, tags):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {

        "model_state_dict": model.state_dict(),

        "input_size": len(all_words),

        "hidden_size": HIDDEN_SIZE,

        "output_size": len(tags),

        "all_words": all_words,

        "tags": tags

    }

    torch.save(checkpoint, MODEL_PATH)

    print("\nModel saved successfully!")
    print(f"Model location: {MODEL_PATH}")


def main():

    print("Loading Intent Dataset...")

    X, y, all_words, tags = load_dataset()

    print("\nDataset Loaded Successfully")

    print("Total Patterns:", len(X))
    print("Total Vocabulary Words:", len(all_words))
    print("Total Intents:", len(tags))

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    print("\nTraining Evaluation Model...")

    evaluation_model = train_model(
        X_train,
        y_train,
        len(all_words),
        len(tags)
    )

    evaluate_model(
        evaluation_model,
        X_test,
        y_test,
        tags
    )

    print("\nTraining Final Model Using Full Dataset...")

    final_model = train_model(
        X,
        y,
        len(all_words),
        len(tags)
    )

    save_model(
        final_model,
        all_words,
        tags
    )

    print("\n========== TRAINING COMPLETED ==========")


if __name__ == "__main__":

    main()
