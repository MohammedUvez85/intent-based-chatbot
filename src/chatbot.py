
import json
import random
import torch
import numpy as np

from pathlib import Path
from src.preprocessing import tokenize, bag_of_words
from src.neural_network import NeuralNet


BASE_DIR = Path(__file__).resolve().parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChatBot:

    def __init__(self):

        model_path = BASE_DIR / "model" / "chatbot_model.pth"
        intents_path = BASE_DIR / "dataset" / "intents.json"

        with open(intents_path, "r", encoding="utf-8") as file:
            self.intents = json.load(file)["intents"]

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )

        self.all_words = checkpoint["all_words"]
        self.tags = checkpoint["tags"]

        self.model = NeuralNet(
            checkpoint["input_size"],
            checkpoint["hidden_size"],
            checkpoint["output_size"]
        ).to(DEVICE)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.confidence_threshold = 0.70

    def get_response(self, message):

        tokens = tokenize(message)

        X = bag_of_words(tokens, self.all_words)

        X = torch.from_numpy(X).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = self.model(X)

            probabilities = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(probabilities, dim=1)

            confidence = confidence.item()

            predicted_tag = self.tags[predicted.item()]

        if confidence < self.confidence_threshold:

            return {
                "response": "I'm sorry, I didn't understand that. Could you please rephrase?",
                "intent": "unknown",
                "confidence": confidence
            }

        for intent in self.intents:

            if intent["tag"] == predicted_tag:

                response = random.choice(intent["responses"])

                return {
                    "response": response,
                    "intent": predicted_tag,
                    "confidence": confidence
                }

        return {
            "response": "Sorry, I couldn't find a suitable response.",
            "intent": "unknown",
            "confidence": confidence
        }


if __name__ == "__main__":

    chatbot = ChatBot()

    print("\n========== INTENT-BASED AI CHATBOT ==========")
    print("Type 'quit' to exit.")
    print("=============================================\n")

    while True:

        message = input("You: ")

        if message.lower() == "quit":
            print("Bot: Goodbye!")
            break

        result = chatbot.get_response(message)

        print("Bot:", result["response"])
        print("Intent:", result["intent"])
        print("Confidence:", round(result["confidence"] * 100, 2), "%")
        print()
