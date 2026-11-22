
import json
import random
import torch

from pathlib import Path

from src.preprocessing import tokenize, bag_of_words, stem
from src.neural_network import NeuralNetwork


BASE_DIR = Path(__file__).resolve().parent.parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

        self.model = NeuralNetwork(
            checkpoint["input_size"],
            checkpoint["hidden_size"],
            checkpoint["output_size"]
        ).to(DEVICE)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.confidence_threshold = 0.70
        self.word_coverage_threshold = 0.50

        self.vocabulary = set(self.all_words)

        # Common words that do not provide much intent information
        self.common_words = {
            "a", "an", "the", "is", "am", "are", "was", "were",
            "what", "who", "where", "when", "why", "how",
            "do", "does", "did", "can", "could", "would",
            "should", "will", "shall", "i", "you", "we",
            "they", "he", "she", "it", "me", "my", "your",
            "our", "their", "this", "that", "these", "those",
            "to", "of", "in", "on", "at", "for", "from",
            "with", "and", "or", "but", "please", "some",
            "tell", "explain", "give", "show", "about"
        }

        self.fallback_response = (
            "I'm sorry, I didn't understand that. "
            "Could you please rephrase?"
        )

    def get_response(self, message):

        # Handle empty messages
        if not message or not message.strip():

            return {
                "response": "Please enter a message.",
                "intent": "unknown",
                "confidence": 0.0
            }

        # Tokenize user message
        tokens = tokenize(message)

        # Keep meaningful words
        meaningful_words = [
            stem(word)
            for word in tokens
            if word.isalnum()
            and stem(word) not in self.common_words
        ]

        # Check recognized words
        recognized_words = [
            word for word in meaningful_words
            if word in self.vocabulary
        ]

        # Reject messages with no recognized meaningful words
        if meaningful_words:

            if not recognized_words:

                return {
                    "response": self.fallback_response,
                    "intent": "unknown",
                    "confidence": 0.0
                }

            coverage = (
                len(recognized_words) / len(meaningful_words)
            )

            if coverage < self.word_coverage_threshold:

                return {
                    "response": self.fallback_response,
                    "intent": "unknown",
                    "confidence": 0.0
                }

        # Convert message into Bag of Words
        X = bag_of_words(tokens, self.all_words)

        # Reject completely empty input vectors
        if not X.any():

            return {
                "response": self.fallback_response,
                "intent": "unknown",
                "confidence": 0.0
            }

        # Convert into PyTorch tensor
        X = torch.from_numpy(X).float().unsqueeze(0).to(DEVICE)

        # Predict intent
        with torch.no_grad():

            output = self.model(X)

            probabilities = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(
                probabilities, dim=1
            )

            confidence = confidence.item()

            predicted_tag = self.tags[predicted.item()]

        # Check prediction confidence
        if confidence < self.confidence_threshold:

            return {
                "response": self.fallback_response,
                "intent": "unknown",
                "confidence": confidence
            }

        # Find matching intent
        for intent in self.intents:

            if intent["tag"] == predicted_tag:

                response = random.choice(intent["responses"])

                return {
                    "response": response,
                    "intent": predicted_tag,
                    "confidence": confidence
                }

        # Default fallback
        return {
            "response": self.fallback_response,
            "intent": "unknown",
            "confidence": confidence
        }


# Run chatbot in terminal
if __name__ == "__main__":

    chatbot = ChatBot()

    print("\n========== INTENT-BASED AI CHATBOT ==========")
    print("Type 'quit' to exit.")
    print("=============================================\n")

    while True:

        message = input("You: ")

        if message.lower().strip() == "quit":

            print("Bot: Goodbye!")
            break

        result = chatbot.get_response(message)

        print("\nBot:", result["response"])
        print("Intent:", result["intent"])

        print(
            "Confidence:",
            round(result["confidence"] * 100, 2),
            "%"
        )

        print()
