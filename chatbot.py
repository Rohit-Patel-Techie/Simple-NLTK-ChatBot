import json
import random
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# NLTK DOWNLOADS (run once)
# =========================
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# =========================
# LOAD INTENTS JSON
# =========================
with open("data/intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

intents = data["intents"]

# =========================
# NLP TOOLS INITIALIZATION
# =========================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# =========================
# TEXT PREPROCESSING
# =========================
def preprocess_text(text):
    """
    Clean and normalize text using NLP techniques
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords & lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return set(processed_tokens)


# =========================
# PREPROCESS ALL PATTERNS
# =========================
pattern_intent_map = []

for intent in intents:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        processed_pattern = preprocess_text(pattern)
        pattern_intent_map.append((processed_pattern, tag))

# =========================
# JACCARD SIMILARITY
# =========================
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# =========================
# INTENT PREDICTION
# =========================
def predict_intent(user_input):
    processed_input = preprocess_text(user_input)

    max_similarity = 0
    predicted_tag = None

    for pattern_set, tag in pattern_intent_map:
        similarity = jaccard_similarity(processed_input, pattern_set)
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_tag = tag

    return predicted_tag, max_similarity


# =========================
# RESPONSE FETCHING
# =========================
def get_response(tag):
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I couldn't find a suitable response."


# =========================
# CHATBOT LOOP (CLI)
# =========================
def chatbot():
    print("\n🤖 ANDC HelpDesk Chatbot")
    print("Type 'bye' or 'exit' to end the chat.\n")

    SIMILARITY_THRESHOLD = 0.25

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Bot:", get_response("goodbye"))
            break

        predicted_tag, confidence = predict_intent(user_input)

        if predicted_tag and confidence >= SIMILARITY_THRESHOLD:
            response = get_response(predicted_tag)
        else:
            response = "Sorry, I couldn't understand that. Please ask something related to Acharya Narendra Dev College."

        print("Bot:", response)


# =========================
# RUN CHATBOT
# =========================
if __name__ == "__main__":
    chatbot()
