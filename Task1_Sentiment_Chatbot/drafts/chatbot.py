import os
import csv
from datetime import datetime

class SimpleKeywordChatbot:
    def __init__(self, log_file="conversation_log.csv"):
        self.log_file = log_file
        # Positive and negative keywords (lowercased)
        self.positive_keywords = {"good", "great", "happy", "love", "wonderful", "excellent", "awesome", "joy", "nice", "glad"}
        self.negative_keywords = {"bad", "sad", "angry", "hate", "terrible", "awful", "frustrated", "annoyed", "wrong", "annoy"}
        
        # Ensure log file has header
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_input", "detected_sentiment", "bot_response"])

    def analyze_sentiment(self, text):
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in self.positive_keywords)
        neg_count = sum(1 for w in words if w in self.negative_keywords)
        
        if pos_count > neg_count:
            return "happy"
        elif neg_count > pos_count:
            return "upset"
        else:
            return "calm"

    def get_response(self, sentiment):
        if sentiment == "happy":
            return "I'm glad you're happy! Let me know how else I can help."
        elif sentiment == "upset":
            return "I'm sorry to hear that. I want to help make things better."
        else:
            return "Okay, got it. What else would you like to discuss?"

    def log_chat(self, user_input, sentiment, response):
        timestamp = datetime.now().isoformat()
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_input, sentiment, response])

    def run(self):
        print("="*60)
        print("Simple Keyword-Based Sentiment Chatbot (v1 Draft)")
        print("Type 'exit' or 'quit' to stop the chat.")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
                
            if not user_input:
                continue
                
            if user_input.lower() in {"exit", "quit"}:
                print("Bot: Goodbye!")
                break
                
            sentiment = self.analyze_sentiment(user_input)
            response = self.get_response(sentiment)
            
            print(f"Bot [Sentiment: {sentiment.upper()}]: {response}")
            self.log_chat(user_input, sentiment, response)

if __name__ == "__main__":
    # Log relative to where the script is
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "conversation_log.csv")
    chatbot = SimpleKeywordChatbot(log_file=log_path)
    chatbot.run()
