import os
from chatbot import SimpleKeywordChatbot

def run_evaluation():
    chatbot = SimpleKeywordChatbot()
    
    # Evaluation dataset: (sentence, ground_truth)
    # Ground truth values: happy, upset, calm
    test_data = [
        # Simple Positive
        ("This is a great day!", "happy"),
        ("I love this product, it is awesome", "happy"),
        ("I am so happy and full of joy", "happy"),
        
        # Simple Negative
        ("This is terrible and bad", "upset"),
        ("I hate waiting in line, it makes me angry", "upset"),
        ("I feel sad and annoyed", "upset"),
        
        # Simple Neutral
        ("The package arrived today", "calm"),
        ("I will eat lunch now", "calm"),
        ("It is raining outside", "calm"),
        
        # Complex Cases (Nuances, negations, intensifiers)
        ("The service was not good at all", "upset"),      # Keyword model will fail (identifies 'good' -> happy)
        ("I am not sad, actually I am feeling fine", "happy"), # Keyword model will fail (identifies 'sad' -> upset)
        ("Nothing makes me happy anymore", "upset")        # Keyword model will fail (identifies 'happy' -> happy)
    ]
    
    print("="*60)
    print("Evaluating Baseline Keyword-Based Sentiment Model")
    print("="*60)
    print(f"{'Sentence':<50} | {'Actual':<8} | {'Predicted':<8} | {'Status'}")
    print("-" * 80)
    
    correct = 0
    total = len(test_data)
    
    # For confusion matrix
    # Labels order: happy, upset, calm
    labels = ["happy", "upset", "calm"]
    cm = {actual: {pred: 0 for pred in labels} for actual in labels}
    
    for sentence, actual in test_data:
        pred = chatbot.analyze_sentiment(sentence)
        cm[actual][pred] += 1
        
        status = "PASS" if pred == actual else "FAIL"
        if pred == actual:
            correct += 1
            
        print(f"{sentence[:48]:<50} | {actual:<8} | {pred:<8} | {status}")
        
    accuracy = correct / total
    print("-" * 80)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print("-" * 80)
    
    # Print Confusion Matrix
    print("\nConfusion Matrix:")
    print(f"{'':<12} | {'Pred HPY':<10} | {'Pred UPS':<10} | {'Pred CLM':<10}")
    print("-" * 52)
    print(f"{'Actual HPY':<12} | {cm['happy']['happy']:<10} | {cm['happy']['upset']:<10} | {cm['happy']['calm']:<10}")
    print(f"{'Actual UPS':<12} | {cm['upset']['happy']:<10} | {cm['upset']['upset']:<10} | {cm['upset']['calm']:<10}")
    print(f"{'Actual CLM':<12} | {cm['calm']['happy']:<10} | {cm['calm']['upset']:<10} | {cm['calm']['calm']:<10}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()
