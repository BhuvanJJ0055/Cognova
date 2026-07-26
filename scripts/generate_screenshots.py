import os
from PIL import Image, ImageDraw

def generate_screenshots():
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "screenshots")
    os.makedirs(target_dir, exist_ok=True)
    
    screenshots = [
        "task1_sentiment_negative.png",
        "task1_sentiment_positive.png",
        "task2_medical_qa.png",
        "task4_arxiv_search.png",
        "task5_multimodal.png",
        "task6_multilingual.png"
    ]
    
    for filename in screenshots:
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            img = Image.new("RGB", (800, 450), color=(24, 30, 42))
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 20, 780, 430], outline=(79, 70, 229), width=3)
            draw.text((50, 200), f"Cognova Screenshot Artifact:\n{filename}", fill=(255, 255, 255))
            img.save(filepath)
            print(f"Generated {filepath}")

if __name__ == "__main__":
    generate_screenshots()
