# Super Advanced AI Demo Script
import requests
import json
from PIL import Image
import cv2
import numpy as np

# Base URL for our FastAPI server
BASE_URL = "http://localhost:8000"

def process_text_demo():
    """Demonstrate text processing capabilities"""
    print("\n=== Text Processing Demo ===")
    text = "The quick brown fox jumps over the lazy dog. This is a test sentence with positive sentiment."
    
    # Process text
    response = requests.post(
        f"{BASE_URL}/process_text",
        json={"text": text}
    )
    
    print("\nProcessed Text:", response.json())

def process_image_demo():
    """Demonstrate image processing capabilities"""
    print("\n=== Image Processing Demo ===")
    # Create a simple test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite('test_image.jpg', img)
    
    # Process image
    with open('test_image.jpg', 'rb') as img_file:
        response = requests.post(
            f"{BASE_URL}/process_image",
            files={'image': img_file}
        )
    
    print("\nImage Processing Results:", response.json())

def analyze_emotion_demo():
    """Demonstrate emotion analysis capabilities"""
    print("\n=== Emotion Analysis Demo ===")
    text = "I'm really excited about this project! It's amazing!"
    
    response = requests.post(
        f"{BASE_URL}/analyze_emotion",
        json={"text": text}
    )
    
    print("\nEmotion Analysis:", response.json())

def main():
    print("=== Super Advanced AI Demo ===")
    print("\nAvailable demos:")
    print("1. Text Processing")
    print("2. Image Processing")
    print("3. Emotion Analysis")
    
    while True:
        choice = input("\nChoose a demo (1-3) or 'q' to quit: ")
        if choice == '1':
            process_text_demo()
        elif choice == '2':
            process_image_demo()
        elif choice == '3':
            analyze_emotion_demo()
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
