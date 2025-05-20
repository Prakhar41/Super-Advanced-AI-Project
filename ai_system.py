import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

class AIAssistant:
    def __init__(self):
        # Initialize NLP model
        self.nlp_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.nlp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Initialize computer vision
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def process_text(self, text):
        """Process text input using NLP model"""
        inputs = self.nlp_tokenizer.encode(text, return_tensors="pt")
        outputs = self.nlp_model.generate(inputs, max_length=100, num_return_sequences=1)
        response = self.nlp_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def process_image(self, image_path):
        """Process image using computer vision"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0
    
    def make_decision(self, context):
        """Make decisions based on multiple inputs"""
        # This is a placeholder for more complex decision making
        return {
            "confidence": 0.85,
            "action": "proceed",
            "reasoning": "Based on current context"
        }

# Create FastAPI app
app = FastAPI()
ai = AIAssistant()

class TextRequest(BaseModel):
    text: str

class ImageRequest(BaseModel):
    image_path: str

@app.post("/process_text")
async def process_text(request: TextRequest):
    return {"response": ai.process_text(request.text)}

@app.post("/process_image")
async def process_image(request: ImageRequest):
    return {"face_detected": ai.process_image(request.image_path)}

@app.post("/make_decision")
async def make_decision(request: dict):
    return ai.make_decision(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
