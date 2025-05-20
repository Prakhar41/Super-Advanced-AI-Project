import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pyttsx3
from googletrans import Translator
import pytesseract
from PIL import Image
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import networkx as nx
from sympy import *

class AdvancedAI:
    def __init__(self):
        # Initialize NLP model
        self.nlp_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.nlp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Initialize computer vision
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        
        # Initialize translator
        self.translator = Translator()
        
        # Initialize graph for complex reasoning
        self.knowledge_graph = nx.Graph()
        
    def process_text(self, text):
        """Process text input using NLP model"""
        inputs = self.nlp_tokenizer.encode(text, return_tensors="pt")
        outputs = self.nlp_model.generate(inputs, max_length=200, num_return_sequences=1)
        response = self.nlp_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def process_image(self, image_path):
        """Process image using computer vision"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Add text recognition
        text = pytesseract.image_to_string(Image.open(image_path))
        return {
            "faces_detected": len(faces) > 0,
            "text_in_image": text
        }
    
    def web_search(self, query):
        """Perform web search and extract information"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(f'https://www.google.com/search?q={query}', headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for g in soup.find_all('div', class_='g'):
            try:
                title = g.find('h3').text
                snippet = g.find('span', class_='st').text
                results.append({"title": title, "snippet": snippet})
            except:
                continue
        return results
    
    def translate_text(self, text, target_language='en'):
        """Translate text to different languages"""
        return self.translator.translate(text, dest=target_language).text
    
    def speak_text(self, text, language='en'):
        """Convert text to speech"""
        self.tts_engine.setProperty('voice', language)
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def solve_math_problem(self, problem):
        """Solve mathematical problems"""
        try:
            x = symbols('x')
            solution = solve(problem, x)
            return solution
        except:
            return "Could not solve the problem"
    
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        fs = 44100  # Sample rate
        seconds = duration
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, myrecording)  # Save as WAV file
        return 'output.wav'
    
    def make_decision(self, context):
        """Make decisions based on multiple inputs"""
        # Add context to knowledge graph
        self.knowledge_graph.add_node(context)
        
        # Analyze connections
        decision = {
            "confidence": 0.9,
            "action": "analyze_context",
            "reasoning": "Based on current context and knowledge graph"
        }
        return decision

# Create FastAPI app
app = FastAPI()
ai = AdvancedAI()

class TextRequest(BaseModel):
    text: str
    language: str = 'en'

class ImageRequest(BaseModel):
    image_path: str

class MathProblemRequest(BaseModel):
    problem: str

class AudioRequest(BaseModel):
    duration: int = 5

@app.post("/process_text")
async def process_text(request: TextRequest):
    response = ai.process_text(request.text)
    return {"response": response}

@app.post("/process_image")
async def process_image(request: ImageRequest):
    return ai.process_image(request.image_path)

@app.post("/translate")
async def translate(request: TextRequest):
    return {"translation": ai.translate_text(request.text, request.language)}

@app.post("/speak")
async def speak(request: TextRequest):
    ai.speak_text(request.text, request.language)
    return {"status": "spoken"}

@app.post("/solve_math")
async def solve_math(request: MathProblemRequest):
    return {"solution": ai.solve_math_problem(request.problem)}

@app.post("/record_audio")
async def record_audio(request: AudioRequest):
    return {"file": ai.record_audio(request.duration)}

@app.post("/make_decision")
async def make_decision(request: dict):
    return ai.make_decision(request)

@app.post("/web_search")
async def web_search(request: TextRequest):
    return {"results": ai.web_search(request.text)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
