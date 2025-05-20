from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json

# Import our AI class
from indispensable_ai import IndispensableAI

# Create FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize AI instance
ai = IndispensableAI()

@app.get("/")
async def home(request: Request):
    """Serve the main interface page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_text")
async def process_text(request: Request):
    """Process text input from the interface"""
    data = await request.json()
    text = data.get("text", "")
    
    # Process text using our AI
    result = ai.process_text(text)
    
    return result

@app.post("/process_image")
async def process_image(request: Request):
    """Process image input from the interface"""
    data = await request.json()
    image_path = data.get("image_path", "")
    
    # Process image using our AI
    result = ai.process_image(image_path)
    
    return result

@app.post("/control_computer")
async def control_computer(request: Request):
    """Control computer actions from the interface"""
    data = await request.json()
    action = data.get("action", "")
    
    # Control computer using our AI
    result = ai.control_computer(action)
    
    return result

@app.post("/analyze_emotion")
async def analyze_emotion(request: Request):
    """Analyze emotion from the interface"""
    data = await request.json()
    text = data.get("text", "")
    
    # Analyze emotion using our AI
    result = ai.analyze_emotion(text)
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
