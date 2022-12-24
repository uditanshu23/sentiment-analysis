from fastapi import FastAPI
import inference
from utils.inference import Inference

inference = inference.main()
app = FastAPI()

@app.get('/')
async def index() -> str:
    return 'Welcome to Senitment Analysis'

@app.get('/predict')
async def predict_sentiment(text: str) -> str:
    result = inference.predict_sentiment(text)
    return result
