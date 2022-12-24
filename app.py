from fastapi import FastAPI
import inference
from utils.inference import Inference

description = '''
# Sentiment Analysis app using FastAPI

it is a simple app to predict the sentiment of the tweet using tensorflow sequential model
'''

inference = inference.main()
app = FastAPI(
    title='Sentiment Analysis',
    description=description,
    version='0.0.1',
    license_info={
        'name': 'MIT',
        'url': 'https://www.mit.edu/~amini/LICENSE.md',
    }
)

@app.get('/')
async def index() -> str:
    return 'Welcome to Senitment Analysis'

@app.get('/predict')
async def predict_sentiment(text: str) -> str:
    result = inference.predict_sentiment(text)
    return {'text': text, 'sentiment': result}
