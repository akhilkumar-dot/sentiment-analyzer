import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Create a FastAPI instance
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, e.g., GET, POST, OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Define a request model for input data
class SentimentRequest(BaseModel):
    text: str

@app.post("/predict_sentiment/")
async def predict_sentiment(request: SentimentRequest):
    text = request.text.strip()

    if not text:
        return {"error": "Text input cannot be empty"}

    # Transform the text using the TF-IDF vectorizer
    text_tfidf = tfidf.transform([text])

    # Use the model to predict sentiment
    sentiment_prob = model.predict_proba(text_tfidf)[0]  # Get probabilities for both classes
    sentiment = model.classes_[np.argmax(sentiment_prob)]  # Predicted sentiment label

    # Confidence is the probability of the predicted sentiment class
    confidence = sentiment_prob[np.argmax(sentiment_prob)] * 100  # Convert to percentage

    # Adjust emoji based on sentiment
    emoji = "😊" if sentiment == "positive" else "😔" if sentiment == "negative" else "😐"

    # Return the sentiment analysis result
    return {
        "predicted_sentiment": sentiment,
        "emoji": emoji,
        "confidence": round(confidence, 2)  # Round to 2 decimal places
    }
