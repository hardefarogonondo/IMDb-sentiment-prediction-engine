# Import Libraries
from config.config import Review
from fastapi import FastAPI, HTTPException
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Initialization
model = load_model('/app/models/sentiment_model')
with open('/app/models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
app = FastAPI()
max_length = 300


@app.post('/predict')
def predict_sentiment(review: Review):
    try:
        sequence = tokenizer.texts_to_sequences([review.review])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        prediction = model.predict(padded_sequence)[0][0]
        result = "Positive" if prediction > 0.5 else "Negative"
        confidence = float(prediction) if result == "Positive" else float(1 - prediction)
        return {"result": result, "confidence": confidence}
    except Exception as e:
        print(f"Error: {e}")
        print(f"Input review: {review.review}")
        raise HTTPException(status_code=500, detail=str(e))
