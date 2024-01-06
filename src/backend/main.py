from fastapi import FastAPI, HTTPException
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

app = FastAPI(title="IMDb Sentiment Analysis Engine API")

# Load the model and tokenizer
model = load_model('../../models/sentiment_model')
with open('../../models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 300  # Ensure this is the same as during training

@app.post('/predict')
def predict_sentiment(review: str):
    try:
        # Tokenize and pad the input text
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        
        # Perform the prediction
        prediction = model.predict(padded_sequence)[0][0]
        
        # Interpret the prediction probability
        result = "Positive" if prediction > 0.5 else "Negative"
        confidence = prediction if result == "Positive" else 1 - prediction
        
        return {"result": result, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
