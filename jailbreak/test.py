# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import logging

# Set up basic logging to see output in Google Cloud Run
logging.basicConfig(level=logging.INFO)

# --- Model Loading ---
# This happens only once when the container starts up.
# The pipeline function from Hugging Face handles all the complexity.
try:
    logging.info("Loading the jailbreak detection model...")
    # Using a specific version for stability
    classifier = pipeline(
        "text-classification", 
        model="GuardrailsAI/prompt-saturation-attack-detector",
        revision="fb39b1a26f6354f9a76722c6e615f69661c1a967"
    )
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    classifier = None

# --- API Definition ---
app = FastAPI()

# Define the structure of the incoming request body
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    """A simple endpoint to check if the service is alive."""
    return {"status": "Model service is running."}


@app.post("/classify")
def classify_prompt(request: PromptRequest):
    """
    Receives a prompt, classifies it as 'SAFE' or 'JAILBREAK', 
    and returns the result.
    """
    if not classifier:
        return {"error": "Model is not available."}, 500

    prompt_text = request.prompt
    logging.info(f"Received prompt for classification: '{prompt_text[:80]}...'")

    # The model returns a list of dictionaries, e.g., [{'label': 'JAILBREAK', 'score': 0.99}]
    results = classifier(prompt_text)
    
    # Extract the primary result
    classification_result = results[0]
    verdict = classification_result['label']
    score = classification_result['score']

    logging.info(f"Classification complete. Verdict: {verdict}, Score: {score:.4f}")

    # For the demo, we are logging the verdict. This is what you'll show in the presentation.
    # In a real system, this verdict could trigger other actions.
    
    return {
        "prompt": prompt_text,
        "verdict": verdict,
        "confidence_score": score
    }