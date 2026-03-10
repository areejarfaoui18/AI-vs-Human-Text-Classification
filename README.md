# AI vs Human Text Classification

This project explores whether a machine learning model can distinguish between **AI-generated text** and **human-written text**. Using a dataset containing text samples and additional metadata (length, sentiment, quality score, topic, etc.), several models were trained to classify the source of the text.

The goal was to build a simple but effective pipeline that combines **text embeddings** with **structured features**.

---

## Project Workflow

1. **Data Cleaning**
   - Removed irrelevant columns (`id`, `notes`)
   - Encoded the target variable (`human = 0`, `ai = 1`)
   - One-hot encoded the `topic` feature
   - Standardized numeric features

2. **Feature Engineering**
   - Generated text embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`)
   - Combined embeddings with numeric features:
     - `length_words`
     - `quality_score`
     - `sentiment`
     - `plagiarism_score`

3. **Model Training**
   Several models were tested:
   - Logistic Regression
   - Random Forest
   - XGBoost

4. **Evaluation**
   Models were evaluated using accuracy, F1 score, and confusion matrices.

---

## Results

The **XGBoost model** achieved the best performance:

- **Accuracy:** ~94%
- **F1 Score:** ~0.94

This shows that combining **text embeddings with structured features** can effectively detect patterns in AI-generated vs human text.

---

## Example Usage

```python
import joblib
from sentence_transformers import SentenceTransformer

# Load model
model = joblib.load("xgb_ai_vs_human_model.pkl")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Example text
text = "Artificial intelligence is transforming many industries."

# Generate embedding
embedding = embedder.encode([text])

# Predict
prediction = model.predict(embedding)

print("Prediction:", "AI" if prediction[0] == 1 else "Human")
