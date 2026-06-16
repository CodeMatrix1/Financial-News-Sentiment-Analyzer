from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert"
)

#prediction
def predict_finbert(text):
    result = classifier(text)[0]
    return result["label"]
