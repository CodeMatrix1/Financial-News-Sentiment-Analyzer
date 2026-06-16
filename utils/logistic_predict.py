import pickle

with open("models/logistic_regression.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

label_map = {
    -1: "Negative",
    0: "Neutral",
    1: "Positive"
}


def predict_logistic(text):
    vec = tfidf.transform([text])
    pred = logistic_model.predict(vec)[0]
    return label_map[pred]