import json
import re
import torch
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from transformers import pipeline

# ✅ Load Emotion Models
distilroberta_emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

go_emotions_classifier = pipeline(
    "text-classification",
    model="monologg/bert-base-cased-goemotions-original",
    return_all_scores=True
)

bart_emotion_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

    
# ✅ Emotion Weights for Adorescore Calculation
emotion_weights = {
    "joy": 1.0, "trust": 0.8, "admiration": 1.0, "amazement": 0.9, "anticipation": 0.7,
    "anger": -0.8, "sadness": -0.7, "disappointment": -0.5, "fear": -0.8, "disgust": -1.0,
    "neutral": 0.0
}

# ✅ Extract Emotions
def get_emotions(text, classifier, possible_labels=None):
    """Extract emotions with intensity and confidence scores."""
    if classifier == bart_emotion_classifier:
        predictions = classifier(text, candidate_labels=possible_labels)
        emotion_scores = {label: round(score, 2) for label, score in zip(predictions['labels'], predictions['scores'])}
    else:
        predictions = classifier(text)
        emotion_scores = {entry['label']: round(entry['score'], 2) for entry in predictions[0]}

    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    primary_emotion = sorted_emotions[0] if sorted_emotions else ("neutral", 1.0)
    secondary_emotion = sorted_emotions[1] if len(sorted_emotions) > 1 else None

    return primary_emotion, secondary_emotion, emotion_scores
