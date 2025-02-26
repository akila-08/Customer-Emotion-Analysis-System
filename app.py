import streamlit as st
import json
import torch
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import re
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from bertopic import BERTopic

# ✅ Load Emotion Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-go-emotion",
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# ✅ Emotion Weights
emotion_weights = {
    "joy": 1.0, "trust": 0.8, "admiration": 1.0, "amazement": 0.9, "anticipation": 0.7,
    "anger": -0.8, "sadness": -0.7, "disappointment": -0.5, "fear": -0.8, "disgust": -1.0,
    "neutral": 0.0
}

# ✅ Activation Levels
def get_activation_level(emotion):
    high_activation = {"ecstasy","vigilance","admiration","terror","amazement","grief","loathing","excitement","rage"}
    medium_activation = {"anger", "fear","joy","surprise", "trust", "anticipation","sadness","disgust"}
    low_activation = {"interest", "serenity", "acceptance", "appreciation","distraction","pensiveness","boredom","annoyance"}

    if emotion.lower() in high_activation:
        return "High"
    elif emotion.lower() in medium_activation:
        return "Medium"
    else:
        return "Low"

# ✅ Extract Emotions
def get_emotions(text):
    predictions = emotion_classifier(text)
    emotion_scores = {entry['label']: round(entry['score'], 2) for entry in predictions[0]}

    # Filter emotions
    filtered_emotions = {k: v for k, v in emotion_scores.items() if v > 0.0}
    sorted_emotions = sorted(filtered_emotions.items(), key=lambda x: x[1], reverse=True)

    # Primary & Secondary Emotion Selection
    primary_emotion, secondary_emotion = None, None
    for emot in sorted_emotions:
        if emot[0] != "neutral":
            primary_emotion = emot
            break
    for emot in sorted_emotions:
        if emot[0] != "neutral" and (primary_emotion is None or emot[0] != primary_emotion[0]):
            secondary_emotion = emot
            break

    return primary_emotion, secondary_emotion, filtered_emotions

business_topics = {
    "Delivery": ["delivery", "shipping", "courier", "on-time", "fast", "late", "delay", "tracking", "parcel", "lost", "transit"],
    "Quality": ["quality", "durability", "sturdy", "premium", "amazing", "well-made", "defective", "cheap", "excellent", "high-quality"],
    "Customer Service": ["customer service", "support", "helpful", "horrible", "rude", "unhelpful", "response", "call", "chat", "friendly", "efficient"],
    "Price & Value": ["price", "expensive", "cheap", "discount", "overpriced", "deal", "cost", "worth", "affordable", "money"],
    "Product Issues": ["broken", "damaged", "faulty", "low-quality", "poor", "malfunctioning", "not working", "misleading", "scam", "fake", "wrong item"],
    "Clothes & Accessories": ["fit", "size", "fabric", "material", "small", "large", "tight", "loose", "comfortable", "stylish", "fashion"],
    "Technology & Software": ["app", "software", "update", "crash", "bug", "slow", "responsive", "user interface", "navigation", "checkout"],
    "Battery & Power": ["battery", "drains", "fast", "charge", "life", "overheating", "hot", "power", "charging issue"],
    "Refunds & Returns": ["refund", "return", "replacement", "defective", "warranty", "policy", "money back", "strict return policy"],
    "Packaging & Handling": ["packaging", "damaged", "eco-friendly", "safe", "secure", "bubble wrap", "box condition"],
    "Sound & Audio": ["sound", "bass", "volume", "audio", "speaker", "loud", "clarity", "distorted"],
    "User Experience": ["interface", "navigation", "checkout", "loading time", "easy to use", "smooth", "intuitive", "glitchy","hard","difficult","difficulty"],
    "Comfort & Feel": ["comfortable", "soft", "firm", "cushion", "cozy", "ergonomic", "adjustable"],
    "Home & Appliances": ["air fryer", "freezer", "projector", "mattress", "kitchen appliance", "heater", "microwave", "washing machine"],
    "Loyalty & Trust": ["scam", "fake reviews", "misleading", "trustworthy", "honest", "reliable", "fraud"],
    "Shopping Experience": ["checkout", "cart", "easy purchase", "hassle-free", "smooth transaction", "order process"],
    "Customer Experience": ["satisfied", "best purchase", "worst experience", "amazing", "terrible", "happy", "disappointed"],
    "Aesthetics & Design": ["sleek", "modern", "stylish", "beautiful", "ugly", "boring", "high-end", "luxury"],
    "Performance & Speed": ["fast", "slow", "efficient", "lag", "smooth", "quick", "processing"],
    "Eco-Friendliness": ["recyclable", "biodegradable", "eco-friendly", "sustainable", "green initiative"],
    "Positive Experience": ["love", "amazing", "impressed", "excellent", "great", "satisfied", "recommend", "happy"],
    "Ease of Use": ["easy to use", "user-friendly", "intuitive", "smooth", "hassle-free"],
    "Value for Money": ["worth the money", "best purchase", "affordable", "good deal", "reasonable"],
    "Performance": ["fast", "efficient", "powerful", "reliable", "high-performance"],
    "Aesthetics & Design": ["sleek", "stylish", "beautiful", "modern", "premium look"],
    "Customer Delight": ["exceeded expectations", "pleasant surprise", "delighted", "beyond expectations"]
}

def preprocess_text(feedback):
    """Tokenizes and removes stopwords while preserving negation phrases."""
    feedback = feedback.lower()
    feedback = re.sub(r'\W+', ' ', feedback)  # Remove special characters
    tokens = word_tokenize(feedback)

    # Identify negations and merge with next word
    negation_words = {"not", "no", "never", "none", "hard"}
    processed_tokens = []
    skip_next = False

    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue
        if tokens[i] in negation_words and i + 1 < len(tokens):
            processed_tokens.append(f"{tokens[i]}_{tokens[i + 1]}")  # e.g., "not_worth"
            skip_next = True
        else:
            processed_tokens.append(tokens[i])

    return processed_tokens


def extract_topics(user_feedback, lda_model, dictionary):
    """Extracts main topics and subtopics using rule-based and LDA-based fallback."""
    
    feedback_words = set(re.findall(r'\b\w+\b', user_feedback.lower()))
    topic_matches = defaultdict(set)

    # **Rule-based Matching**
    for main_topic, seed_words in business_topics.items():
        matched_words = set(seed_words) & feedback_words
        if matched_words:
            topic_matches[main_topic] = matched_words

    # **LDA Topic Extraction (Fallback)**
    if not topic_matches:
        bow = dictionary.doc2bow(feedback_words)
        topic_distribution = lda_model.get_document_topics(bow)
        topics_sorted = sorted(topic_distribution, key=lambda x: x[1], reverse=True)

        if topics_sorted:
            main_topic_id = topics_sorted[0][0]
            extracted_main_topic = lda_model.show_topic(main_topic_id, topn=1)[0][0]
            extracted_subtopics = [word[0] for word in lda_model.show_topic(main_topic_id, topn=5)][1:]

            topic_matches[extracted_main_topic] = set(extracted_subtopics)

    return {
        "main": list(topic_matches.keys()),
        "subtopics": {topic: list(words) for topic, words in topic_matches.items()}
    }



# ✅ Compute Adorescore
def compute_adorescore(emotions):
    weighted_sum = sum(emotion_weights.get(emotion.lower(), 0) * score for emotion, score in emotions.items())
    total_weight = sum(emotions.values())
    base_score = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0
    return round(base_score, 2)

# ✅ Streamlit App Layout
st.set_page_config(page_title="Customer Emotion Analysis", layout="wide")
st.title("📊 Customer Emotion Analysis System")

# ✅ Text Input
user_feedback = st.text_area("📝 Enter customer feedback:", placeholder="Type your feedback here...")

if st.button("Analyze Feedback"):
    if user_feedback:
        # ✅ Process Emotions
        primary_emotion, secondary_emotion, emotion_scores = get_emotions(user_feedback)

        # ✅ Compute Adorescore
        adorescore = compute_adorescore(emotion_scores)

        # ✅ Preprocess User Input
        processed_feedback = preprocess_text(user_feedback)

        # ✅ Create Dictionary & Corpus for LDA
        dictionary = corpora.Dictionary([processed_feedback])  
        corpus = [dictionary.doc2bow(processed_feedback)]

        # ✅ Train LDA Model
        num_topics = 5
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

        # ✅ Extract Topics using LDA (instead of BERTopic for single input)
        topic_info = extract_topics(user_feedback, lda_model, dictionary)

        # ✅ Display Results
        st.subheader("🔍 Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎭 Primary Emotion")
            st.write(f"**Emotion:** {primary_emotion[0].capitalize()}")
            st.write(f"**Activation Level:** {get_activation_level(primary_emotion[0])}")
            st.write(f"**Intensity:** {primary_emotion[1]}")

        with col2:
            st.markdown("### 🎭 Secondary Emotion")
            if secondary_emotion:
                st.write(f"**Emotion:** {secondary_emotion[0].capitalize()}")
                st.write(f"**Activation Level:** {get_activation_level(secondary_emotion[0])}")
                st.write(f"**Intensity:** {secondary_emotion[1]}")
            else:
                st.write("No secondary emotion detected.")

        # ✅ Adorescore & Topics Display (Collapsible)
        with st.expander("📊 **Adorescore & Topics Details**", expanded=False):
            st.metric(label="Adorescore", value=adorescore)

            st.markdown("### 📌 **Main Topics & Subtopics**")
            if topic_info["main"]:
                for main_topic, subtopics in topic_info["subtopics"].items():
                    st.write(f"**🟢 {main_topic}**")
                    st.write(f"🔹 {', '.join(subtopics)}")
            else:
                st.write("No relevant topics detected.")

        # ✅ Visualization: Emotion Scores (Pie Chart)
        if emotion_scores:
            st.markdown("### 📈 Emotion Scores (Hover to View)")
            df = pd.DataFrame({
                "Emotion": list(emotion_scores.keys()),
                "Score": list(emotion_scores.values())
            })

            fig = px.pie(
                data_frame=df,
                names="Emotion",
                values="Score",
                title="Emotion Distribution",
                hover_data={"Score": True},  
                labels={"Score": "Intensity"}  
            )
            fig.update_traces(textinfo="percent")  

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please enter feedback before analyzing.")

st.markdown("---")
st.markdown("🔹 *Developed with ❤️ using BERT-based emotion analysis*")
