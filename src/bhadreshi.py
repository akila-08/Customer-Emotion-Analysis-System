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
from datetime import datetime
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
from bertopic import BERTopic

# ✅ Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-go-emotion",
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)


# ✅ Emotion Weights for Adorescore Calculation
emotion_weights = {
    "joy": 1.0, "trust": 0.8, "admiration": 1.0, "amazement": 0.9, "anticipation": 0.7,
    "anger": -0.8, "sadness": -0.7, "disappointment": -0.5, "fear": -0.8, "disgust": -1.0,
    "neutral": 0.0
}

# ✅ Activation Level Function
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

feedback_list = [
    "I absolutely love this product! It exceeded my expectations.",
    "The customer service was horrible. I waited 30 minutes just to talk to someone!",
    "I’m so disappointed with my purchase. It broke after just one use.",
    "Wow! I wasn’t expecting this to be so good. Very impressed!",
    "I feel cheated. The product description was completely misleading.",
    "This is my favorite purchase of the year! Highly recommend it.",
    "The app is extremely slow and keeps crashing. Very frustrating experience.",
    "I ordered a shirt, but the size was completely wrong. Poor quality control!",
    "Why is this product so expensive? It’s definitely not worth the price.",
    "The packaging was damaged when it arrived. Not acceptable!",
    "I had to return the product. The refund process was quick and smooth.",
    "The user interface of the app is outdated and hard to navigate.",
    "The customer support team was very helpful and solved my issue quickly.",
    "I love the sleek design of this gadget. Looks premium and high-end.",
    "This perfume has a wonderful fragrance that lasts all day!",
    "My package was delayed by a week. Very poor service.",
    "The laptop gets too hot after only 10 minutes of use.",
    "I am very satisfied with this purchase. Great value for money!",
    "I got a defective product and had to request a replacement.",
    "The restaurant’s food delivery was extremely late. Unacceptable!",
    "The smartwatch battery drains too fast. Not worth it.",
    "The new update on the app is amazing! So many great features.",
    "I bought a gaming console and I’m loving it! Best investment.",
    "This jacket is very comfortable and stylish. Perfect for winter!",
    "The sales team was very pushy and made me uncomfortable.",
    "Why does this product have so many fake reviews? Seems misleading.",
    "I love the eco-friendly packaging. Great initiative!",
    "The mattress is too firm for my liking. Not as described.",
    "The earbuds are super lightweight and have excellent sound quality.",
    "I feel scammed! The product doesn’t match the description at all.",
    "The refund took forever to process. Very frustrating.",
    "The online checkout process was so smooth and fast!",
    "My parcel was lost in transit. The company was unhelpful in resolving it.",
    "The fabric of this dress is really soft and comfortable.",
    "The mobile app keeps logging me out randomly. Needs improvement.",
    "The air fryer is easy to clean and very efficient!",
    "I wish there were more color options available for this product.",
    "The phone camera quality is much better than I expected!",
    "The speaker volume is too low even at max settings.",
    "I received a used product instead of a new one. Very disappointed.",
    "I had the best customer service experience! Friendly and efficient.",
    "The chair is uncomfortable for long hours of sitting.",
    "My parcel was delivered super fast. Great job!",
    "The fitness tracker motivates me to stay active. Love it!",
    "The keyboard is very responsive and great for typing.",
    "The store’s return policy is too strict. Not fair to customers.",
    "The subscription plan is too expensive for the features provided.",
    "This hairdryer is very powerful and dries hair quickly!",
    "The freezer stopped working after just a month of use.",
    "I love how easy it is to assemble this furniture.",
    "The projector has an amazing picture quality. Perfect for home theater!"
]

def get_emotions(text, min_confidence=0.05):
    """Extract emotions with intensity and confidence scores, skipping 'neutral'."""
    predictions = emotion_classifier(text)
    emotion_scores = {entry['label']: round(entry['score'], 2) for entry in predictions[0]}

    # Remove emotions with score = 0.0
    filtered_emotions = {k: v for k, v in emotion_scores.items() if v > 0.0}

    # Sort remaining emotions by confidence
    sorted_emotions = sorted(filtered_emotions.items(), key=lambda x: x[1], reverse=True)

    for emot in sorted_emotions:
        if emot[0]!="neutral":
            primary_emotion=emot
            break
    for emot in sorted_emotions:
        if emot[0]!=primary_emotion[0] and emot[0] != "neutral":
            secondary_emotion=emot
            break
    return primary_emotion, secondary_emotion, filtered_emotions


def preprocess_text(text):
    """Tokenizes and removes stopwords while preserving negation phrases."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)

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

# ✅ Preprocess All Feedback
processed_feedback = [preprocess_text(feedback) for feedback in feedback_list]

# ✅ Create Dictionary & Corpus for LDA
dictionary = corpora.Dictionary(processed_feedback)
corpus = [dictionary.doc2bow(text) for text in processed_feedback]

# ✅ Train LDA Model
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# ✅ Topic Modeling with BERTopic
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.1, metric="cosine")
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model=embedding_model)

# ✅ Fit BERTopic Model
topics, _ = topic_model.fit_transform(feedback_list)
    

def extract_topics(feedback, lda_model, dictionary):
    #Extracts main topics and subtopics using rule-based and LDA-based fallback.
    
    feedback_words = set(re.findall(r'\b\w+\b', feedback.lower()))
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
    
# ✅ Extract Topic Relevance Scores using LDA
def compute_topic_relevance(feedback):
    #Computes topic relevance scores using LDA.
    bow = dictionary.doc2bow(preprocess_text(feedback))
    topic_distribution = lda_model.get_document_topics(bow)
    return {lda_model.show_topic(topic[0], topn=1)[0][0]: float(round(topic[1], 2)) for topic in topic_distribution}

# ✅ Compute Adorescore
def compute_adorescore(emotions, topics):
    #Computes Adorescore using weighted emotion impact and topic relevance.
    weighted_sum = sum(emotion_weights.get(emotion.lower(), 0) * score for emotion, score in emotions.items())
    total_weight = sum(emotions.values())

    # Normalize Adorescore to range (-100 to 100)
    base_score = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0
    breakdown = {topic: round(base_score * (1.0 - 0.2 * i)) for i, topic in enumerate(topics["main"])}
    overall_score = sum(breakdown.values()) // len(breakdown) if breakdown else base_score

    return round(overall_score, 2), breakdown

# ✅ Process Each Feedback
output_data = []
for feedback in feedback_list:
    date = datetime.now().strftime("%Y-%m-%d")  # Assign current date

    # Get emotions
    primary_emotion, secondary_emotion, emotion_scores = get_emotions(feedback)

    # Get extracted topics
    topic_info = extract_topics(feedback, lda_model, dictionary) 

    # Compute Adorescore
    overall_score, breakdown = compute_adorescore(emotion_scores, topic_info)

    # Save result
    output_data.append({
        "date": date,
        "feedback": feedback,
        "emotions": {
            "primary": {"emotion": primary_emotion[0].capitalize(), "activation": get_activation_level(primary_emotion[0]), "intensity": primary_emotion[1]},
            "secondary": {"emotion": secondary_emotion[0].capitalize() if secondary_emotion else None, "activation": get_activation_level(secondary_emotion[0]) if secondary_emotion else None, "intensity": secondary_emotion[1] if secondary_emotion else None}
        },
        "topics": topic_info,
        "adorescore": {"overall": overall_score, "breakdown": breakdown}
    })

# ✅ Save to JSON
with open("bhadresi_op.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("✅ Adorescore Analysis Complete! JSON file saved.")