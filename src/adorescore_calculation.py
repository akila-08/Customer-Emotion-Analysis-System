def compute_adorescore(emotions, topics):
    """Computes Adorescore using weighted emotion impact and topic relevance."""
    weighted_sum = sum(emotion_weights.get(emotion.lower(), 0) * score for emotion, score in emotions.items())
    total_weight = sum(emotions.values())

    # Normalize Adorescore to range (-100 to 100)
    base_score = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0
    breakdown = {topic: round(base_score * (1.0 - 0.2 * i)) for i, topic in enumerate(topics["main"])}
    overall_score = sum(breakdown.values()) // len(breakdown) if breakdown else base_score

    return round(overall_score, 2), breakdown
