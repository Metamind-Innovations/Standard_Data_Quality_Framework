import numpy as np

def calculate_rating(metric_value, is_positive=True):
    """
    Calculate a rating on a scale of 1-5 based on the metric value.
    
    For positive metrics (is_positive=True):
    - 0.0 to 0.2 -> 1
    - 0.2 to 0.4 -> 2
    - 0.4 to 0.6 -> 3
    - 0.6 to 0.8 -> 4
    - 0.8 to 1.0 -> 5
    
    For negative metrics (is_positive=False):
    - 0.0 to 0.2 -> 5
    - 0.2 to 0.4 -> 4
    - 0.4 to 0.6 -> 3
    - 0.6 to 0.8 -> 2
    - 0.8 to 1.0 -> 1
    """
    # Ensure metric_value is between 0 and 1
    metric_value = max(0, min(1, metric_value))
    
    if is_positive:
        if metric_value <= 0.2:
            return 1
        elif metric_value <= 0.4:
            return 2
        elif metric_value <= 0.6:
            return 3
        elif metric_value <= 0.8:
            return 4
        else:
            return 5
    else:
        if metric_value <= 0.2:
            return 5
        elif metric_value <= 0.4:
            return 4
        elif metric_value <= 0.6:
            return 3
        elif metric_value <= 0.8:
            return 2
        else:
            return 1

def get_ratings(check_results):
    ratings = {}
    
    positive_metrics = ["population_representativity", "metadata_granularity", 
                        "semantic_coherence_option1", "semantic_coherence_option2", 
                        "relational_consistency"]
    
    negative_metrics = ["accuracy", "coherence", "completeness"]
    
    for metric, (value, explanation) in check_results.items():
        if metric in positive_metrics:
            ratings[metric] = (calculate_rating(value, True), value, explanation)
        elif metric in negative_metrics:
            ratings[metric] = (calculate_rating(value, False), value, explanation)
        else:
            ratings[metric] = (0, value, explanation)
    
    return ratings

def get_overall_rating(ratings):
    rating_values = [r[0] for r in ratings.values() if r[0] > 0]
    return np.mean(rating_values) if rating_values else 0