import numpy as np

def calculate_rating(metric_value):
    """
    Calculate a rating on a scale of 1-5 based on the metric value.
    All metrics now follow the same logic: higher values = better scores.
    
    - 0.0 to 0.2 -> 1
    - 0.2 to 0.4 -> 2
    - 0.4 to 0.6 -> 3
    - 0.6 to 0.8 -> 4
    - 0.8 to 1.0 -> 5
    """
    # Ensure metric_value is between 0 and 1
    metric_value = max(0, min(1, metric_value))
    
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

def get_ratings(check_results):
    ratings = {}
    
    # All metrics now use the same rating calculation (higher is better)
    for metric, (value, explanation) in check_results.items():
        ratings[metric] = (calculate_rating(value), value, explanation)
    
    return ratings

def get_overall_rating(ratings):
    rating_values = [r[0] for r in ratings.values() if r[0] > 0]
    return np.mean(rating_values) if rating_values else 0