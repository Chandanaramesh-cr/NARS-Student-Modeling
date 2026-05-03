# evaluation.py
# Four evaluation metrics for comparing student models against ground truth.

import math


def mean_absolute_error(predicted: dict, ground_truth: dict) -> float:
    """Average absolute difference between predicted and true mastery per concept."""
    concepts = list(ground_truth.keys())
    total = sum(abs(predicted[c] - ground_truth[c]) for c in concepts)
    return total / len(concepts)


def convergence_speed(belief_series: list[float], ground_truth: float,
                      threshold=0.05) -> int:
    """
    How many observations until the model's estimate is within `threshold`
    of ground truth and stays there for 3+ consecutive steps.
    Returns len(series) if it never converges.
    """
    streak = 0
    for i, val in enumerate(belief_series):
        if abs(val - ground_truth) <= threshold:
            streak += 1
            if streak >= 3:
                return i - 2   # first step of the streak
        else:
            streak = 0
    return len(belief_series)


def convergence_speed_all(series_by_concept: dict, ground_truth: dict,
                           threshold=0.05) -> float:
    """Average convergence speed across all concepts."""
    speeds = [
        convergence_speed(series_by_concept[c], ground_truth[c], threshold)
        for c in ground_truth
    ]
    return sum(speeds) / len(speeds)


def misconception_detection_accuracy(final_beliefs: dict,
                                     ground_truth: dict,
                                     mastery_threshold=0.6) -> float:
    """
    Binary classification accuracy: did the model correctly label each
    concept as mastered or not, compared to ground truth?
    """
    correct = 0
    for concept in ground_truth:
        predicted_mastered = final_beliefs[concept] >= mastery_threshold
        actual_mastered    = ground_truth[concept]  >= mastery_threshold
        if predicted_mastered == actual_mastered:
            correct += 1
    return correct / len(ground_truth)


def belief_stability(belief_series: list[float], window=5) -> float:
    """
    Average standard deviation over the last `window` steps — lower is more stable.
    Measures how much the model is still fluctuating at the end.
    """
    tail = belief_series[-window:]
    if len(tail) < 2:
        return 0.0
    mean = sum(tail) / len(tail)
    var  = sum((x - mean) ** 2 for x in tail) / (len(tail) - 1)
    return math.sqrt(var)


def full_evaluation_report(model_name: str,
                            series_by_concept: dict,
                            final_beliefs: dict,
                            ground_truth: dict) -> dict:
    """Run all four metrics and return a summary dict."""
    return {
        "model":         model_name,
        "mae":           round(mean_absolute_error(final_beliefs, ground_truth), 4),
        "convergence":   round(convergence_speed_all(series_by_concept, ground_truth), 2),
        "misconception": round(misconception_detection_accuracy(final_beliefs, ground_truth), 4),
        "stability":     round(
            sum(belief_stability(series_by_concept[c]) for c in ground_truth) / len(ground_truth),
            4
        ),
    }
