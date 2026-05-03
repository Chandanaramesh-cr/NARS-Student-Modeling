# student_simulator.py
# Simulates three student types, each with a fixed ground truth per concept.
# The simulator generates quiz answers based on that ground truth + noise.

import random

random.seed(42)

CONCEPTS = [
    "fraction_basics",
    "fraction_addition",
    "fraction_multiplication",
    "fraction_division",
    "decimal_conversion",
    "percent_problems",
]

# Ground truth = actual probability of the student getting a question right.
# These are fixed; the learning models have to figure them out from answers.
GROUND_TRUTH = {
    "strong": {
        "fraction_basics":        0.95,
        "fraction_addition":      0.90,
        "fraction_multiplication":0.92,
        "fraction_division":      0.88,
        "decimal_conversion":     0.93,
        "percent_problems":       0.85,
    },
    "struggling": {
        "fraction_basics":        0.45,
        "fraction_addition":      0.30,
        "fraction_multiplication":0.35,
        "fraction_division":      0.25,
        "decimal_conversion":     0.40,
        "percent_problems":       0.28,
    },
    "misconception": {
        # knows the easy ones, wrong model for addition and division
        "fraction_basics":        0.90,
        "fraction_addition":      0.20,   # adds numerators + denominators
        "fraction_multiplication":0.85,
        "fraction_division":      0.15,   # flips the wrong fraction
        "decimal_conversion":     0.80,
        "percent_problems":       0.75,
    },
}


def answer(profile: str, concept: str) -> bool:
    """Simulate one quiz answer for a student profile on a given concept."""
    p = GROUND_TRUTH[profile][concept]
    return random.random() < p


def generate_quiz_sequence(profile: str, n_rounds: int = 10) -> list[tuple[str, bool]]:
    """
    Returns a list of (concept, correct) pairs.
    Each concept appears n_rounds times, interleaved randomly.
    """
    events = []
    for concept in CONCEPTS:
        for _ in range(n_rounds):
            events.append((concept, answer(profile, concept)))
    random.shuffle(events)
    return events
