# rule_based_baseline.py
# Simple rule-based student tracker.
# Keeps a running correct-rate per concept using an exponential moving average.

class RuleBasedConcept:
    def __init__(self, name, alpha=0.3):
        self.name  = name
        self.alpha = alpha       # smoothing factor
        self.score = 0.5        # start neutral
        self.n     = 0

    def update(self, correct: bool):
        obs = 1.0 if correct else 0.0
        if self.n == 0:
            self.score = obs
        else:
            self.score = self.alpha * obs + (1.0 - self.alpha) * self.score
        self.n += 1

    @property
    def expectation(self):
        return self.score


class RuleBasedStudentModel:
    def __init__(self, concepts: list[str]):
        self.concepts = {name: RuleBasedConcept(name) for name in concepts}
        self.history  = []

    def update(self, concept_name: str, correct: bool):
        self.concepts[concept_name].update(correct)
        self.history.append((concept_name, correct, self.concepts[concept_name].score))

    def belief(self, concept_name: str):
        return self.concepts[concept_name].score

    def knowledge_state(self):
        return {name: c.expectation for name, c in self.concepts.items()}

    def mastered(self, concept_name: str, threshold=0.7):
        return self.concepts[concept_name].score >= threshold
