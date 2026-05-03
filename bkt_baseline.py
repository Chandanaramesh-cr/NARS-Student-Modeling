# bkt_baseline.py
# Bayesian Knowledge Tracing baseline.
# Standard 4-parameter BKT per concept, following Corbett & Anderson (1994).

class BKTConcept:
    def __init__(self, name, p_init=0.3, p_learn=0.1, p_slip=0.1, p_guess=0.2):
        self.name    = name
        self.p_known = p_init    # P(known) — current mastery estimate
        self.p_learn = p_learn   # P(learn | not known, answer given)
        self.p_slip  = p_slip    # P(wrong | known)
        self.p_guess = p_guess   # P(right | not known)

    def update(self, correct: bool):
        pk = self.p_known
        ps = self.p_slip
        pg = self.p_guess

        if correct:
            # P(known | correct)
            num = pk * (1.0 - ps)
            den = num + (1.0 - pk) * pg
        else:
            # P(known | incorrect)
            num = pk * ps
            den = num + (1.0 - pk) * (1.0 - pg)

        pk_given_obs = num / (den + 1e-9)

        # apply learning
        self.p_known = pk_given_obs + (1.0 - pk_given_obs) * self.p_learn

    @property
    def expectation(self):
        return self.p_known


class BKTStudentModel:
    def __init__(self, concepts: list[str]):
        self.concepts = {name: BKTConcept(name) for name in concepts}
        self.history  = []

    def update(self, concept_name: str, correct: bool):
        self.concepts[concept_name].update(correct)
        self.history.append((concept_name, correct, self.concepts[concept_name].p_known))

    def belief(self, concept_name: str):
        return self.concepts[concept_name].p_known

    def knowledge_state(self):
        return {name: c.expectation for name, c in self.concepts.items()}

    def mastered(self, concept_name: str, threshold=0.7):
        return self.concepts[concept_name].p_known >= threshold
