# nars_engine.py
# Non-Axiomatic Logic inference engine for student knowledge tracking.
# Truth-value math follows Wang (2013), Non-Axiomatic Logic, World Scientific.

K = 1.0   # evidential horizon


# ---------- truth functions ----------

def tv_and(a, b):
    return a * b

def tv_or(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)

def tv_not(a):
    return 1.0 - a


def revision(f1, c1, f2, c2):
    """Merge two independent pieces of evidence about the same statement."""
    w1 = c1 / (1.0 - c1 + 1e-9)
    w2 = c2 / (1.0 - c2 + 1e-9)
    w  = w1 + w2
    f  = (w1 * f1 + w2 * f2) / (w + 1e-9)
    c  = w / (w + K)
    return round(f, 4), round(min(c, 0.9999), 4)


def expectation(f, c):
    """Expected utility of a belief — used to rank knowledge estimates."""
    return c * (f - 0.5) + 0.5


# ---------- concept / belief store ----------

class Concept:
    """
    Represents one topic (e.g. 'fraction_addition').
    Holds the current belief: how well the student knows it.
    """
    def __init__(self, name):
        self.name = name
        self.f = 0.5   # frequency  — starts at maximum uncertainty
        self.c = 0.0   # confidence — no evidence yet

    def observe(self, correct: bool, obs_confidence=0.9):
        """
        Feed one quiz answer into this concept's belief.
        correct=True  → observed frequency 1.0
        correct=False → observed frequency 0.0
        """
        obs_f = 1.0 if correct else 0.0
        self.f, self.c = revision(self.f, self.c, obs_f, obs_confidence)

    @property
    def expectation(self):
        return expectation(self.f, self.c)

    def __repr__(self):
        return f"<{self.name}> f={self.f:.3f} c={self.c:.3f} E={self.expectation:.3f}"


# ---------- student model ----------

class NARSStudentModel:
    """
    Full student model built on NAL belief revision.

    Each concept is tracked independently. Every time the student answers
    a question, we call concept.observe() and the NAL revision rule
    merges old evidence with new evidence — no separate prior needed.
    """

    def __init__(self, concepts: list[str]):
        self.concepts = {name: Concept(name) for name in concepts}
        self.history  = []   # (concept, correct, f_after, c_after)

    def update(self, concept_name: str, correct: bool):
        c = self.concepts[concept_name]
        c.observe(correct)
        self.history.append((concept_name, correct, c.f, c.c))

    def belief(self, concept_name: str):
        """Return (f, c) for one concept."""
        c = self.concepts[concept_name]
        return c.f, c.c

    def knowledge_state(self):
        """Return a dict of concept → expectation value."""
        return {name: con.expectation for name, con in self.concepts.items()}

    def mastered(self, concept_name: str, threshold=0.7):
        return self.concepts[concept_name].expectation >= threshold

    def snapshot(self):
        return {
            name: {"f": con.f, "c": con.c, "E": con.expectation}
            for name, con in self.concepts.items()
        }
