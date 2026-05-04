# nars_engine.py
# Non-Axiomatic Logic inference engine for student knowledge tracking.
# Truth-value math follows Wang (2013), Non-Axiomatic Logic, World Scientific.
#
# Narsese belief statement format (NAL-1 / NAL-4):
#   <student --> [knows-{concept}]>. %{frequency};{confidence}%
#
# Example — student answered fraction_addition correctly:
#   <student --> [knows-fraction_addition]>. %1.0;0.9%
#
# Example — student answered fraction_addition incorrectly:
#   <student --> [knows-fraction_addition]>. %0.0;0.9%
#
# Each statement is fed into the NAL revision rule, which merges the new
# evidence with the existing belief to produce an updated truth value.

K = 1.0   # evidential horizon (Wang, 2013)
OBS_CONFIDENCE = 0.9   # confidence assigned to a single quiz observation


# ---------- Narsese formatting ----------

def narsese_statement(concept_name: str, correct: bool) -> str:
    """
    Format a quiz answer as a Narsese belief statement.
    This is the exact string that would be sent to OpenNARS.

    Correct answer   → frequency 1.0 (strong positive evidence)
    Incorrect answer → frequency 0.0 (strong negative evidence)
    Confidence is fixed at OBS_CONFIDENCE for every observation.
    """
    freq  = 1.0 if correct else 0.0
    label = concept_name.replace("_", "-")
    return f"<student --> [knows-{label}]>. %{freq:.1f};{OBS_CONFIDENCE}%"


def parse_truth_value(statement: str):
    """
    Parse frequency and confidence back out of a Narsese truth-value string.
    e.g. '%1.0;0.9%' → (1.0, 0.9)
    """
    tv   = statement.split("%")[1]
    f, c = tv.split(";")
    return float(f), float(c)


# ---------- NAL truth functions ----------

def tv_and(a, b):
    return a * b

def tv_or(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)

def tv_not(a):
    return 1.0 - a


def revision(f1, c1, f2, c2):
    """
    NAL revision rule — merge two independent pieces of evidence.
    Converts confidence to weight, takes weighted average of frequencies,
    then converts total weight back to confidence.
    """
    w1 = c1 / (1.0 - c1 + 1e-9)
    w2 = c2 / (1.0 - c2 + 1e-9)
    w  = w1 + w2
    f  = (w1 * f1 + w2 * f2) / (w + 1e-9)
    c  = w / (w + K)
    return round(f, 4), round(min(c, 0.9999), 4)


def expectation(f, c):
    """
    NAL expectation — converts truth value to a single scalar.
    E = 0.5 at maximum uncertainty (f=0.5, c=0).
    E → 1 as high-frequency belief gains confidence.
    E → 0 as low-frequency belief gains confidence.
    """
    return c * (f - 0.5) + 0.5


# ---------- Concept / belief store ----------

class Concept:
    """
    Represents one knowledge concept (e.g. 'fraction_addition').
    Maintains a NAL truth value (f, c) updated by the revision rule.
    Also stores the full Narsese statement log for inspection.
    """
    def __init__(self, name):
        self.name        = name
        self.f           = 0.5   # frequency  — maximum uncertainty
        self.c           = 0.0   # confidence — no evidence yet
        self.narsese_log = []    # every Narsese statement fed in

    def observe(self, correct: bool):
        """
        Translate one quiz answer into a Narsese statement, log it,
        then apply the NAL revision rule to update the belief.
        """
        statement = narsese_statement(self.name, correct)
        self.narsese_log.append(statement)

        obs_f, obs_c = parse_truth_value(statement)
        self.f, self.c = revision(self.f, self.c, obs_f, obs_c)

    @property
    def expectation(self):
        return expectation(self.f, self.c)

    def current_narsese(self) -> str:
        """Current belief state as a Narsese statement."""
        label = self.name.replace("_", "-")
        return f"<student --> [knows-{label}]>. %{self.f:.4f};{self.c:.4f}%"

    def __repr__(self):
        return (f"{self.current_narsese()}  "
                f"[E={self.expectation:.3f}, obs={len(self.narsese_log)}]")


# ---------- Student model ----------

class NARSStudentModel:
    """
    Full NARS-based student model.

    Every quiz answer is first converted to a Narsese belief statement:
        <student --> [knows-{concept}]>. %{f};{c}%

    That statement is then processed by the NAL revision rule, which
    merges the new evidence with the concept's existing belief state.
    No training data, no priors, no retraining — the model builds its
    knowledge estimate purely from observed evidence.
    """

    def __init__(self, concepts: list[str]):
        self.concepts = {name: Concept(name) for name in concepts}
        self.history  = []   # (narsese_statement, f_after, c_after)

    def update(self, concept_name: str, correct: bool):
        """Feed one quiz answer into the model."""
        c = self.concepts[concept_name]
        c.observe(correct)
        self.history.append((c.narsese_log[-1], c.f, c.c))

    def belief(self, concept_name: str):
        """Return current (f, c) truth value for one concept."""
        c = self.concepts[concept_name]
        return c.f, c.c

    def knowledge_state(self):
        """Return a dict of concept → expectation value."""
        return {name: con.expectation for name, con in self.concepts.items()}

    def mastered(self, concept_name: str, threshold=0.7):
        return self.concepts[concept_name].expectation >= threshold

    def snapshot(self):
        """Full snapshot including Narsese statement per concept."""
        return {
            name: {
                "narsese":      con.current_narsese(),
                "f":            con.f,
                "c":            con.c,
                "E":            con.expectation,
                "observations": len(con.narsese_log)
            }
            for name, con in self.concepts.items()
        }

    def print_narsese_log(self, concept_name: str):
        """Print every Narsese statement fed in for one concept."""
        print(f"\nNarsese log for '{concept_name}':")
        for stmt in self.concepts[concept_name].narsese_log:
            print(f"  INPUT:  {stmt}")
        print(f"  BELIEF: {self.concepts[concept_name].current_narsese()}")
        print(f"  E = {self.concepts[concept_name].expectation:.4f}")
