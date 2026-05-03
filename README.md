# NARS Student Modeling

Adaptive student knowledge tracking using Non-Axiomatic Logic (NAL) belief revision, compared against BKT and a rule-based baseline.

## What this is

Standard student models like Bayesian Knowledge Tracing (BKT) work fine when a student is consistently right or consistently wrong. They break down when a student has a *misconception*  they get some questions right but fail on specific problem types. This project tests whether NAL-based belief revision can handle that case better.

The core idea: every concept gets a belief `<f, c>` where `f` is how often the student answers correctly and `c` is how much evidence we have. When a new answer comes in, the NAL revision rule merges old and new evidence mathematically, without needing to retrain or reset anything.

## Files

| File | What it does |
|------|-------------|
| `nars_engine.py` | NAL truth functions + student model built on belief revision |
| `student_simulator.py` | Three student profiles with fixed ground truth, generates quiz answers |
| `bkt_baseline.py` | Standard 4-parameter BKT per concept |
| `rule_based_baseline.py` | Exponential moving average tracker |
| `evaluation.py` | MAE, convergence speed, misconception accuracy, belief stability |
| `run_experiment.py` | Runs everything, saves results.json and four figures |

## How to run

```bash
pip install numpy matplotlib
python3 run_experiment.py
```

Output goes into a `results/` folder: `results.json` + four `.png` figures.

## Results summary

NARS outperforms both baselines on MAE and misconception detection across all three student profiles. The biggest gap is on the misconception student — NARS correctly identifies which concepts the student knows vs. doesn't, while BKT's floor/ceiling parameters prevent it from tracking the asymmetric pattern.

| Model | Strong MAE | Struggling MAE | Misconception accuracy |
|-------|-----------|----------------|----------------------|
| NARS | 0.043 | 0.079 | 1.00 |
| BKT | 0.095 | 0.255 | 0.83 |
| Rule-Based | 0.097 | 0.211 | 0.83 |

## Background

- Wang, P. (2013). *Non-Axiomatic Logic: A Model of Intelligent Reasoning*. World Scientific.
- Corbett, A. & Anderson, J. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278.
