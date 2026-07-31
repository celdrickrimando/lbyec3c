# Student Status Prediction — Handoff Document
*(Last updated: after exp10, native categorical Random Forest — not yet submitted)*

## READ THIS FIRST if you are a new Claude session
This project has an established history — do not restart from scratch or re-suggest
things already tried and rejected below. Current best submission is **0.78209**
(Random Forest with native categorical splits, `exp10_status_rf_native_cat.m`) —
**currently #1 on the leaderboard.** MATLAB ONLY — Python is not an option for the
actual deliverable (Python was used only as a fast local proxy to test ideas before
committing a Kaggle submission).

## The task
Predict a student's `status` (Dropout / Enrolled / Graduate) from `train.csv` /
`test.csv` for the Kaggle competition **"Intelligent Engineering Systems Term Project
AY25-26 Term 3"**. Submissions are MATLAB `.m` scripts producing a `submission.csv`
matching `sample_submission.csv`'s format (`stud_id`, `status`).

The professor's instruction: **weed out unnecessary columns, use the rest.**

## Column reference (raw columns, numbered — used throughout this log)
1. marital_status
2. application_mode
3. application_order
4. course
5. daytime_evening_attendance
6. previous_qualification
7. previous_qualification_(grade)
8. nationality
9. mothers_qualification
10. fathers_qualification
11. mothers_occupation
12. fathers_occupation
13. admission_grade
14. displaced
15. educational_special_needs
16. debtor
17. tuition_fees_up_to_date
18. gender
19. scholarship_holder
20. age_at_enrollment
21. international
22. curricular_units_1st_sem_(credited)
23. curricular_units_1st_sem_(enrolled)
24. curricular_units_1st_sem_(evaluations)
25. curricular_units_1st_sem_(approved)
26. curricular_units_1st_sem_(grade)
27. curricular_units_1st_sem_(without_evaluations)
28. curricular_units_2nd_sem_(credited)
29. curricular_units_2nd_sem_(enrolled)
30. curricular_units_2nd_sem_(evaluations)
31. curricular_units_2nd_sem_(approved)
32. curricular_units_2nd_sem_(grade)
33. curricular_units_2nd_sem_(without_evaluations)
34. unemployment_rate
35. inflation_rate
36. gdp

Note: MATLAB's `readtable` mangles parenthesized names, e.g.
`curricular_units_1st_sem_(grade)` becomes `curricular_units_1st_sem__grade_`
(parens → underscores). All scripts already account for this.

## Methodology used throughout
Because Kaggle submissions are limited, all tuning was done **locally in Python**
first (5-fold cross-validation accuracy as a fast proxy for the real leaderboard
score), and only promising candidates were actually submitted to Kaggle to conserve
submission count. The local proxy reliably gets *relative ranking* right (which
option beats which) but can be off by ~1-2 points in absolute value — treat it as a
filter, not an exact prediction.

---

## Scripts that exist (in order created)
- `exp7_status_knn.m` — k-NN + `fscmrmr` automatic feature selection + manual
  `manualExclude` list. **Best k-NN score: 0.73236** (15 columns excluded — see
  Attempt 5 below). Matches the assignment's literal "exclude columns" instruction
  most directly, but has a real accuracy ceiling (see k-NN section below).
- `exp8_status_rf.m` — Random Forest via `fitcensemble('Method','Bag')`, one-hot
  encoded categoricals. **CURRENT BEST: 0.77848.** Uses all 36 raw columns (RF
  barely benefits from exclusion — see below).
- `exp9_status_gb.m` — Boosting via `fitcensemble('Method','AdaBoostM2')` (MATLAB
  has no true multiclass gradient boosting; `LogitBoost` only supports binary and
  errors on 3 classes). **Scored 0.76311 — worse than RF. Do not pursue further.**
- `exp10_status_rf_native_cat.m` — Random Forest but passing categorical columns to
  `fitcensemble` **natively** (as MATLAB `categorical` type in the table) instead of
  manually one-hot encoding into ~250 sparse dummy columns. Also uses
  `PredictorSelection='interaction-curvature'` to reduce bias toward
  high-cardinality columns. **CURRENT BEST: 0.78209 — #1 on the leaderboard.**

## Full attempt log (all real Kaggle scores)

| # | Script | Setup | Kaggle Score | Notes |
|---|--------|-------|--------------|-------|
| 1 | exp7 (early version) | Automatic fscmrmr only, no manual exclusion, no engineered features | 0.72603 | First working submission |
| 2 | exp7 | Exclude 9,10,11,12 | 0.72603 | Same as #1 |
| 3 | exp7 | Exclude 3,8,9,10,20,21,34,35,36 | 0.68625 | Worse — macro columns (34-36) hurt when removed |
| 5 | exp7 | Exclude 2,3,4,5,6,7,9,10,11,12,15,20,21,27,30 (15 cols) | **0.73236** | Best k-NN result |
| 6 | exp7 | Attempt 5 + exclude 8 (nationality) | 0.72965 | Worse — nationality should stay |
| — | exp8 | RF, all 36 cols, `MinLeafSize=2`, `MaxNumSplits=200` | **0.77848** | New best, big jump over k-NN |
| — | exp8 | Same + `NumVariablesToSample=sqrt(nFeatures)` (true RF, not just bagging) | 0.77848 | Identical to 5 decimals — suspicious, never fully resolved whether the tweak actually took effect (see "Unresolved" below) |
| — | exp8 | Same + manual exclude 9, 27 (RF-specific search) | 0.77576 | Slightly worse than unpruned RF — inconclusive whether this exact config was cleanly tested (permission-denied file error occurred once mid-testing; a rerun gave 0.7743 local CV, actual Kaggle 0.77576) |
| — | exp9 | AdaBoostM2, all 36 cols | 0.76311 | Worse than RF. Do not pursue MATLAB boosting further — the algorithm ceiling is below RF here |
| — | exp10 | RF, native categorical splits (not one-hot), `interaction-curvature` predictor selection | **0.78209** | **NEW BEST — #1 on leaderboard.** Confirms native categorical handling beats one-hot encoding for this dataset in MATLAB. |

**Current best: 0.78209 (Random Forest, exp10, native categorical splits — no
manual one-hot encoding). Currently #1 on the leaderboard.**

## Local-only exploration (Python proxy, not submitted — for context on what NOT to re-try)

### k-NN: extensively tuned, hit a real ceiling (~0.76 local / ~0.73 Kaggle)
Tried and all **failed to beat** the one-hot + 15-column-exclusion baseline (0.7619-0.7637 local):
- Target encoding instead of one-hot (0.734)
- Down-weighting categorical dummy columns in the distance metric (0.755)
- PCA dimensionality reduction before k-NN (0.761)
- Alternative distance metrics: Manhattan, Chebyshev, cosine (0.753-0.762)
- Wider k range 25→301 (monotonically worse as k increases; k=17 is a genuine peak)
- Greedy exclusion search: only marginal further gains, and the one promising triple
  (removing nationality among others) was contradicted by a real Kaggle test showing
  nationality removal hurts — the local proxy has known noise on small-sample
  interaction effects, don't trust triples/pairs blindly, verify on Kaggle.

**Conclusion: k-NN's distance-based approach is a poor fit for this data — the real
signal is threshold-like (e.g. "did the student pass most units"), which trees
capture naturally and distance metrics don't. Do not keep tuning k-NN hyperparameters
expecting to beat ~0.73-0.75; that avenue is exhausted.**

### Tree/ensemble models compared locally (Python sklearn, for direction-finding only)
| Model | Local CV accuracy |
|---|---|
| k-NN (tuned) | 0.76 |
| Random Forest (tuned) | 0.775-0.785 |
| Extra Trees | 0.778 |
| HistGradientBoosting | 0.764 |
| AdaBoost (proxy for MATLAB's AdaBoostM2) | 0.775 max |
| **True Gradient Boosting (sklearn)** | **0.783** |
| GB + Extra Trees soft-voting ensemble | **0.786 (best found, Python-only)** |

**Critical limitation: MATLAB's `fitcensemble` has no true multiclass gradient
boosting and no Extra-Trees equivalent.** The 0.786 combo is not reproducible in
MATLAB. `AdaBoostM2` was tested as the closest available substitute and scored worse
than RF on the actual leaderboard (0.76311) — don't re-attempt MATLAB boosting
methods expecting to beat RF.

### Other ideas discussed and dismissed (with reasoning, don't re-raise without new info)
- **Mixed effects regression (GLME)**: MATLAB's `fitglme` doesn't support multinomial
  (3-class) outcomes, only binomial/Poisson/normal/gamma. Would require collapsing to
  binary target, changing the problem scope. Not pursued.
- **SARIMAX-GARCH**: Not applicable — this is cross-sectional data (independent
  students), not a time series. No genuine time ordering exists in the data.
- **Rolling-origin CV**: Data does cluster into 10 distinct macroeconomic snapshots
  (unique combos of unemployment/inflation/gdp, ~260-420 students each) that could
  stand in for "cohorts," but their true chronological order is unknown, so genuine
  rolling-origin validation isn't implementable without more info. A "leave-cohort-out"
  robustness check was proposed but never executed — could still be worth doing.
- **Neural network (patternnet)**: Discussed as plausible but likely to underperform
  RF given the dataset's modest size (~3300 rows) — tabular data generally favors
  tree ensembles over NNs at this scale. Never actually tested locally or in MATLAB.

## Unresolved items (pick up here)
1. **exp10 (native categorical RF) is now the confirmed best (0.78209, #1 on
   leaderboard).** Worth exploring further in this direction — e.g. tuning
   `MinLeafSize`/`MaxNumSplits`/tree count specifically for the native-categorical
   version (all tuning so far was done on the one-hot version before this approach
   existed, so the current hyperparameters may not be optimal for this data
   representation).
2. **The "0.77848 twice, identical to 5 decimals" mystery (exp8, one-hot version)
   was never resolved**, but is now less important since exp10 has superseded exp8
   as the best approach. Low priority unless returning to one-hot encoding for some
   reason.
3. **Leave-cohort-out validation** (see below) — proposed, never executed. Could
   reveal whether the model generalizes across the 10 economic-snapshot cohorts, as a
   robustness check independent of the score-chasing.
4. **Now in 1st place** — future tuning is about protecting/extending the lead
   rather than catching up. Small, careful tweaks to exp10 (tree count, leaf size,
   `MaxNumSplits`) are the most promising next lever, since this exact configuration
   hasn't been hyperparameter-tuned yet — the values used were carried over from the
   one-hot RF experiments, not re-optimized for native categorical splits.

## Update — exp11 attempt: blended ensemble (RF + LDA + ECOC-SVM), not yet submitted
User asked to push past 0.80. Before writing MATLAB, re-verified the ceiling with a
fresh Python proxy pass (not just trusting the old numbers), including one new idea
not in the log above: a **manually blended soft-voting ensemble of three
MATLAB-reproducible models** (Random Forest, LDA, ECOC-SVM), since MATLAB has no true
gradient boosting or Extra Trees to reach the 0.786 Python-only combo.

Python proxy results (5-fold CV, ordinal/one-hot encoded — NOT the same encoding as
the real native-categorical MATLAB RF, so treat as directional only):
- RF alone (proxy): 0.7709–0.7734 depending on hyperparams — hyperparameter grid
  (`n_estimators` 300/600, `min_samples_leaf` 1/2/4, `max_features` sqrt/None) moved
  this by less than 0.005 in either direction. **Confirms the log's earlier finding:
  RF hyperparameters are already near a plateau, more tuning won't move the needle.**
- LDA alone (proxy, one-hot + standardized numeric): 0.7640
- SVM (RBF) alone (proxy, one-hot + standardized numeric): 0.7670
- Naive Bayes: 0.242 — unusable with one-hot input, discarded.
- **Weighted blend RF+LDA (0.7/0.3): 0.7791** — best single lever found, roughly
  +0.008–0.01 over RF alone in this proxy.
- Stacking (logistic-regression meta-learner on out-of-fold RF+LDA+SVM
  probabilities): 0.7776 — no better than the simple weighted blend, so not worth the
  extra complexity.

**Wrote `exp11_status_blend.m`**: keeps exp10's native-categorical RF as-is, adds
`fitcdiscr` (pseudoLinear, for rank-deficient one-hot data) and `fitcecoc`
(SVM learners, RBF kernel, `FitPosterior=true`) trained on a one-hot + standardized
numeric representation, tunes blend weights on 5-fold out-of-fold posterior
probabilities inside the script itself, then retrains all three on the full training
set and blends their test-set posteriors with the tuned weights. **Not yet submitted
to Kaggle — next session should submit it and log the real score here.**

**Expectation, stated plainly:** the Python proxy suggests this blend could move the
real Kaggle score from ~0.782 to perhaps **~0.785–0.79**, based on the proxy's own
+0.008-0.01 blend-over-RF gap. There is no evidence in any experiment run so far
(Python or MATLAB, RF/GB/ET/k-NN/blends/stacking) that **0.80 is reachable with this
feature set** — every avenue tried tops out in the high 0.77–0.79 range. If exp11
doesn't clear 0.80 either, the honest read is that 0.80+ would need something outside
what's been tried yet (e.g., genuinely new features from the raw data, not just
re-combining/re-weighting the same 36 columns and existing engineered ratios) rather
than a different model or blend of the same signal.

## Realistic expectations going forward
Every standard classifier tested (in Python, as a proxy) tops out in the 0.76-0.79
local CV range on this dataset with this feature set. **0.80-0.85 has not been shown
achievable with any single model tried so far, in MATLAB or otherwise.** If the next
session is asked to "get to 0.80-0.85," be upfront about this ceiling rather than
promising it — the realistic next steps are the native-categorical RF (exp10),
possibly a bit more RF hyperparameter tuning, or accepting that ~0.78 is a strong,
competitive, honestly-earned result (it already beats the leaderboard's non-leading
positions and is within ~0.002 of first place).
