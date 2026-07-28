# Student Status Prediction — Handoff & Attempt Log

## What we're trying to do
Predict a student's `status` (Dropout / Enrolled / Graduate) from `train.csv` / `test.csv`
for the Kaggle competition **"Intelligent Engineering Systems Term Project AY25-26 Term 3"**,
using a MATLAB k-NN classifier (`exp7_status_knn.m`).

The professor's instruction was to **weed out unnecessary columns** and use the
remaining ones. Our approach combines two layers of column selection:

1. **Automatic layer (fscmrmr)** — after manual exclusions are applied, the script
   one-hot encodes remaining text columns, engineers a few extra numeric features
   (approval rate, average grade, etc.), then runs MRMR feature-importance ranking
   and keeps only columns scoring above a threshold.
2. **Manual layer (`manualExclude` list in the script)** — a hard-coded list of raw
   column names removed *before* the automatic step even runs, based on either
   domain judgment (e.g. "parents' occupation shouldn't determine dropout risk")
   or evidence from testing (a column that consistently hurts accuracy when included).

Because Kaggle submissions are limited, later rounds of testing were done **locally**
in Python, using 5-fold cross-validation accuracy on `train.csv` as a fast stand-in
for the real leaderboard score, before committing to a Kaggle submission.

## Column reference (raw columns, numbered)
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

---

## Attempt Log

| # | Columns excluded (manual) | Source of score | Score | Notes |
|---|---------------------------|------------------|-------|-------|
| 1 | None (automatic fscmrmr selection only, no manual exclusion, no engineered features) | Kaggle | **0.72603** | First working submission. fscmrmr mostly picked rare/sparse one-hot dummy columns (e.g. specific nationalities, obscure occupations) rather than strong numeric predictors. |
| 2 | 9, 10, 11, 12 (mothers_qualification, fathers_qualification, mothers_occupation, fathers_occupation) | Kaggle | **0.72603** | Same score as attempt 1. |
| 3 | 3, 8, 9, 10, 20, 21, 34, 35, 36 (application_order, nationality, mothers_qualification, fathers_qualification, age_at_enrollment, international, unemployment_rate, inflation_rate, gdp) | Kaggle | **0.68625** | Worse. Suggests removing the macroeconomic columns (34-36) and/or age_at_enrollment hurt more than expected — these carry real signal. |
| 4 | Baseline: none excluded, all 36 raw columns | Local CV (proxy, not submitted) | 0.6775 | Reference point for local search — matches the general direction of attempt 1 (low without pruning). |
| 5 | 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 15, 20, 21, 27, 30 (application_mode, application_order, course, daytime_evening_attendance, previous_qualification, previous_qualification_(grade), mothers_qualification, fathers_qualification, mothers_occupation, fathers_occupation, educational_special_needs, age_at_enrollment, international, curricular_units_1st_sem_(without_evaluations), curricular_units_2nd_sem_(evaluations)) | Kaggle | **0.73236** | New best score. Local CV had predicted ~0.7524 (proxy overestimated actual by ~0.02, consistent with the proxy running a bit optimistic on this set). Confirmed as the current best submission — beats both attempt 2 (0.72603) and attempt 3 (0.68625). |

## Local CV proxy vs. actual Kaggle score (calibration check)

| Attempt | Local CV proxy | Actual Kaggle | Gap |
|---------|-----------------|----------------|-----|
| 2 (excl 9,10,11,12) | 0.7065 | 0.72603 | proxy under by 0.020 |
| 3 (excl 3,8,9,10,20,21,34,35,36) | 0.6869 | 0.68625 | proxy over by 0.001 |
| 5 (excl 15 cols, current best) | 0.7524 | 0.73236 | proxy over by 0.020 |

Takeaway: the proxy reliably gets the **ranking** right (which set is better than which),
but can be off by ~1-2 points in absolute terms — treat it as a fast filter to shortlist
promising candidates, not as an exact prediction of leaderboard score.

## Key takeaways so far
- **Helps when excluded**: application_mode, application_order, course, daytime_evening_attendance,
  previous_qualification (+ its grade), mothers/fathers qualification & occupation,
  educational_special_needs, age_at_enrollment, international, and two of the "without_evaluations"/"evaluations" curricular columns.
- **Hurts when excluded**: unemployment_rate, inflation_rate, gdp (macro columns — attempt 3 showed this clearly), nationality (kept by the search despite earlier assumption it was noise).
- k-NN neighbor count `k=17` has consistently come out as a strong choice across tests.

## Next step
Attempt 5 (0.73236) is the current best and matches what's loaded into `exp7_status_knn.m`.
Two directions to push further:
1. **Continue the local search from this point** — try a further round of backward
   elimination (and also test adding back any of the 15 excluded columns individually,
   in case one was only redundant in combination with another) to find an even
   stronger candidate before spending another Kaggle submission.
2. **Tune around the current best** — e.g. sweep `k` more finely, or try adjusting
   the `fscmrmr` threshold in the automatic layer, while keeping this exclusion set fixed.

