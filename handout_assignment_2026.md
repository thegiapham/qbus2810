# QBUS2810 Statistical Modelling for Business
## Semester 1, 2026 — Group Assignment
### Daily Bike Sharing Demand: Explanation, Prediction and Model Discipline

| | |
|---|---|
| **Weighting** | 30% of final mark |
| **Due date** | 29 May, 2026 |
| **Submission** | Assignment on Canvas |
| **Recommended length** | 18–25 pages, excluding Python code appendix |

> This assignment must be completed in your Canvas group. You may use GenAI, but any use must be acknowledged carefully and all prompts used must be included in an appendix. Strong analysis and clear judgement matter more than mechanically produced output.

---

## General Instructions

- Use **Python** for this assignment.
- All plots, tables, model output and forecast results discussed in the report must be reproducible from your Python code.
- Include an appendix containing all Python code. A heavy penalty applies if code is missing, does not run, or does not match the submitted report.
- Submit as a professionally edited **PDF**, not a raw notebook export.
- No strict page limit, but irrelevant, repetitive or poorly edited material will reduce your mark.
- Throughout this assignment, distinguish between an **explanatory model** and an **operational prediction model**. A choice acceptable for one goal may be unsuitable for another.
- For predictive tasks, assume the forecast is made for **same-day total demand** after calendar information and daily weather expectations are available, but **before** any user counts for that day are observed.
- Not every column should automatically be treated as an ordinary predictor. Variables that are contemporaneous with the response, mechanically linked to it, or unavailable at decision time must be handled carefully. Ignoring this without justification will be penalised.

---

## Pre-analysis Instructions for Data

- Use the file `data/bike_sharing_day_2026.csv` only.
- Open `generate_train_test_split.ipynb`.
- Replace `state` with the **sum of student ID numbers** for your group members.
- Run the notebook to create train and test datasets.
- Use **train data** for: exploratory analysis, estimation, diagnostics, and model selection.
- Use **test data** only for the final forecast comparison in Task 6.
- Any out-of-sample performance measure before Task 6 must come from a validation approach within training data (e.g. cross-validation or a training-only validation split).

---

## Business Problem

Capital Bikeshare and transport planners in Washington, D.C. want to:
1. Understand the drivers of daily bike rental demand.
2. Build a model that can predict total daily rentals accurately enough to support planning, staffing and bike rebalancing decisions.

**Strategic question:** Should future planning emphasis lean more toward commuter-reliability demand, leisure/fair-weather demand, or remain broadly balanced?

---

## Data and Response Variable

- Daily observations from the Capital Bikeshare system for **2011 and 2012**, with weather and calendar information.
- Main response variable: **`cnt`** (total daily rental count). A transformation such as `log(cnt)` is acceptable if well justified.
- See `Data_Source_and_Variable_Notes.md` for source details, variable definitions and cautions.

---

## Primary Goals

1. **Explanatory:** Understand how daily bike demand is associated with weather and calendar conditions, and whether the apparent effect of temperature remains after controlling for confounders.
2. **Predictive:** Develop a parsimonious but accurate model using only variables defensible for the prediction setting.
3. **Decision:** Advise management whether planning should lean toward commuter-reliability, leisure/fair-weather, or remain balanced — being explicit about what the data can and cannot justify causally.

---

## Tasks

### Task 1 — Exploratory Analysis and Variable Admissibility *(5 marks)*

Conduct a suitable exploratory analysis relevant to the study goals.

Before fitting major models, classify variables as:
- **Clearly admissible**
- **Admissible only with caveats**
- **Inadmissible**

Must specifically comment on: `casual`, `registered`, and at least one date/calendar field.

Exploratory analysis should be **purposeful** — motivate later modelling choices rather than display many unconnected plots.

---

### Task 2 — Simple Relationship: Temperature and Total Demand *(5 marks)*

Analyse the relationship between `temp` and `cnt`. Include:
- Suitable plot(s)
- At least one simple model
- Discussion of whether a linear relationship looks plausible
- Discussion of whether model assumptions could reasonably hold
- Discussion of which omitted variables are likely to bias the simple relationship and why

If you transform the response or predictor, justify that choice.

---

### Task 3 — Explanatory Multiple Regression Model *(10 marks)*

Build an explanatory multiple linear regression model for total daily demand.

**Must include:**
- `temp`
- At least one calendar or categorical factor
- Any additional regressors needed to address confounding or important structure

Do not use variables already argued to be inadmissible.

**Discussion must cover:**
- Test for a relationship between demand and temperature
- Test for a relationship between demand and at least one categorical/calendar factor
- Whether model assumptions could hold for this data
- How well the data fit the model
- Whether multicollinearity is a problem

---

### Task 4 — Predictive Model Selection Under Admissibility Constraints *(15 marks)*

Conduct a variable and model selection exercise for predicting total daily rentals.

**Requirements — compare at least 8 admissible model specifications, including:**
- At least **2 interaction effects**
- At least **2 transformations or nonlinear effects** on regressors and/or the response
- At least **one polynomial or spline effect**

Properly motivate and discuss modelling choices. A long list of models with no reasoning will not receive full credit.

Report a concise comparison and select a preferred predictive model using **training data only**. Any out-of-sample accuracy discussed here must be cross-validated or validation-sample performance — **not** performance on the held-out test set.

**For the final preferred model:**
- Report the model clearly
- Provide suitable diagnostics
- Discuss any collinearity issues
- Clearly interpret any nonlinear effects

---

### Task 5 — Outliers, Leverage, Influence and Robustness *(5 marks)*

Investigate outliers, leverage and influence for at least one strong model on the training data.

- Identify at least **one influential day** and discuss why it may be influential.
- Perform at least **one defensible robustness check**, e.g.:
  - Alternative specification
  - Transformation
  - Carefully justified exclusion (evidence of data error)
  - Another well-motivated modelling adjustment
- Discuss whether substantive conclusions materially change.

---

### Task 6 — Forecast Evaluation on Test Data *(10 marks)*

Using at least the **3 best admissible predictive models**, generate forecasts for total daily rentals in the test dataset.

**Present:**
- A suitable summary table
- Appropriate plot(s)
- Forecast accuracy measures: **RMSE**, **MAD**, and **forecast R²**

Re-discuss conclusions in light of out-of-sample results.

> Do the best explanatory and best predictive models align? Why or why not?

---

### Task 7 — Final Report for a Decision Maker *(5 marks)*

Write a plain-English final report for a non-technical decision maker at Capital Bikeshare / Washington D.C. transport authority.

**Must:**
- Summarise the main findings
- Explain the main uncertainties and limitations
- State clearly which variables should **not** be treated as ordinary operational predictors
- Distinguish between empirical association, predictive usefulness and causal interpretation
- Provide and justify a recommendation about whether planning emphasis should lean toward commuter-reliability, leisure/fair-weather, or remain broadly balanced

---

## Marking Notes

| Component | Marks |
|---|---|
| Tasks 1–7 | 55 |
| Overall presentation quality | up to 5 bonus |
| **Total** | **60** |

- Clear, efficient and professional communication will be rewarded.
- Unclear, poorly justified or mechanically generated analysis will lose marks even if numerical output looks plausible.

---

## Data Source

UCI Bike Sharing Dataset (daily file), covering **2011-01-01 to 2012-12-31**.  
Official dataset page: https://archive-beta.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset
