import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\USYD\QBUS2810\qbus2810\qbus2810_assignment.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

new_src = """## 4.1 Candidate Models

We compare **16 admissible predictive model specifications**, progressing from a simple baseline through transformations, interactions, and flexible spline terms. The forecast setting assumes calendar information and daily weather expectations are available before the day's rentals are observed — so `casual`, `registered`, and `instant` are inadmissible.

All continuous predictors (`temp`, `hum`, `windspeed`) showed nonlinear LOWESS relationships with `cnt` in Section 4.0, motivating spline and polynomial terms. Season and working-day subgroups showed diverging temperature sensitivities, motivating interaction effects.

---

### Model Groups

**Group A — Baseline and response transformation**

| Model | Formula | Key feature |
|---|---|---|
| M1 | `cnt ~ temp + C(season) + yr + C(weathersit) + hum + windspeed + workingday` | Baseline (replicates Task 3) |
| M2 | `log_cnt ~ temp + C(season) + yr + C(weathersit) + hum + windspeed + workingday` | Log response |

M1 establishes the benchmark. M2 tests whether a log transformation stabilises variance — motivated by the right-skewed distribution of `cnt`.

**Group B — Polynomial and calendar granularity**

| Model | Formula | Key feature |
|---|---|---|
| M3 | `cnt ~ temp + temp² + C(season) + yr + C(weathersit) + hum + windspeed + workingday` | Quadratic temp |
| M4 | `cnt ~ temp + C(mnth) + yr + C(weathersit) + hum + windspeed + workingday` | Month dummies instead of season |
| M7 | `log_cnt ~ temp + temp² + C(season) + yr + C(weathersit) + hum + windspeed + workingday` | Log response + quadratic temp |

M3 and M7 address the nonlinear LOWESS relationship between `temp` and `cnt` with a polynomial. M4 replaces the 4-level `season` with 12 monthly dummies for finer seasonal resolution.

**Group C — Interaction effects**

| Model | Formula | Key feature |
|---|---|---|
| M5 | `cnt ~ temp*workingday + C(season) + yr + C(weathersit) + hum + windspeed` | Interaction: temp × workingday |
| M6 | `cnt ~ temp*C(season) + yr + C(weathersit) + hum + windspeed + workingday` | Interaction: temp × season |
| M9 | `log_cnt ~ temp*C(season) + C(mnth) + yr + C(weathersit) + hum + windspeed + workingday` | Log + temp × season + month |

M5 tests whether temperature sensitivity differs between commuter and leisure demand. M6 allows each season to have its own temperature slope. M9 combines both: log response, season interaction, and monthly granularity.

**Group D — Natural cubic splines on multiple predictors**

| Model | Formula | Key feature |
|---|---|---|
| M8  | `cnt ~ cr(temp,4) + C(season) + yr + C(weathersit) + hum + windspeed + workingday` | Spline on temp only |
| M10 | `cnt ~ cr(temp,4) + C(season) + yr + C(weathersit) + cr(hum,4) + cr(windspeed,4) + workingday` | Splines on temp, hum, windspeed |
| M11 | `cnt ~ cr(temp,4) + C(season) + yr + C(weathersit) + log(hum) + cr(windspeed,4) + workingday` | Spline temp/windspeed, log hum |
| M12 | `cnt ~ cr(temp,4) + C(season) + yr + C(weathersit) + cr(hum,4) + log(windspeed) + workingday` | Spline temp/hum, log windspeed |

M8 extends the spline idea from temp to all three continuous weather predictors in M10–M12, testing alternative transformations for `hum` and `windspeed`.

**Group E — Splines combined with interactions**

| Model | Formula | Key feature |
|---|---|---|
| M13 | `cnt ~ cr(temp,4)*C(season) + yr + C(weathersit) + cr(hum,4) + cr(windspeed,4) + workingday` | Spline temp × season |
| M14 | `cnt ~ cr(temp,4)*workingday + C(season) + yr + C(weathersit) + cr(hum,4) + cr(windspeed,4)` | Spline temp × workingday |
| M15 | `log_cnt ~ cr(temp,4)*workingday + C(season) + yr + C(weathersit) + cr(hum,4) + cr(windspeed,4)` | Log + spline temp × workingday |
| M16 | `log_cnt ~ cr(temp,4)*C(season) + yr + C(weathersit) + cr(hum,4) + cr(windspeed,4) + workingday` | Log + spline temp × season |

M13–M16 are the most flexible specifications, allowing the nonlinear temp curve to differ by season or working day, with and without log response.

---

### Requirements satisfied

| Requirement | Models |
|---|---|
| ≥ 2 interaction effects | M5 (temp×workingday), M6/M9/M13/M16 (temp×season), M14/M15 (spline×workingday) |
| ≥ 2 transformations | M2/M7/M9/M15/M16 (log response), M3/M7 (temp²), M11 (log hum), M12 (log windspeed) |
| ≥ 1 polynomial or spline | M3/M7 (quadratic polynomial), M8–M16 (natural cubic spline `cr()`) |"""

for i, cell in enumerate(nb['cells']):
    if cell.get('id') == 't4-intro':
        nb['cells'][i]['source'] = new_src
        print(f'Updated t4-intro at cell {i}')
        break

with open(r'd:\USYD\QBUS2810\qbus2810\qbus2810_assignment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done.')
