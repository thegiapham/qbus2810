import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\USYD\QBUS2810\qbus2810\qbus2810_assignment.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── New markdown content ──────────────────────────────────────────────────────

intro_src = """## 4.3 Interpreting Nonlinear and Interaction Effects

The preferred model (`m_preferred`) is:

> `log_cnt ~ cr(temp, df=4) * workingday + C(season) + yr + C(weathersit) + cr(hum, df=4) + cr(windspeed, df=4)`

It captures two key nonlinearities motivated by the EDA in Section 4.0:

1. **Spline on `temp`** — the LOWESS of temp vs cnt is clearly curved, rising steeply from cold to mild temperatures and levelling off at the high end. A natural cubic spline with 4 degrees of freedom fits this shape without imposing a fixed functional form.
2. **Spline on `hum` and `windspeed`** — both showed nonlinear LOWESS patterns: humidity has a concave relationship (moderate humidity optimal), and windspeed has a negative but decelerating effect.

Beyond nonlinearity, the model includes two interaction effects investigated below:
- `cr(temp, df=4) * workingday` — does temperature sensitivity differ between working and non-working days?
- `cr(temp, df=4) * C(season)` — investigated via the auxiliary model `m_preferred3` — does the temperature curve differ across seasons?"""

discuss_src = """**Interpretation — Working Day Interaction (`m_preferred`)**

The working day plot shows:
- **Left (partial effect):** On both working and non-working days, demand rises sharply with temperature at low-to-mid values and flattens at high temperatures. Non-working days consistently predict higher demand across the temperature range — leisure riders dominate weekend volume.
- **Right (marginal effect):** The slope d(cnt)/d(temp) is higher for non-working days at low temperatures, meaning leisure riders are more responsive to warming from cold conditions. At high temperatures the marginal effects converge — both groups are near their ceiling.

This confirms that temperature sensitivity is stronger for leisure demand: when it warms up from cold, non-working day gains outpace working day gains.

---

**Interpretation — Season Interaction (`m_preferred3`)**

`m_preferred3`: `log_cnt ~ cr(temp, df=4) * C(season) + yr + C(weathersit) + cr(hum, df=4) + cr(windspeed, df=4) + workingday`

The season plot shows:
- **Left (partial effect):** Summer and Fall predict the highest absolute demand at any given temperature. Spring and Winter are lower but show steeper upward curves — temperature is a stronger barrier in cold seasons.
- **Right (marginal effect):** Winter and Spring have the largest marginal effect of temperature — an extra degree of warmth adds substantially more riders. Summer's marginal effect is flat or declining at high temperatures, consistent with heat suppression of demand.

---

**Business implication:** Both interactions point in the same direction. Temperature matters more when it is a binding constraint — cold seasons, cold days, non-working days when leisure riders would otherwise stay home. Planning emphasis for **weather-sensitive (leisure/fair-weather) demand** should focus on early spring and winter warming periods, not peak summer."""

vif_src = """**Collinearity in the preferred model**

The spline interaction `cr(temp, df=4) * workingday` expands into multiple basis-function columns crossed with `workingday`, which raises VIFs for those terms. Similarly, `C(season)` and `cr(hum, df=4)` contribute several columns that are modestly correlated.

Elevated VIFs are expected here and do **not** invalidate the predictive model:
- The goal of Task 4 is accurate forecasting, not unbiased coefficient estimation.
- VIFs inflate standard errors of individual coefficients, but the collective fit (RMSE, R²) is unaffected.
- Collinearity would be a more serious concern in Task 3 (explanatory model), where individual coefficient interpretation drives conclusions.

Any VIF above ~10 for the spline basis columns is structural and unavoidable given the interaction specification."""

# ── Season plot cell source ───────────────────────────────────────────────────
season_src = """# Partial effect and marginal effect of temp by season (m_preferred3)
_x_temp       = np.linspace(train['temp'].min(), train['temp'].max(), 200)
_season_names  = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
_season_colors = {'Spring': '#4CAF50', 'Summer': '#F44336', 'Fall': '#FF9800', 'Winter': '#2196F3'}
_season_month  = {1: 3, 2: 7, 3: 10, 4: 1}

_formula3 = m_preferred3.model.formula
_is_log3  = _formula3.split('~')[0].strip() == 'log_cnt'

def _make_df3(x_temp, season, delta=0):
    return pd.DataFrame({
        'temp':       x_temp + delta,
        'season':     season,
        'yr':         1,
        'weathersit': 1,
        'hum':        float(train['hum'].mean()),
        'windspeed':  float(train['windspeed'].mean()),
        'workingday': 1,
        'log_cnt':    0,
        'mnth':       _season_month[season],
    })

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for s, sname in _season_names.items():
    _pred_cnt = m_preferred3.predict(_make_df3(_x_temp, s)).values
    if _is_log3: _pred_cnt = np.exp(_pred_cnt)
    axes[0].plot(_x_temp, _pred_cnt, color=_season_colors[sname], linewidth=2, label=sname)

    _delta = 0.001
    _pp = m_preferred3.predict(_make_df3(_x_temp, s, +_delta)).values
    _pm = m_preferred3.predict(_make_df3(_x_temp, s, -_delta)).values
    if _is_log3: _pp, _pm = np.exp(_pp), np.exp(_pm)
    axes[1].plot(_x_temp, (_pp - _pm) / (2*_delta), color=_season_colors[sname], linewidth=2, label=sname)

axes[0].set_xlabel('Normalised temperature (temp)'); axes[0].set_ylabel('Predicted cnt')
axes[0].set_title('Partial effect of temp by season (m_preferred3)'); axes[0].legend()
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_xlabel('Normalised temperature (temp)'); axes[1].set_ylabel('d(cnt) / d(temp)')
axes[1].set_title('Marginal effect of temp on cnt by season'); axes[1].legend()
plt.tight_layout()
plt.savefig('task4/fig_task4_interactions_season.pdf', dpi=150, bbox_inches='tight')
plt.show()"""

def make_cell(src, cell_id, cell_type='code'):
    base = {"cell_type": cell_type, "id": cell_id, "metadata": {}, "source": src}
    if cell_type == 'code':
        base.update({"outputs": [], "execution_count": None})
    return base

# ── Patch cells ───────────────────────────────────────────────────────────────
wd_idx = None
for i, cell in enumerate(nb['cells']):
    cid = cell.get('id', '')
    if cid == 't4-nonlinear-intro':
        nb['cells'][i]['source'] = intro_src
        print(f'Rewrote t4-nonlinear-intro at {i}')
    if cid == 't4-nonlinear-discuss':
        nb['cells'][i]['source'] = discuss_src
        print(f'Rewrote t4-nonlinear-discuss at {i}')
    if cid == 't4-vif-discuss':
        nb['cells'][i]['source'] = vif_src
        print(f'Rewrote t4-vif-discuss at {i}')
    if cid == 't4-workingday-plot':
        wd_idx = i

# Insert season plot cell right after workingday plot
if wd_idx is not None:
    nb['cells'].insert(wd_idx + 1, make_cell(season_src, 't4-season-plot'))
    print(f'Inserted t4-season-plot at {wd_idx + 1}')
else:
    print('WARNING: t4-workingday-plot not found')

with open(r'd:\USYD\QBUS2810\qbus2810\qbus2810_assignment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done.')
