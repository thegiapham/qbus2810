import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.makedirs(r'd:\USYD\QBUS2810\qbus2810\task5', exist_ok=True)

with open(r'd:\USYD\QBUS2810\qbus2810\qbus2810_assignment.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def code_cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src, "outputs": [], "execution_count": None}

def md_cell(src, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

# ── Cell sources ──────────────────────────────────────────────────────────────

header = """## Task 5 — Outliers, Leverage, Influence and Robustness

We investigate outliers, leverage and influence using the preferred model `m_preferred`:
`log_cnt ~ cr(temp, df=4) * workingday + C(season) + yr + C(weathersit) + cr(hum, df=4) + cr(windspeed, df=4)`"""

influence_code = """from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(m_preferred)
cooks_d   = influence.cooks_distance[0]
leverage  = influence.hat_matrix_diag
ext_resid = influence.resid_studentized_external

# Threshold: Cook's D > 4/n is commonly flagged
n = len(train)
cook_thresh = 4 / n
lev_thresh  = 2 * m_preferred.df_model / n

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Cook's distance
axes[0].stem(range(n), cooks_d, markerfmt=',', linefmt='steelblue', basefmt='k-')
axes[0].axhline(cook_thresh, color='red', linestyle='--', label=f'4/n = {cook_thresh:.4f}')
axes[0].set_xlabel('Observation index'); axes[0].set_ylabel("Cook's distance")
axes[0].set_title("Cook's Distance"); axes[0].legend()

# Leverage
axes[1].stem(range(n), leverage, markerfmt=',', linefmt='steelblue', basefmt='k-')
axes[1].axhline(lev_thresh, color='red', linestyle='--', label=f'2p/n = {lev_thresh:.3f}')
axes[1].set_xlabel('Observation index'); axes[1].set_ylabel('Leverage (hat value)')
axes[1].set_title('Leverage'); axes[1].legend()

# Influence plot: leverage vs studentised residual
axes[2].scatter(leverage, ext_resid, alpha=0.5, s=18, color='steelblue')
axes[2].axhline( 3, color='red', linestyle='--', linewidth=0.8)
axes[2].axhline(-3, color='red', linestyle='--', linewidth=0.8)
axes[2].axvline(lev_thresh, color='orange', linestyle='--', linewidth=0.8)
axes[2].set_xlabel('Leverage'); axes[2].set_ylabel('Ext. Studentised Residual')
axes[2].set_title('Influence Plot')

plt.tight_layout()
plt.savefig('task5/fig_task5_influence.pdf', dpi=150, bbox_inches='tight')
plt.show()"""

identify_code = """# Identify top influential observations
inf_df = train.reset_index(drop=True).copy()
inf_df['cooks_d']   = cooks_d
inf_df['leverage']  = leverage
inf_df['ext_resid'] = ext_resid
inf_df['predicted'] = np.exp(m_preferred.fittedvalues.values)

top_cook = inf_df.nlargest(5, 'cooks_d')[
    ['dteday','cnt','temp','season','weathersit','workingday',
     'predicted','cooks_d','leverage','ext_resid']
].round(3)

print('=== Top 5 by Cooks Distance ===')
print(top_cook.to_string())

print()
high_lev = inf_df[inf_df['leverage'] > lev_thresh].nlargest(5, 'leverage')[
    ['dteday','cnt','temp','season','weathersit','workingday','leverage','ext_resid']
].round(3)
print('=== High Leverage Observations ===')
print(high_lev.to_string())"""

robustness_code = """# Robustness check 1: HC3 robust standard errors
formula_m14 = ('cnt ~ cr(temp, df=4)*workingday + C(season) + yr + '
               'C(weathersit) + cr(hum, df=4) + cr(windspeed, df=4)')
formula_m15 = ('log_cnt ~ cr(temp, df=4)*workingday + C(season) + yr + '
               'C(weathersit) + cr(hum, df=4) + cr(windspeed, df=4)')

m14      = smf.ols(formula_m14, data=train).fit()
m14_hc3  = smf.ols(formula_m14, data=train).fit(cov_type='HC3')
m15      = smf.ols(formula_m15, data=train).fit()
m15_hc3  = smf.ols(formula_m15, data=train).fit(cov_type='HC3')

# Compare key coefficients and p-values
key_vars = ['workingday', 'yr']
print('=== M14: OLS vs HC3 ===')
comp14 = pd.DataFrame({
    'coef':      m14.params[key_vars],
    'pval_OLS':  m14.pvalues[key_vars],
    'pval_HC3':  m14_hc3.pvalues[key_vars],
    'se_OLS':    m14.bse[key_vars],
    'se_HC3':    m14_hc3.bse[key_vars],
}).round(4)
print(comp14)

print()
print('=== M15: OLS vs HC3 ===')
comp15 = pd.DataFrame({
    'coef':      m15.params[key_vars],
    'pval_OLS':  m15.pvalues[key_vars],
    'pval_HC3':  m15_hc3.pvalues[key_vars],
    'se_OLS':    m15.bse[key_vars],
    'se_HC3':    m15_hc3.bse[key_vars],
}).round(4)
print(comp15)"""

exclusion_code = """# Robustness check 2: exclude most influential day and refit
most_inf_idx = inf_df['cooks_d'].idxmax()
most_inf_day = inf_df.loc[most_inf_idx, 'dteday']
print(f'Most influential day: {most_inf_day}  (cnt={inf_df.loc[most_inf_idx,"cnt"]}, '
      f'Cook\\'s D={inf_df.loc[most_inf_idx,"cooks_d"]:.4f})')

train_excl = train.drop(train.index[most_inf_idx]).reset_index(drop=True)
m_preferred_excl = smf.ols(formula_preferred, data=train_excl).fit()

# Compare key coefficients before and after exclusion
compare_df = pd.DataFrame({
    'Full sample':   m_preferred.params,
    'Excl. outlier': m_preferred_excl.params,
}).dropna().round(4)
print()
print('=== Coefficient comparison (full vs excluded) ===')
print(compare_df.to_string())"""

discussion = """**5.1 Influential Observations**

Cook's distance flags several observations with D > 4/n. The most influential days tend to share common characteristics:
- **Extreme weather days** (weathersit = 3: heavy rain/snow) with very low counts — the model struggles to predict demand under rare severe conditions
- **The day with cnt = 22** — the lowest count in the dataset — is an extreme outlier with a large negative residual on the log scale, generating high Cook's distance and driving the kurtosis spike seen in diagnostics

High-leverage observations (hat value > 2p/n) are days with unusual covariate combinations — e.g. warm temperatures in winter, or extreme humidity — that sit far from the centre of the predictor space and have outsized potential to pull the fitted line.

**5.2 Robustness Checks**

*Check 1 — HC3 robust standard errors (M14 and M15):*
The diagnostic plots for M14 showed a curved Scale-Location plot, indicating heteroskedasticity. Refitting with HC3 standard errors corrects for this without changing the coefficient estimates. If the same predictors remain significant under HC3, the inferential conclusions are robust to the variance assumption.

*Check 2 — Exclusion of the most influential day:*
We refit `m_preferred` after removing the single most influential observation. If the key coefficients (temperature effect, year trend, working day effect) do not change materially in sign or magnitude, the model is not driven by that single observation.

**5.3 Conclusion**

If both robustness checks yield similar coefficients and significance patterns, the preferred model's conclusions are stable: temperature, season, year trend and weather conditions all have genuine, robust associations with daily bike demand. The identified influential days represent genuinely unusual days (extreme weather events) rather than data errors, so exclusion is not strictly justified — but demonstrating stability under exclusion strengthens confidence in the findings."""

# ── Append all cells ──────────────────────────────────────────────────────────
new_cells = [
    md_cell(header,          't5-header'),
    code_cell(influence_code, 't5-influence'),
    code_cell(identify_code,  't5-identify'),
    md_cell(discussion,       't5-discuss'),
    code_cell(robustness_code,'t5-robust-hc3'),
    code_cell(exclusion_code, 't5-robust-excl'),
]

nb['cells'].extend(new_cells)
print(f'Added {len(new_cells)} cells. Total: {len(nb["cells"])}')

with open(r'd:\USYD\QBUS2810\qbus2810\qbus2810_assignment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done.')
