import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\USYD\QBUS2810\qbus2810\qbus2810_assignment.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 1: workingday effect using m_preferred ───────────────────────────────
wd_src = """# Partial effect and marginal effect of temp by working day (m_preferred)
_x_temp    = np.linspace(train['temp'].min(), train['temp'].max(), 200)
_wd_labels = {0: 'Non-working day', 1: 'Working day'}
_wd_colors = {0: '#9C27B0', 1: '#FF9800'}

_formula = m_preferred.model.formula
_is_log  = _formula.split('~')[0].strip() == 'log_cnt'

def _make_df_wd(x_temp, wd, delta=0):
    return pd.DataFrame({
        'temp':        x_temp + delta,
        'season':      3,
        'yr':          1,
        'weathersit':  1,
        'hum':         float(train['hum'].mean()),
        'windspeed':   float(train['windspeed'].mean()),
        'workingday':  wd,
        'log_cnt':     0,
        'mnth':        10,
        'log_windspeed': float(train['log_windspeed'].mean()) if 'log_windspeed' in train.columns else 0,
        'log_hum':       float(train['log_hum'].mean()) if 'log_hum' in train.columns else 0,
    })

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for wd, label in _wd_labels.items():
    _pred_cnt = m_preferred.predict(_make_df_wd(_x_temp, wd)).values
    if _is_log: _pred_cnt = np.exp(_pred_cnt)
    axes[0].plot(_x_temp, _pred_cnt, color=_wd_colors[wd], linewidth=2, label=label)

    _delta = 0.001
    _pp = m_preferred.predict(_make_df_wd(_x_temp, wd, +_delta)).values
    _pm = m_preferred.predict(_make_df_wd(_x_temp, wd, -_delta)).values
    if _is_log: _pp, _pm = np.exp(_pp), np.exp(_pm)
    axes[1].plot(_x_temp, (_pp - _pm) / (2*_delta), color=_wd_colors[wd], linewidth=2, label=label)

axes[0].set_xlabel('Normalised temperature (temp)'); axes[0].set_ylabel('Predicted cnt')
axes[0].set_title('Partial effect of temp by working day'); axes[0].legend()
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_xlabel('Normalised temperature (temp)'); axes[1].set_ylabel('d(cnt) / d(temp)')
axes[1].set_title('Marginal effect of temp on cnt by working day'); axes[1].legend()
plt.tight_layout()
plt.savefig('fig_task4_interactions_wd.png', dpi=150, bbox_inches='tight')
plt.show()"""

# ── Cell 2: season effect using m_preferred3 ─────────────────────────────────
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
plt.savefig('fig_task4_interactions_season.png', dpi=150, bbox_inches='tight')
plt.show()"""

# ── Find t4-nonlinear-plot index ──────────────────────────────────────────────
plot_idx = None
preferred_idx = None
for i, cell in enumerate(nb['cells']):
    cid = cell.get('id', '')
    if cid == 't4-nonlinear-plot':
        plot_idx = i
    if cid == 't4-preferred-fit':
        preferred_idx = i

# ── Insert two new cells after t4-nonlinear-plot ──────────────────────────────
def make_cell(src, cell_id):
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "source": src, "outputs": [], "execution_count": None}

nb['cells'].insert(plot_idx + 1, make_cell(season_src, 't4-season-plot'))
nb['cells'].insert(plot_idx + 1, make_cell(wd_src,     't4-workingday-plot'))
print(f'Inserted workingday cell at {plot_idx+1}, season cell at {plot_idx+2}')

# ── Add m_preferred3 to t4-preferred-fit if missing ──────────────────────────
src = nb['cells'][preferred_idx]['source']
if isinstance(src, list): src = ''.join(src)
if 'm_preferred3' not in src:
    addon = ("\nformula_preferred3 = ('log_cnt ~ temp*C(season) + C(mnth) + yr + "
             "C(weathersit) + hum + windspeed + workingday')\n"
             "m_preferred3 = smf.ols(formula_preferred3, data=train).fit()\n"
             "print('m_preferred3 R²:', round(m_preferred3.rsquared, 4))")
    nb['cells'][preferred_idx]['source'] = src + addon
    print('Added m_preferred3 to t4-preferred-fit')

with open(r'd:\USYD\QBUS2810\qbus2810\qbus2810_assignment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done.')
