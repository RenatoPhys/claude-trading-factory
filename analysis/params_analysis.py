"""
analysis/params_analysis.py
=============================
Análise de parâmetros: quais ranges geram melhor performance por estratégia.
Compara distribuição de params nos top 25% vs bottom 25% de Sharpe.

Uso:
    python analysis/params_analysis.py
    python analysis/params_analysis.py --strategy pattern_rsi_trend
    python analysis/params_analysis.py --min-trades 200
"""

import json
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).parent
ROOT         = ANALYSIS_DIR.parent
LOG_PATH     = ROOT / 'backtest_log.csv'
OUT_DIR      = ANALYSIS_DIR / 'output'
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Análise de parâmetros por estratégia')
parser.add_argument('--strategy',   type=str, default=None, help='Filtrar por estratégia')
parser.add_argument('--min-trades', type=int, default=100,  help='Minimo de trades (default 100)')
args = parser.parse_args()

MIN_TRADES = args.min_trades

# ---------------------------------------------------------------------------
# Params numéricos por estratégia
# ---------------------------------------------------------------------------
STRATEGY_PARAMS = {
    'gold_rsi_trend'           : ['sl', 'tp', 'length_rsi', 'rsi_low', 'rsi_high'],
    'pattern_rsi_trend'        : ['sl', 'tp', 'length_rsi', 'rsi_low', 'rsi_high'],
    'pattern_rsi_anti_trend'   : ['sl', 'tp', 'length_rsi', 'rsi_low', 'rsi_high'],
    'bb_trend'                 : ['sl', 'tp', 'bb_length', 'std'],
    'bb_anti_trend'            : ['sl', 'tp', 'bb_length', 'std'],
    'macd_crossover_trend'     : ['sl', 'tp', 'fast_period', 'slow_period', 'signal_period'],
    'macd_crossover_anti_trend': ['sl', 'tp', 'fast_period', 'slow_period', 'signal_period'],
    'momentum_breakout'        : ['sl', 'tp', 'lookback_period', 'momentum_threshold', 'volume_factor'],
}

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------
df = pd.read_csv(LOG_PATH)
df = df[df['total_trades'] >= MIN_TRADES].copy()
df['params_dict'] = df['params'].apply(json.loads)

# Extrai todos os params numéricos como colunas
all_param_keys = set()
for d in df['params_dict']:
    all_param_keys.update(k for k, v in d.items() if isinstance(v, (int, float)) and k != 'allowed_hours')

for key in all_param_keys:
    df[f'p_{key}'] = df['params_dict'].apply(lambda d: d.get(key))

strategies = sorted(df['strategy'].unique())
if args.strategy:
    strategies = [s for s in strategies if args.strategy in s]
    if not strategies:
        print(f'Estratégia "{args.strategy}" não encontrada. Opções: {sorted(df["strategy"].unique())}')
        exit(1)

SEP = '=' * 100

# ---------------------------------------------------------------------------
# Para cada estratégia: comparar params top vs bot quartil
# ---------------------------------------------------------------------------
for strat in strategies:
    df_s = df[df['strategy'] == strat].copy()
    if len(df_s) < 8:
        continue

    params_cols = STRATEGY_PARAMS.get(strat, [])
    n_q = max(1, int(len(df_s) * 0.25))
    top_q = df_s.nlargest(n_q, 'sharpe_ratio')
    bot_q = df_s.nsmallest(n_q, 'sharpe_ratio')

    print(f'\n{SEP}')
    print(f'  ESTRATÉGIA: {strat}  ({len(df_s)} runs | top/bot quartil = {n_q} runs cada)')
    print(SEP)

    # Correlações numéricas com Sharpe
    print(f'\n  Correlação (Pearson) dos parâmetros com Sharpe:')
    print(f'  {"Parâmetro":<22}  {"Corr c/ Sharpe":>14}  {"Corr c/ Sortino":>15}  {"Corr c/ Retorno":>15}')
    print(f'  {"-"*22}  {"-"*14}  {"-"*15}  {"-"*15}')
    corr_rows = []
    for p in params_cols:
        col = f'p_{p}'
        if col not in df_s.columns or df_s[col].isna().all():
            continue
        valid = df_s[[col, 'sharpe_ratio', 'sortino_ratio', 'total_return']].dropna()
        if len(valid) < 5:
            continue
        c_sh  = valid[col].corr(valid['sharpe_ratio'])
        c_so  = valid[col].corr(valid['sortino_ratio'])
        c_ret = valid[col].corr(valid['total_return'])
        flag  = '  <- relevante' if abs(c_sh) >= 0.2 else ''
        print(f'  {p:<22}  {c_sh:>+14.4f}  {c_so:>+15.4f}  {c_ret:>+15.4f}{flag}')
        corr_rows.append({'strategy': strat, 'param': p,
                          'corr_sharpe': c_sh, 'corr_sortino': c_so, 'corr_return': c_ret})

    # Top vs Bot: distribuição dos parâmetros
    print(f'\n  Parâmetros nos TOP {n_q} vs PIORES {n_q} runs (Sharpe):')
    print(f'  {"Param":<22}  {"Top mean":>10}  {"Top med":>9}  {"Bot mean":>10}  {"Bot med":>9}  {"Diferença":>10}')
    print(f'  {"-"*22}  {"-"*10}  {"-"*9}  {"-"*10}  {"-"*9}  {"-"*10}')
    for p in params_cols:
        col = f'p_{p}'
        if col not in df_s.columns:
            continue
        top_vals = top_q[col].dropna()
        bot_vals = bot_q[col].dropna()
        if len(top_vals) == 0 or len(bot_vals) == 0:
            continue
        diff = top_vals.mean() - bot_vals.mean()
        flag = '  ^' if diff > 0 else '  v'
        print(f'  {p:<22}  {top_vals.mean():>10.3f}  {top_vals.median():>9.3f}  '
              f'{bot_vals.mean():>10.3f}  {bot_vals.median():>9.3f}  {diff:>+10.3f}{flag}')

    # Position type nos top runs
    pt_top = top_q['params_dict'].apply(lambda d: d.get('position_type', 'N/A')).value_counts()
    pt_bot = bot_q['params_dict'].apply(lambda d: d.get('position_type', 'N/A')).value_counts()
    print(f'\n  position_type — Top {n_q}: {dict(pt_top)}')
    print(f'  position_type — Bot {n_q}: {dict(pt_bot)}')

    # Plots por estratégia
    if not params_cols:
        continue

    fig, axes = plt.subplots(2, len(params_cols), figsize=(max(12, 3.5 * len(params_cols)), 8),
                             squeeze=False)
    fig.suptitle(f'Parâmetros — {strat}  |  Top vs Bottom 25% (Sharpe)', fontsize=11, fontweight='bold')

    for j, p in enumerate(params_cols):
        col = f'p_{p}'
        if col not in df_s.columns:
            continue

        # Row 0: histogramas top vs bot
        top_vals = top_q[col].dropna()
        bot_vals = bot_q[col].dropna()
        all_vals = df_s[col].dropna()
        bins = np.linspace(all_vals.min(), all_vals.max(), 15)
        axes[0, j].hist(top_vals, bins=bins, alpha=0.65, color='#2ecc71', label='Top 25%', density=True)
        axes[0, j].hist(bot_vals, bins=bins, alpha=0.65, color='#e74c3c', label='Bot 25%', density=True)
        axes[0, j].set_title(f'{p}', fontsize=9)
        axes[0, j].legend(fontsize=7)
        axes[0, j].grid(alpha=0.3)
        if j == 0:
            axes[0, j].set_ylabel('Densidade')

        # Row 1: scatter param vs Sharpe
        valid = df_s[[col, 'sharpe_ratio']].dropna()
        axes[1, j].scatter(valid[col], valid['sharpe_ratio'],
                           alpha=0.4, s=15, color='steelblue', edgecolors='none')
        axes[1, j].axhline(0, color='red', linestyle='--', lw=1)
        axes[1, j].set_xlabel(p, fontsize=9)
        if j == 0:
            axes[1, j].set_ylabel('Sharpe Ratio')
        axes[1, j].grid(alpha=0.3)

        # Linha de tendência
        z = np.polyfit(valid[col], valid['sharpe_ratio'], 1)
        x_line = np.linspace(valid[col].min(), valid[col].max(), 100)
        axes[1, j].plot(x_line, np.polyval(z, x_line), 'r-', lw=1.5, alpha=0.8)

    plt.tight_layout()
    safe_name = strat.replace('/', '_')
    plt.savefig(OUT_DIR / f'params_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Plot salvo: params_{safe_name}.png')

# ---------------------------------------------------------------------------
# Summary global: sl e tp vs Sharpe (todas estratégias)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('sl e tp vs Sharpe — Todas as Estratégias', fontsize=12, fontweight='bold')

strategies_all = sorted(df['strategy'].unique())
cmap = plt.cm.Set2(np.linspace(0, 1, len(strategies_all)))

for i, strat in enumerate(strategies_all):
    sub = df[df['strategy'] == strat]
    sl_col = sub['p_sl'].dropna()
    tp_col = sub['p_tp'].dropna()
    sr_col = sub.loc[sl_col.index, 'sharpe_ratio']
    axes[0].scatter(sl_col, sr_col, alpha=0.4, s=18, color=cmap[i], label=strat, edgecolors='none')
    axes[1].scatter(tp_col, sub.loc[tp_col.index, 'sharpe_ratio'],
                    alpha=0.4, s=18, color=cmap[i], edgecolors='none')

for ax, xlabel in zip(axes, ['sl (Stop Loss)', 'tp (Take Profit)']):
    ax.axhline(0, color='red', linestyle='--', lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(alpha=0.3)

axes[0].legend(fontsize=6, loc='upper right', ncol=2)
axes[0].set_title('Stop Loss vs Sharpe')
axes[1].set_title('Take Profit vs Sharpe')

plt.tight_layout()
plt.savefig(OUT_DIR / 'params_sl_tp_global.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\n{SEP}')
print(f'  Outputs salvos em analysis/output/')
print(f'  PNGs : params_{{strategy}}.png (um por estratégia) | params_sl_tp_global.png')
print(f'{SEP}\n')
