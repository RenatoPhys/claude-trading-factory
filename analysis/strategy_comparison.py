"""
analysis/strategy_comparison.py
=================================
Comparação de performance entre estratégias.

Uso:
    python analysis/strategy_comparison.py
    python analysis/strategy_comparison.py --min-trades 200
"""

import ast
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
parser = argparse.ArgumentParser(description='Comparação por estratégia')
parser.add_argument('--min-trades', type=int, default=100, help='Minimo de trades (default 100)')
args = parser.parse_args()

MIN_TRADES = args.min_trades

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------
df = pd.read_csv(LOG_PATH)
df = df[df['total_trades'] >= MIN_TRADES].copy()
df['params_dict']    = df['params'].apply(json.loads)
df['position_type']  = df['params_dict'].apply(lambda d: d.get('position_type', 'N/A'))
df['sl']             = df['params_dict'].apply(lambda d: d.get('sl'))
df['tp']             = df['params_dict'].apply(lambda d: d.get('tp'))
df['is_positive']    = df['sharpe_ratio'] > 0

SEP  = '=' * 110
SEP2 = '-' * 110
strategies = sorted(df['strategy'].unique())

# ---------------------------------------------------------------------------
# Tabela de stats por estratégia
# ---------------------------------------------------------------------------
def pct_pos(x):
    return (x > 0).mean() * 100

stats = (
    df.groupby('strategy')
    .agg(
        runs         = ('sharpe_ratio', 'count'),
        pct_positive = ('sharpe_ratio', pct_pos),
        sharpe_med   = ('sharpe_ratio', 'median'),
        sharpe_mean  = ('sharpe_ratio', 'mean'),
        sharpe_max   = ('sharpe_ratio', 'max'),
        sharpe_min   = ('sharpe_ratio', 'min'),
        sortino_med  = ('sortino_ratio', 'median'),
        sortino_max  = ('sortino_ratio', 'max'),
        calmar_med   = ('calmar_ratio', 'median'),
        ret_mean     = ('total_return', 'mean'),
        ret_median   = ('total_return', 'median'),
        ret_max      = ('total_return', 'max'),
        dd_median    = ('max_drawdown', 'median'),
        dd_max       = ('max_drawdown', 'max'),
    )
    .sort_values('sharpe_med', ascending=False)
)

print(f'\n{SEP}')
print('  COMPARAÇÃO POR ESTRATÉGIA — STATS GERAIS')
print(SEP)
print(stats.round(4).to_string())
stats.to_csv(OUT_DIR / 'strategy_stats.csv')

# ---------------------------------------------------------------------------
# Melhor run por estratégia
# ---------------------------------------------------------------------------
print(f'\n{SEP}')
print('  MELHOR RUN POR ESTRATÉGIA (Sharpe)')
print(SEP)

best_by_strat = df.loc[df.groupby('strategy')['sharpe_ratio'].idxmax()]
for _, row in best_by_strat.sort_values('sharpe_ratio', ascending=False).iterrows():
    print(f'\n  {row["strategy"]}')
    print(f'    Sharpe  : {row["sharpe_ratio"]:>8.4f} | Sortino : {row["sortino_ratio"]:>8.4f} | Calmar  : {row["calmar_ratio"]:>8.4f}')
    print(f'    Retorno : R$ {row["total_return"]:>12,.0f} | Trades  : {int(row["total_trades"]):>6d} | Win rate: {row["win_rate"]*100:.1f}%')
    print(f'    Horas   : {row["allowed_hours"]}')
    print(f'    Params  : {row["params"]}')

# ---------------------------------------------------------------------------
# Pior run por estratégia
# ---------------------------------------------------------------------------
print(f'\n{SEP}')
print('  PIOR RUN POR ESTRATÉGIA (Sharpe)')
print(SEP)

worst_by_strat = df.loc[df.groupby('strategy')['sharpe_ratio'].idxmin()]
for _, row in worst_by_strat.sort_values('sharpe_ratio', ascending=True).iterrows():
    print(f'\n  {row["strategy"]}')
    print(f'    Sharpe  : {row["sharpe_ratio"]:>8.4f} | Sortino : {row["sortino_ratio"]:>8.4f} | Drawdown: {row["max_drawdown"]:.2f}%')
    print(f'    Retorno : R$ {row["total_return"]:>12,.0f} | Trades  : {int(row["total_trades"]):>6d} | Win rate: {row["win_rate"]*100:.1f}%')
    print(f'    Horas   : {row["allowed_hours"]}')
    print(f'    Params  : {row["params"]}')

# ---------------------------------------------------------------------------
# Sharpe médio por position_type × estratégia
# ---------------------------------------------------------------------------
print(f'\n{SEP}')
print('  SHARPE MEDIANO — position_type × estratégia')
print(SEP)

pt_pivot = (
    df.groupby(['strategy', 'position_type'])['sharpe_ratio']
    .median()
    .unstack('position_type')
    .round(4)
)
print(pt_pivot.to_string())
pt_pivot.to_csv(OUT_DIR / 'strategy_position_type_sharpe.csv')

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparação por Estratégia', fontsize=14, fontweight='bold')

cmap = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

# 1. % runs positivas
colors_bar = ['#2ecc71' if v >= 50 else '#e74c3c' for v in stats['pct_positive']]
bars = axes[0, 0].barh(stats.index, stats['pct_positive'], color=colors_bar, edgecolor='white', height=0.6)
axes[0, 0].axvline(50, color='gray', linestyle='--', lw=1.5)
for bar, v in zip(bars, stats['pct_positive']):
    axes[0, 0].text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{v:.0f}%', va='center', fontsize=8)
axes[0, 0].set_title('% de Runs Positivas (Sharpe > 0)')
axes[0, 0].set_xlabel('%')
axes[0, 0].set_xlim(0, 105)
axes[0, 0].grid(axis='x', alpha=0.3)
axes[0, 0].tick_params(axis='y', labelsize=8)

# 2. Sharpe médio e mediano por estratégia
x = np.arange(len(stats))
w = 0.38
axes[0, 1].barh(x - w/2, stats['sharpe_mean'], w, label='Média',   color='steelblue', alpha=0.85)
axes[0, 1].barh(x + w/2, stats['sharpe_med'],  w, label='Mediana', color='darkorange', alpha=0.85)
axes[0, 1].set_yticks(x)
axes[0, 1].set_yticklabels(stats.index, fontsize=8)
axes[0, 1].axvline(0, color='red', linestyle='--', lw=1.2)
axes[0, 1].set_title('Sharpe Médio e Mediano por Estratégia')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. Boxplot Sharpe por estratégia
data_sharpe = [df[df['strategy'] == s]['sharpe_ratio'].values for s in stats.index]
bp = axes[1, 0].boxplot(data_sharpe, vert=False, patch_artist=True, labels=stats.index)
for patch, color in zip(bp['boxes'], cmap):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1, 0].axvline(0, color='red', linestyle='--', lw=1.2)
axes[1, 0].set_title('Distribuição do Sharpe por Estratégia')
axes[1, 0].set_xlabel('Sharpe Ratio')
axes[1, 0].tick_params(axis='y', labelsize=8)
axes[1, 0].grid(axis='x', alpha=0.3)

# 4. Retorno médio
colors_ret = ['#2ecc71' if v >= 0 else '#e74c3c' for v in stats['ret_mean']]
bars2 = axes[1, 1].barh(stats.index, stats['ret_mean'] / 1_000, color=colors_ret, edgecolor='white', height=0.6)
axes[1, 1].axvline(0, color='gray', linestyle='--', lw=1.5)
for bar, v in zip(bars2, stats['ret_mean'] / 1_000):
    axes[1, 1].text(v + (1 if v >= 0 else -1), bar.get_y() + bar.get_height() / 2,
                    f'R$ {v:.0f}k', va='center', fontsize=7,
                    ha='left' if v >= 0 else 'right')
axes[1, 1].set_title('Retorno Total Médio por Estratégia')
axes[1, 1].set_xlabel('R$ mil')
axes[1, 1].tick_params(axis='y', labelsize=8)
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Violin plot separado (distribuição completa)
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(14, 5))
fig2.suptitle('Distribuição do Sharpe por Estratégia (Violin)', fontsize=12, fontweight='bold')

parts = ax.violinplot(data_sharpe, vert=True, showmedians=True, showextrema=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(cmap[i])
    pc.set_alpha(0.75)
parts['cmedians'].set_color('black')
ax.axhline(0, color='red', linestyle='--', lw=1.5, alpha=0.7)
ax.set_xticks(range(1, len(stats) + 1))
ax.set_xticklabels(stats.index, rotation=15, ha='right', fontsize=8)
ax.set_ylabel('Sharpe Ratio')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'strategy_violin.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\n{SEP}')
print(f'  Outputs salvos em analysis/output/')
print(f'  PNGs : strategy_comparison.png | strategy_violin.png')
print(f'  CSVs : strategy_stats.csv | strategy_position_type_sharpe.csv')
print(f'{SEP}\n')
