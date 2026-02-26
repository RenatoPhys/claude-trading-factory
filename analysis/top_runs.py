"""
analysis/top_runs.py
=====================
Rankings das melhores e piores runs do backtest_log.csv.

Uso:
    python analysis/top_runs.py
    python analysis/top_runs.py --top 30 --min-trades 200
"""

import ast
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
parser = argparse.ArgumentParser(description='Rankings das melhores/piores runs')
parser.add_argument('--top',        type=int, default=20,  help='Numero de runs a mostrar (default 20)')
parser.add_argument('--min-trades', type=int, default=100, help='Minimo de trades por run (default 100)')
args = parser.parse_args()

TOP_N      = args.top
MIN_TRADES = args.min_trades

# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------
df = pd.read_csv(LOG_PATH)
df_f = df[df['total_trades'] >= MIN_TRADES].copy()

n_total    = len(df)
n_f        = len(df_f)
n_positive = (df_f['sharpe_ratio'] > 0).sum()
pct_pos    = n_positive / n_f * 100

SEP  = '=' * 112
SEP2 = '-' * 112

# ---------------------------------------------------------------------------
# Sumário geral
# ---------------------------------------------------------------------------
print(f'\n{SEP}')
print(f'  BACKTEST LOG — TOP / WORST RUNS')
print(SEP)
print(f'  Total de runs no log          : {n_total}')
print(f'  Runs com >= {MIN_TRADES} trades         : {n_f}')
print(f'  Runs com Sharpe > 0           : {n_positive}  ({pct_pos:.1f}%)')
print(f'  Sharpe  — mediana / max / min : {df_f["sharpe_ratio"].median():.4f} / {df_f["sharpe_ratio"].max():.4f} / {df_f["sharpe_ratio"].min():.4f}')
print(f'  Sortino — mediana / max / min : {df_f["sortino_ratio"].median():.4f} / {df_f["sortino_ratio"].max():.4f} / {df_f["sortino_ratio"].min():.4f}')
print(f'  Retorno total médio           : R$ {df_f["total_return"].mean():>12,.0f}')
print(f'  Retorno total mediano         : R$ {df_f["total_return"].median():>12,.0f}')
print(SEP)

# ---------------------------------------------------------------------------
# Formatador de tabela
# ---------------------------------------------------------------------------
DISP_COLS = [
    'strategy', 'allowed_hours',
    'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
    'total_return', 'max_drawdown', 'total_trades',
    'win_rate', 'profit_factor', 'params',
]

def fmt(df_sub: pd.DataFrame) -> pd.DataFrame:
    s = df_sub[DISP_COLS].copy()
    s['total_return']  = s['total_return'].map(lambda x: f'R$ {x:>11,.0f}')
    s['sharpe_ratio']  = s['sharpe_ratio'].map(lambda x: f'{x:>8.4f}')
    s['sortino_ratio'] = s['sortino_ratio'].map(lambda x: f'{x:>8.4f}')
    s['calmar_ratio']  = s['calmar_ratio'].map(lambda x: f'{x:>8.4f}')
    s['max_drawdown']  = s['max_drawdown'].map(lambda x: f'{x:>6.2f}%')
    s['win_rate']      = s['win_rate'].map(lambda x: f'{x * 100:>5.1f}%')
    s['profit_factor'] = s['profit_factor'].map(lambda x: f'{x:>6.3f}')
    s['total_trades']  = s['total_trades'].map(lambda x: f'{int(x):>6d}')
    return s

def show_ranking(df_sub: pd.DataFrame, title: str, col: str, ascending: bool = False) -> None:
    ranked = df_sub.sort_values(col, ascending=ascending).head(TOP_N)
    print(f'\n{SEP}')
    print(f'  TOP {TOP_N} — {title}')
    print(SEP)
    print(fmt(ranked).to_string(index=True))
    ranked.to_csv(OUT_DIR / f'top_{col}.csv', index=False)

# ---------------------------------------------------------------------------
# Rankings — melhores
# ---------------------------------------------------------------------------
show_ranking(df_f, 'Sharpe Ratio',  'sharpe_ratio')
show_ranking(df_f, 'Sortino Ratio', 'sortino_ratio')
show_ranking(df_f, 'Retorno Total', 'total_return')
show_ranking(df_f, 'Calmar Ratio',  'calmar_ratio')

# ---------------------------------------------------------------------------
# Rankings — piores
# ---------------------------------------------------------------------------
print(f'\n{SEP}')
print(f'  PIORES {TOP_N} — Sharpe Ratio')
print(SEP)
worst = df_f.sort_values('sharpe_ratio', ascending=True).head(TOP_N)
print(fmt(worst).to_string(index=True))
worst.to_csv(OUT_DIR / 'worst_sharpe.csv', index=False)

print(f'\n{SEP}')
print(f'  PIORES {TOP_N} — Retorno Total')
print(SEP)
worst_ret = df_f.sort_values('total_return', ascending=True).head(TOP_N)
print(fmt(worst_ret).to_string(index=True))
worst_ret.to_csv(OUT_DIR / 'worst_total_return.csv', index=False)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Distribuição de Performance — Backtest Log', fontsize=14, fontweight='bold')

strategies = sorted(df_f['strategy'].unique())
cmap       = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

# 1. Boxplot Sharpe por estratégia
data_sharpe = [df_f[df_f['strategy'] == s]['sharpe_ratio'].values for s in strategies]
bp = axes[0, 0].boxplot(data_sharpe, patch_artist=True,
                        labels=[s.replace('_', '\n') for s in strategies], vert=True)
for patch, color in zip(bp['boxes'], cmap):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0, 0].axhline(0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='Sharpe = 0')
axes[0, 0].set_title('Sharpe por Estratégia')
axes[0, 0].set_ylabel('Sharpe Ratio')
axes[0, 0].tick_params(axis='x', labelsize=7)
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Histograma geral do Sharpe
med  = df_f['sharpe_ratio'].median()
mean = df_f['sharpe_ratio'].mean()
axes[0, 1].hist(df_f['sharpe_ratio'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0, 1].axvline(0,    color='red',    linestyle='--', lw=1.5, label='Sharpe = 0')
axes[0, 1].axvline(med,  color='orange', linestyle='--', lw=1.5, label=f'Mediana = {med:.2f}')
axes[0, 1].axvline(mean, color='green',  linestyle='--', lw=1.5, label=f'Média = {mean:.2f}')
axes[0, 1].set_title('Histograma do Sharpe Ratio')
axes[0, 1].set_xlabel('Sharpe Ratio')
axes[0, 1].set_ylabel('Frequência')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# 3. Scatter Sharpe vs Total Return
colors_dot = df_f['strategy'].map({s: i for i, s in enumerate(strategies)})
sc = axes[1, 0].scatter(
    df_f['sharpe_ratio'], df_f['total_return'] / 1_000,
    c=colors_dot, cmap='Set2', alpha=0.6, s=20, edgecolors='none'
)
axes[1, 0].axvline(0, color='red',  linestyle='--', lw=1, alpha=0.7)
axes[1, 0].axhline(0, color='gray', linestyle='--', lw=1, alpha=0.7)
axes[1, 0].set_title('Sharpe vs Retorno Total')
axes[1, 0].set_xlabel('Sharpe Ratio')
axes[1, 0].set_ylabel('Retorno Total (R$ mil)')
axes[1, 0].grid(alpha=0.3)

# 4. Sharpe vs Max Drawdown
axes[1, 1].scatter(
    df_f['max_drawdown'], df_f['sharpe_ratio'],
    c=colors_dot, cmap='Set2', alpha=0.6, s=20, edgecolors='none'
)
axes[1, 1].axhline(0, color='red', linestyle='--', lw=1, alpha=0.7)
axes[1, 1].set_title('Max Drawdown vs Sharpe Ratio')
axes[1, 1].set_xlabel('Max Drawdown (%)')
axes[1, 1].set_ylabel('Sharpe Ratio')
axes[1, 1].grid(alpha=0.3)

# Legenda de estratégias
handles = [plt.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=cmap[i], markersize=8, label=s)
           for i, s in enumerate(strategies)]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT_DIR / 'top_runs_overview.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\n{SEP}')
print(f'  Outputs salvos em analysis/output/')
print(f'  PNGs  : top_runs_overview.png')
print(f'  CSVs  : top_sharpe_ratio.csv | top_sortino_ratio.csv | top_total_return.csv')
print(f'          top_calmar_ratio.csv | worst_sharpe.csv | worst_total_return.csv')
print(f'{SEP}\n')
