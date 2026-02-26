"""
analysis/hours_analysis.py
============================
Análise de quais horas do dia geram melhor performance.

Uso:
    python analysis/hours_analysis.py
    python analysis/hours_analysis.py --min-trades 200
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
parser = argparse.ArgumentParser(description='Análise por hora do dia')
parser.add_argument('--min-trades', type=int, default=100, help='Minimo de trades (default 100)')
args = parser.parse_args()

MIN_TRADES = args.min_trades
ALL_HOURS  = list(range(9, 18))  # 9h a 17h

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------
df = pd.read_csv(LOG_PATH)
df = df[df['total_trades'] >= MIN_TRADES].copy()
df['hours_list'] = df['allowed_hours'].apply(ast.literal_eval)
df['n_hours']    = df['hours_list'].apply(len)

SEP = '=' * 100

# ---------------------------------------------------------------------------
# Stats por hora
# ---------------------------------------------------------------------------
rows = []
for h in ALL_HOURS:
    mask = df['hours_list'].apply(lambda lst: h in lst)
    sub  = df[mask]
    rows.append({
        'hour'          : h,
        'n_runs'        : len(sub),
        'pct_of_total'  : len(sub) / len(df) * 100,
        'pct_positive'  : (sub['sharpe_ratio'] > 0).mean() * 100,
        'sharpe_mean'   : sub['sharpe_ratio'].mean(),
        'sharpe_median' : sub['sharpe_ratio'].median(),
        'sharpe_max'    : sub['sharpe_ratio'].max(),
        'sortino_mean'  : sub['sortino_ratio'].mean(),
        'sortino_median': sub['sortino_ratio'].median(),
        'ret_mean'      : sub['total_return'].mean(),
        'ret_median'    : sub['total_return'].median(),
    })

hour_df = pd.DataFrame(rows).set_index('hour')

print(f'\n{SEP}')
print('  ANÁLISE POR HORA DO DIA')
print(SEP)
print(hour_df.round(4).to_string())
hour_df.to_csv(OUT_DIR / 'hours_stats.csv')

# ---------------------------------------------------------------------------
# Frequência por quartil (top 25% vs bot 25% por Sharpe)
# ---------------------------------------------------------------------------
n_q        = max(1, int(len(df) * 0.25))
top_q      = df.nlargest(n_q,  'sharpe_ratio')
bot_q      = df.nsmallest(n_q, 'sharpe_ratio')

top_freq = {h: top_q['hours_list'].apply(lambda lst: h in lst).sum() for h in ALL_HOURS}
bot_freq = {h: bot_q['hours_list'].apply(lambda lst: h in lst).sum() for h in ALL_HOURS}
top_pct  = {h: v / n_q * 100 for h, v in top_freq.items()}
bot_pct  = {h: v / n_q * 100 for h, v in bot_freq.items()}

print(f'\n{SEP}')
print('  FREQUÊNCIA DAS HORAS — QUARTIL SUPERIOR vs INFERIOR (Sharpe)')
print(f'  (Top 25%: {n_q} runs | Bot 25%: {n_q} runs)')
print(SEP)
print(f'  {"Hora":>5}  {"Top 25%":>8}  {"Bot 25%":>8}  {"Diferença":>10}  Sinal')
print(f'  {"-"*5}  {"-"*8}  {"-"*8}  {"-"*10}  {"-"*20}')
for h in ALL_HOURS:
    diff = top_pct[h] - bot_pct[h]
    sinal = '^ FAVORAVEL' if diff > 8 else ('v desfavoravel' if diff < -8 else '  neutro')
    print(f'  {h:>4}h  {top_pct[h]:>7.1f}%  {bot_pct[h]:>7.1f}%  {diff:>+10.1f}%  {sinal}')

# ---------------------------------------------------------------------------
# Análise por número de horas simultâneas
# ---------------------------------------------------------------------------
print(f'\n{SEP}')
print('  SHARPE MEDIANO POR Nº DE HORAS SIMULTÂNEAS')
print(SEP)

n_hours_stats = (
    df.groupby('n_hours')
    .agg(
        runs           = ('sharpe_ratio', 'count'),
        sharpe_median  = ('sharpe_ratio', 'median'),
        sharpe_mean    = ('sharpe_ratio', 'mean'),
        pct_positive   = ('sharpe_ratio', lambda x: (x > 0).mean() * 100),
        ret_median     = ('total_return',  'median'),
    )
)
print(n_hours_stats.round(4).to_string())
n_hours_stats.to_csv(OUT_DIR / 'n_hours_stats.csv')

# ---------------------------------------------------------------------------
# Top 20 runs: combinações de horas
# ---------------------------------------------------------------------------
print(f'\n{SEP}')
print('  TOP 20 RUNS — COMBINAÇÃO DE HORAS (por Sharpe)')
print(SEP)
print(f'  {"#":>3}  {"Estratégia":<35}  {"Horas":<35}  {"Sharpe":>8}  {"Sortino":>8}  {"Retorno":>14}  {"Trades":>6}')
print(f'  {"-"*3}  {"-"*35}  {"-"*35}  {"-"*8}  {"-"*8}  {"-"*14}  {"-"*6}')
for rank, (_, row) in enumerate(df.nlargest(20, 'sharpe_ratio').iterrows(), 1):
    print(f'  {rank:>3}  {row["strategy"]:<35}  {str(row["hours_list"]):<35}  '
          f'{row["sharpe_ratio"]:>8.4f}  {row["sortino_ratio"]:>8.4f}  '
          f'R$ {row["total_return"]:>10,.0f}  {int(row["total_trades"]):>6d}')

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise por Hora do Dia', fontsize=14, fontweight='bold')

hours_labels = [f'{h}h' for h in ALL_HOURS]

# 1. Sharpe mediano por hora
colors_sh = ['#2ecc71' if v >= 0 else '#e74c3c' for v in hour_df['sharpe_median']]
bars = axes[0, 0].bar(hours_labels, hour_df['sharpe_median'], color=colors_sh, edgecolor='white', width=0.6)
axes[0, 0].axhline(0, color='gray', linestyle='--', lw=1.5)
for bar, v in zip(bars, hour_df['sharpe_median']):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2,
                    v + (0.01 if v >= 0 else -0.03),
                    f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
axes[0, 0].set_title('Sharpe Mediano quando a Hora é Incluída')
axes[0, 0].set_ylabel('Sharpe Mediano')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. % runs positivas por hora
colors_pp = ['#2ecc71' if v >= 50 else '#e74c3c' for v in hour_df['pct_positive']]
axes[0, 1].bar(hours_labels, hour_df['pct_positive'], color=colors_pp, edgecolor='white', width=0.6)
axes[0, 1].axhline(50, color='red', linestyle='--', lw=1.5, label='50%')
axes[0, 1].set_title('% de Runs Positivas por Hora')
axes[0, 1].set_ylabel('%')
axes[0, 1].set_ylim(0, 100)
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Frequência Top vs Bot quartil
x = np.arange(len(ALL_HOURS))
w = 0.38
axes[1, 0].bar(x - w/2, [top_pct[h] for h in ALL_HOURS], w, label='Top 25% (Sharpe)', color='#2ecc71', alpha=0.85)
axes[1, 0].bar(x + w/2, [bot_pct[h] for h in ALL_HOURS], w, label='Bot 25% (Sharpe)', color='#e74c3c', alpha=0.85)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(hours_labels)
axes[1, 0].set_title('Frequência das Horas: Top 25% vs Bottom 25%')
axes[1, 0].set_ylabel('% de runs que incluem a hora')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Heatmap de co-ocorrência nos top 25%
n_h    = len(ALL_HOURS)
cooc   = np.zeros((n_h, n_h))
for lst in top_q['hours_list']:
    for i, h1 in enumerate(ALL_HOURS):
        if h1 in lst:
            for j, h2 in enumerate(ALL_HOURS):
                if h2 in lst:
                    cooc[i, j] += 1

im = axes[1, 1].imshow(cooc, cmap='YlOrRd', aspect='auto', interpolation='nearest')
axes[1, 1].set_xticks(range(n_h))
axes[1, 1].set_yticks(range(n_h))
axes[1, 1].set_xticklabels(hours_labels, fontsize=8)
axes[1, 1].set_yticklabels(hours_labels, fontsize=8)
axes[1, 1].set_title(f'Co-ocorrência de Horas — Top 25% (Sharpe)')
plt.colorbar(im, ax=axes[1, 1], shrink=0.85)
thresh = cooc.max() * 0.55
for i in range(n_h):
    for j in range(n_h):
        axes[1, 1].text(j, i, f'{int(cooc[i, j])}', ha='center', va='center',
                        fontsize=7, color='white' if cooc[i, j] > thresh else 'black')

plt.tight_layout()
plt.savefig(OUT_DIR / 'hours_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# Plot extra: Sharpe mediano por nº de horas
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(n_hours_stats.index.astype(str), n_hours_stats['sharpe_median'],
        color='steelblue', edgecolor='white', width=0.6)
ax2.axhline(0, color='red', linestyle='--', lw=1.5)
ax2.set_title('Sharpe Mediano por Número de Horas Simultâneas na Run')
ax2.set_xlabel('Número de horas')
ax2.set_ylabel('Sharpe Mediano')
ax2.grid(axis='y', alpha=0.3)
for i, (nh, row) in enumerate(n_hours_stats.iterrows()):
    ax2.text(i, row['sharpe_median'] + 0.02, f'n={int(row["runs"])}',
             ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / 'n_hours_sharpe.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\n{SEP}')
print(f'  Outputs salvos em analysis/output/')
print(f'  PNGs : hours_analysis.png | n_hours_sharpe.png')
print(f'  CSVs : hours_stats.csv | n_hours_stats.csv')
print(f'{SEP}\n')
