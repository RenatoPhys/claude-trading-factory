"""
analysis/run_all.py
====================
Executa todos os scripts de análise em sequência.

Uso:
    python analysis/run_all.py
    python analysis/run_all.py --min-trades 200 --top 30
"""

import argparse
import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent
PYTHON       = sys.executable

parser = argparse.ArgumentParser(description='Executa todos os scripts de análise')
parser.add_argument('--top',        type=int, default=20,  help='Numero de runs a mostrar (default 20)')
parser.add_argument('--min-trades', type=int, default=100, help='Minimo de trades (default 100)')
args = parser.parse_args()

SCRIPTS = [
    ('Top Runs',              ANALYSIS_DIR / 'top_runs.py',           ['--top', str(args.top), '--min-trades', str(args.min_trades)]),
    ('Strategy Comparison',   ANALYSIS_DIR / 'strategy_comparison.py', ['--min-trades', str(args.min_trades)]),
    ('Hours Analysis',        ANALYSIS_DIR / 'hours_analysis.py',      ['--min-trades', str(args.min_trades)]),
    ('Params Analysis',       ANALYSIS_DIR / 'params_analysis.py',     ['--min-trades', str(args.min_trades)]),
]

SEP = '=' * 70

print(f'\n{SEP}')
print(f'  ANÁLISE COMPLETA — backtest_log.csv')
print(f'  min_trades={args.min_trades} | top_n={args.top}')
print(SEP)

for name, script, extra_args in SCRIPTS:
    print(f'\n{"-"*70}')
    print(f'  >> {name}')
    print(f'{"-"*70}')
    result = subprocess.run([PYTHON, str(script)] + extra_args)
    if result.returncode != 0:
        print(f'  ERRO em {name} (exit code {result.returncode})')
    else:
        print(f'  OK   {name} concluido')

print(f'\n{SEP}')
print(f'  Todos os outputs em: analysis/output/')
print(SEP + '\n')
