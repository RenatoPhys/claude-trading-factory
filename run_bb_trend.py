"""
bb_trend — WIN@N | t5
======================
Edite o bloco CONFIG e execute:

    python run_bb_trend.py
"""

from _engine import execute

# ==============================================================
#  CONFIG — EDITE AQUI
# ==============================================================

STRATEGY  = 'bb_trend'
SYMBOL    = 'WIN@N'
TIMEFRAME = 't5'
DATA_INI  = '2026-01-01'
DATA_FIM  = '2026-03-21'
DAYTRADE  = True
INITIAL_CASH = 30_000

# 'backtest'  → teste rapido com FIXED_PARAMS
# 'optimize'  → otimizacao por hora com Optuna
MODE = 'backtest'

# --- Parametros fixos (modo backtest) -------------------------
FIXED_PARAMS = dict(
    sl            = 655,
    tp            = 1971,
    bb_length     = 11,
    std           = 2.8,
    allowed_hours = [11, 13, 14, 16, 17],
    position_type = 'both',
)

# --- Ranges para otimizacao (modo optimize) -------------------
PARAM_RANGES = dict(
    sl            = ('int',         100,  800),
    tp            = ('int',         100, 2000),
    bb_length     = ('int',          10,   40),
    std           = ('float',       1.0,  3.0),
    position_type = ('categorical', ['long', 'short', 'both']),
)

HOURS_TO_OPTIMIZE    = [9, 10, 11, 12, 13, 14, 15, 16, 17]
OPTIMIZE_METRIC      = 'sharpe_ratio'
N_TRIALS             = 100
MIN_TRADES           = 30
MIN_SHARPE_TO_SELECT = 0.0

# ==============================================================
execute(locals())
