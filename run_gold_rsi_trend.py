"""
gold_rsi_trend — WIN@N | t5
============================
Edite o bloco CONFIG e execute:

    python run_gold_rsi_trend.py
"""

from _engine import execute

# ==============================================================
#  CONFIG — EDITE AQUI
# ==============================================================

STRATEGY  = 'gold_rsi_trend'
SYMBOL    = 'WIN@N'
TIMEFRAME = 't5'
DATA_INI  = '2019-01-01'
DATA_FIM  = '2026-06-30'
DAYTRADE  = True
INITIAL_CASH = 30_000

# 'backtest'  → teste rapido com FIXED_PARAMS
# 'optimize'  → otimizacao por hora com Optuna
MODE = 'backtest'

# --- Parametros fixos (modo backtest) -------------------------
FIXED_PARAMS = dict(
    sl            = 400,
    tp            = 1500,
    length_rsi    = 9,
    rsi_low       = 30,
    rsi_high      = 70,
    allowed_hours = [10, 11, 16, 17],
    position_type = 'both',
)

# --- Ranges para otimizacao (modo optimize) -------------------
PARAM_RANGES = dict(
    sl            = ('int',         100,  800),
    tp            = ('int',         100, 2000),
    length_rsi    = ('int',           6,   20),
    rsi_low       = ('int',          20,   45),
    rsi_high      = ('int',          55,   85),
    position_type = ('categorical', ['long', 'short', 'both']),
)

HOURS_TO_OPTIMIZE    = [9, 10, 11, 12, 13, 14, 15, 16, 17]
OPTIMIZE_METRIC      = 'sharpe_ratio'
N_TRIALS             = 100
MIN_TRADES           = 30
MIN_SHARPE_TO_SELECT = 0.0

# ==============================================================
execute(locals())
