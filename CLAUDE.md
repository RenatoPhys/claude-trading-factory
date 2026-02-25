# CLAUDE.md — claude-trading-factory

Guia para Claude Code trabalhar neste diretório.

## Visão Geral

Esse diretório elabora e testa novas estratégias de algo trading, trabalhando com futuros WIN e WDO pricipalmente. Por enquanto, todas as estratégias consistem em entradas (entries.py), e as saídas se dão por take profit, stop loss, ou fim do dia.


## Executar

Sempre rodar a partir de `claude-trading-factory/` como working directory:

```bash
python run.py
```

Python: `C:/Users/User/anaconda3/python.exe`

---

## run.py — Bloco CONFIG

Edite apenas o bloco `CONFIG` no topo do arquivo. O restante não precisa ser alterado.

### Variáveis principais

| Variável | Descrição |
|---|---|
| `STRATEGY` | Nome da função em `entries/entries.py` |
| `SYMBOL` | Símbolo (ex: `'WIN@N'`, `'EURUSD'`) |
| `TIMEFRAME` | Timeframe dos dados (ex: `'t5'`) |
| `DATA_INI` / `DATA_FIM` | Período do backtest (`'YYYY-MM-DD'`) |
| `MODE` | `'backtest'` ou `'optimize'` |

### MODE = 'backtest'

Roda uma vez com parâmetros fixos, imprime métricas e salva plots em `results/`.

```python
MODE = 'backtest'
FIXED_PARAMS = dict(
    sl            = 400,
    tp            = 1500,
    length_rsi    = 9,
    rsi_low       = 30,
    rsi_high      = 70,
    allowed_hours = [10, 11, 16, 17],
    position_type = 'both',
)
```

Output em: `results/backtest_{SYMBOL}_{STRATEGY}_{TF}_{TIMESTAMP}/`
- `config.json`, `metrics.json`, 4 plots PNG

### MODE = 'optimize'

Optuna por hora, maximiza `OPTIMIZE_METRIC`. Filtra horas com `MIN_TRADES` e `MIN_SHARPE_TO_SELECT`.

```python
MODE = 'optimize'
PARAM_RANGES = dict(
    sl            = ('int',         100,  800),
    tp            = ('int',         100, 1200),
    length_rsi    = ('int',           6,   16),
    rsi_low       = ('int',          20,   50),
    rsi_high      = ('int',          50,   85),
    position_type = ('categorical', ['long', 'short', 'both']),
)
HOURS_TO_OPTIMIZE    = [9, 10, 11, 12, 13, 14, 15, 16, 17]
OPTIMIZE_METRIC      = 'sharpe_ratio'   # ou 'sortino_ratio', 'calmar_ratio', etc.
N_TRIALS             = 100
MIN_TRADES           = 30
MIN_SHARPE_TO_SELECT = 0.0
```

Output em: `results/run_{SYMBOL}_{STRATEGY}_{TF}_{TIMESTAMP}/`
- `config.json`, `results_hour_NN.json` por hora, `combined_strategy.json`

Após aprovação: copiar `combined_strategy.json` para `selected/combined_strategy_N.json` e adicionar `magic_number`.

---

## Estratégias — entries/entries.py

Fonte única de todas as funções de entrada. Exportadas automaticamente via `entries/__init__.py`.

### Assinatura padrão

```python
def minha_estrategia(df, param1, param2, ..., allowed_hours=None, position_type="both"):
    """
    Returns:
        pd.Series: posições (-1=short, 0=flat, +1=long), mesmo índice que df
    """
    df = df.copy()
    # ... lógica ...
    if allowed_hours is not None:
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
    return df['position']
```

**Regras obrigatórias:**
- Sempre `df = df.copy()` no início
- Sempre aplicar filtro `allowed_hours` antes do return
- Retornar `df['position']` (Series), não o DataFrame inteiro
- `position_type`: `"long"` (só +1), `"short"` (só -1), `"both"` (ambos)

### Estratégias disponíveis

| Função | Indicador | Parâmetros específicos |
|---|---|---|
| `gold_rsi_trend` | RSI (contra-tendência) | `length_rsi, rsi_low, rsi_high` |
| `pattern_rsi_trend` | RSI + pct_change (tendência) | `length_rsi, rsi_low, rsi_high` |
| `pattern_rsi_anti_trend` | RSI + pct_change (contra-tendência) | `length_rsi, rsi_low, rsi_high` |
| `bb_trend` | Bollinger Bands (tendência) | `bb_length, std` |
| `bb_anti_trend` | Bollinger Bands (reversão) | `bb_length, std` |
| `macd_crossover_trend` | MACD cruzamento (tendência) | `fast_period, slow_period, signal_period` |
| `macd_crossover_anti_trend` | MACD cruzamento (contra-tendência) | `fast_period, slow_period, signal_period` |
| `momentum_breakout` | Momentum + volume | `lookback_period, momentum_threshold, volume_factor` |

### Adicionar nova estratégia

1. Adicionar função em `entries/entries.py` seguindo a assinatura padrão
2. Em `run.py`, setar `STRATEGY = 'nome_da_funcao'`
3. Ajustar `FIXED_PARAMS` ou `PARAM_RANGES` conforme os parâmetros da função
4. Rodar `python run.py`

---

## Símbolos suportados — config/dicts_params.py

### B3 Futuros

| Símbolo | Custo/lot | Valor/ponto/lot | Path |
|---|---|---|---|
| `WIN@N` | R$ 1,00 | R$ 0,20 | `path_b3` |
| `WDO@N` | R$ 1,20 | R$ 10,00 | `path_b3` |
| `WSP@N` | R$ 1,00 | R$ 2,50 | `path_b3` |
| `BIT@N` | R$ 1,00 | R$ 0,01 | `path_b3` |

### Forex/Tickmill

Custo padrão: 3 (por lote). Valor por lote: 100.000 (exceto STOXX50, UK100, DE40, FRANCE40, US500 = 1). Path: `path_tickmill`.

Pares disponíveis: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, NZDUSD, e ~30 outros pares/commodities.

---

## Backtester — uso direto (sem run.py)

```python
import importlib
from futures_backtester import Backtester
from config.dicts_params import dict_custos, dict_valor_lot, dict_path

module = importlib.import_module('entries')
entrada = getattr(module, 'nome_da_funcao')

bt = Backtester(
    symbol       = 'WIN@N',
    timeframe    = 't5',
    data_ini     = '2019-01-01',
    data_fim     = '2026-06-30',
    tp           = 1500,
    sl           = 400,
    slippage     = 0,
    tc           = dict_custos['WIN@N'],
    lote         = 1,
    valor_lote   = dict_valor_lot['WIN@N'],
    initial_cash = 30_000,
    path_base    = dict_path['WIN@N'],
    daytrade     = True,
)
df, metrics = bt.run(signal_function=entrada, signal_args={
    'length_rsi'    : 9,
    'rsi_low'       : 30,
    'rsi_high'      : 70,
    'allowed_hours' : [10, 11, 16, 17],
    'position_type' : 'both',
})
```

Métricas disponíveis: `total_return`, `annual_return`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `profit_factor`, `win_rate`, `max_drawdown`, `total_trades`.

---

## selected/ — JSONs de estratégias validadas

Formato do `combined_strategy_N.json`:

```json
{
  "symbol": "WIN@N",
  "timeframe": "t5",
  "strategy": "pattern_rsi_trend",
  "magic_number": 1111,
  "hours": [10, 11, 16, 17],
  "hour_params": {
    "10": {"tp": 1400, "sl": 600, "length_rsi": 9, "rsi_low": 26, "rsi_high": 74, "position_type": "both", "allowed_hours": [10]}
  },
  "tc": 1.0,
  "valor_lote": 0.2,
  "lote": 1,
  "daytrade": true,
  "optimize_metric": "sortino_ratio"
}
```

`magic_number` identifica o EA no MetaTrader 5. Deve ser único por estratégia deployada.

---

## Convenções

- Capital inicial: **R$ 30.000**
- Posições: `+1 = long`, `-1 = short`, `0 = flat`
- Lucros são **brutos** (pré-IR). Day trade IR = 20%
- Dados CSV nomeados: `{symbol}_{timeframe}.csv` em `path_b3` ou `path_tickmill`
- Arquivos de resultado do MT5: `results_{symbol}_{timeframe}_{strategy}_magic_{magic_number}.csv`
