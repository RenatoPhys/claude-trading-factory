"""
Trading Factory — Engine
========================
Importado pelos scripts run_{strategy}.py.
Nao executar diretamente.
"""

import sys
import json
import importlib
import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from futures_backtester import Backtester
from config.dicts_params import dict_custos, dict_valor_lot, dict_path


class _JsonEncoder(json.JSONEncoder):
    """Converte tipos numpy para Python nativo antes de serializar."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


_METRIC_KEYS = (
    'total_return', 'annual_return',
    'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
    'profit_factor', 'win_rate', 'max_drawdown', 'total_trades',
)


def execute(cfg: dict) -> None:
    """Executa backtest ou otimizacao com base no dicionario de configuracao cfg."""

    STRATEGY     = cfg['STRATEGY']
    SYMBOL       = cfg['SYMBOL']
    TIMEFRAME    = cfg['TIMEFRAME']
    DATA_INI     = cfg['DATA_INI']
    DATA_FIM     = cfg['DATA_FIM']
    DAYTRADE     = cfg.get('DAYTRADE', True)
    INITIAL_CASH = cfg.get('INITIAL_CASH', 30_000)
    MODE         = cfg['MODE']

    _mod   = importlib.import_module('entries')
    _entry = getattr(_mod, STRATEGY)

    # ----------------------------------------------------------
    # Helpers internos
    # ----------------------------------------------------------

    def _make_bt(tp, sl) -> Backtester:
        return Backtester(
            symbol       = SYMBOL,
            timeframe    = TIMEFRAME,
            data_ini     = DATA_INI,
            data_fim     = DATA_FIM,
            tp           = tp,
            sl           = sl,
            slippage     = 0,
            tc           = dict_custos[SYMBOL],
            lote         = 1,
            valor_lote   = dict_valor_lot[SYMBOL],
            initial_cash = INITIAL_CASH,
            path_base    = dict_path[SYMBOL],
            daytrade     = DAYTRADE,
        )

    def _run(params: dict) -> dict:
        signal_args = {k: v for k, v in params.items() if k not in ('tp', 'sl')}
        bt = _make_bt(params['tp'], params['sl'])
        _, metrics = bt.run(signal_function=_entry, signal_args=signal_args)
        return metrics

    def _print_metrics(metrics: dict, header: str = '') -> None:
        SEP = '-' * 52
        if header:
            print(f'\n{SEP}')
            print(f'  {header}')
        print(SEP)
        print(f"  Retorno total  : R$ {metrics.get('total_return', 0):>14,.2f}")
        print(f"  Retorno anual  : {metrics.get('annual_return', 0):>13.2f} %")
        print(f"  Sharpe         : {metrics.get('sharpe_ratio', 0):>16.4f}")
        print(f"  Sortino        : {metrics.get('sortino_ratio', 0):>16.4f}")
        print(f"  Calmar         : {metrics.get('calmar_ratio', 0):>16.4f}")
        print(f"  Max Drawdown   : {metrics.get('max_drawdown', 0):>13.2f} %")
        print(f"  Trades         : {int(metrics.get('total_trades', 0)):>16d}")
        print(f"  Win rate       : {metrics.get('win_rate', 0) * 100:>13.1f} %")
        print(f"  Profit factor  : {metrics.get('profit_factor', 0):>16.4f}")
        print(SEP)

    def _save_plots(bt: Backtester, out_dir: Path, prefix: str = '') -> None:
        plots = [
            (f'{prefix}1_equity_drawdown.png',        lambda: bt.plot_equity_curve(include_drawdown=True)),
            (f'{prefix}2_pnl_by_direction.png',        lambda: bt.plot_by_position()),
            (f'{prefix}3_pnl_cumulative_by_hour.png',  lambda: bt.plot_cumulative_by_hour()),
            (f'{prefix}4_pnl_by_hour.png',             lambda: bt.plot_profit_by_hour()),
        ]
        saved = []
        for filename, plot_fn in plots:
            try:
                plt.close('all')
                plot_fn()
                plt.savefig(out_dir / filename, dpi=150, bbox_inches='tight')
                saved.append(filename)
            except Exception as e:
                print(f'  Aviso: erro ao gerar {filename}: {e}')
            finally:
                plt.close('all')
        if saved:
            print(f'  Plots salvos ({len(saved)}): {out_dir.name}/')

    def _run_and_plot(params: dict, out_dir: Path, prefix: str = '') -> dict:
        signal_args = {k: v for k, v in params.items() if k not in ('tp', 'sl')}
        bt = _make_bt(params['tp'], params['sl'])
        _, metrics = bt.run(signal_function=_entry, signal_args=signal_args)
        _save_plots(bt, out_dir, prefix=prefix)
        return metrics

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # ----------------------------------------------------------
    # MODE: backtest
    # ----------------------------------------------------------
    if MODE == 'backtest':
        FIXED_PARAMS = cfg['FIXED_PARAMS']
        run_dir = ROOT / 'results' / f'backtest_{SYMBOL}_{STRATEGY}_{TIMEFRAME}_{ts}'
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f'\nBacktest  |  {STRATEGY}  |  {SYMBOL} {TIMEFRAME}')
        print(f'Periodo:  {DATA_INI} a {DATA_FIM}')
        print(f'Params:   {FIXED_PARAMS}')

        metrics = _run_and_plot(FIXED_PARAMS, run_dir)
        _print_metrics(metrics, f'{STRATEGY}  —  {SYMBOL} {TIMEFRAME}')

        (run_dir / 'config.json').write_text(json.dumps({
            'strategy': STRATEGY, 'symbol': SYMBOL, 'timeframe': TIMEFRAME,
            'data_ini': DATA_INI, 'data_fim': DATA_FIM, 'daytrade': DAYTRADE,
            'initial_cash': INITIAL_CASH, 'params': FIXED_PARAMS,
        }, indent=2, cls=_JsonEncoder))
        (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, cls=_JsonEncoder))

        # --- backtest_log.csv ---
        log_path = ROOT / 'backtest_log.csv'
        log_row = pd.DataFrame([{
            'timestamp'     : ts,
            'strategy'      : STRATEGY,
            'symbol'        : SYMBOL,
            'timeframe'     : TIMEFRAME,
            'data_ini'      : DATA_INI,
            'data_fim'      : DATA_FIM,
            'allowed_hours' : str(FIXED_PARAMS.get('allowed_hours', [])),
            'params'        : json.dumps(FIXED_PARAMS, cls=_JsonEncoder),
            'total_return'  : metrics.get('total_return'),
            'annual_return' : metrics.get('annual_return'),
            'sharpe_ratio'  : metrics.get('sharpe_ratio'),
            'sortino_ratio' : metrics.get('sortino_ratio'),
            'calmar_ratio'  : metrics.get('calmar_ratio'),
            'max_drawdown'  : metrics.get('max_drawdown'),
            'total_trades'  : metrics.get('total_trades'),
            'win_rate'      : metrics.get('win_rate'),
            'profit_factor' : metrics.get('profit_factor'),
        }])
        log_row.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
        print(f'  Log: backtest_log.csv')

    # ----------------------------------------------------------
    # MODE: optimize
    # ----------------------------------------------------------
    elif MODE == 'optimize':
        PARAM_RANGES         = cfg['PARAM_RANGES']
        HOURS_TO_OPTIMIZE    = cfg['HOURS_TO_OPTIMIZE']
        OPTIMIZE_METRIC      = cfg.get('OPTIMIZE_METRIC', 'sharpe_ratio')
        N_TRIALS             = cfg.get('N_TRIALS', 100)
        MIN_TRADES           = cfg.get('MIN_TRADES', 30)
        MIN_SHARPE_TO_SELECT = cfg.get('MIN_SHARPE_TO_SELECT', 0.0)

        run_dir = ROOT / 'results' / f'run_{SYMBOL}_{STRATEGY}_{TIMEFRAME}_{ts}'
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / 'config.json').write_text(json.dumps({
            'strategy': STRATEGY, 'symbol': SYMBOL, 'timeframe': TIMEFRAME,
            'data_ini': DATA_INI, 'data_fim': DATA_FIM, 'daytrade': DAYTRADE,
            'initial_cash': INITIAL_CASH, 'optimize_metric': OPTIMIZE_METRIC,
            'n_trials': N_TRIALS, 'min_trades': MIN_TRADES,
            'hours': HOURS_TO_OPTIMIZE, 'param_ranges': PARAM_RANGES,
        }, indent=2, cls=_JsonEncoder))

        print(f'\nOtimizando  |  {STRATEGY}  |  {SYMBOL} {TIMEFRAME}')
        print(f'Metrica: {OPTIMIZE_METRIC}  |  Trials/hora: {N_TRIALS}')
        print(f'Horas: {HOURS_TO_OPTIMIZE}')
        print(f'Resultados em: {run_dir.name}/\n')
        print(f'  {"hora":>4}  {"Sharpe":>7}  {"Return":>14}  {"Trades":>6}  params')
        print(f'  {"-"*4}  {"-"*7}  {"-"*14}  {"-"*6}  {"-"*40}')

        hour_results = []

        for hour in HOURS_TO_OPTIMIZE:

            def _make_objective(h: int):
                def objective(trial: optuna.Trial) -> float:
                    params = {'allowed_hours': [h]}
                    for name, spec in PARAM_RANGES.items():
                        if spec[0] == 'int':
                            params[name] = trial.suggest_int(name, spec[1], spec[2])
                        elif spec[0] == 'float':
                            params[name] = trial.suggest_float(name, spec[1], spec[2])
                        elif spec[0] == 'categorical':
                            params[name] = trial.suggest_categorical(name, spec[1])
                    m = _run(params)
                    if m.get('total_trades', 0) < MIN_TRADES:
                        return float('-inf')
                    return m.get(OPTIMIZE_METRIC, float('-inf'))
                return objective

            study = optuna.create_study(direction='maximize')
            study.optimize(_make_objective(hour), n_trials=N_TRIALS, show_progress_bar=False)

            best = study.best_params.copy()
            best['allowed_hours'] = [hour]
            best_m = _run(best)

            result = {
                'hour'       : hour,
                'best_params': best,
                'best_value' : study.best_value,
                'metrics'    : {k: best_m.get(k) for k in _METRIC_KEYS},
            }
            hour_results.append(result)
            (run_dir / f'results_hour_{hour:02d}.json').write_text(json.dumps(result, indent=2, cls=_JsonEncoder))

            sr  = best_m.get('sharpe_ratio', 0)
            ret = best_m.get('total_return', 0)
            tr  = int(best_m.get('total_trades', 0))
            print(f'  {hour:>4}h  {sr:>7.3f}  R$ {ret:>12,.0f}  {tr:>6d}  {best}')

        selected = [
            r for r in hour_results
            if r['metrics']['sharpe_ratio'] > MIN_SHARPE_TO_SELECT
            and r['metrics']['total_trades'] >= MIN_TRADES
        ]

        combined = {
            'symbol'         : SYMBOL,
            'timeframe'      : TIMEFRAME,
            'strategy'       : STRATEGY,
            'hours'          : [r['hour'] for r in selected],
            'hour_params'    : {str(r['hour']): r['best_params'] for r in selected},
            'tc'             : dict_custos[SYMBOL],
            'valor_lote'     : dict_valor_lot[SYMBOL],
            'lote'           : 1,
            'daytrade'       : DAYTRADE,
            'optimize_metric': OPTIMIZE_METRIC,
            'direction'      : 'maximize',
        }
        (run_dir / 'combined_strategy.json').write_text(json.dumps(combined, indent=2, cls=_JsonEncoder))

        print(f'\nRun completo.')
        print(f'Horas selecionadas (Sharpe > {MIN_SHARPE_TO_SELECT}): {combined["hours"]}')

        if selected:
            print('Gerando plots por hora...')
            for r in selected:
                h      = r['hour']
                prefix = f'hour_{h:02d}_'
                _run_and_plot(r['best_params'], run_dir, prefix=prefix)
                print(f'  hora {h:02d}h OK')

        print(f'\ncombined_strategy.json -> {run_dir.name}/')
        print('Copie para selected/ se aprovado.')

    else:
        print(f'Modo invalido: {MODE!r}. Use "backtest" ou "optimize".')
