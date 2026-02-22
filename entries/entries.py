
import numpy as np
import pandas as pd
import pandas_ta as ta

def gold_rsi_trend(df, length_rsi, rsi_low, rsi_high, allowed_hours=None, position_type="both"):

    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    
    # Calcula o RSI
    df['rsi'] = df.ta.rsi(length= length_rsi).fillna(0)
    
    # Condições de entrada
    cond1 = (df.rsi < rsi_low) & (df.rsi.shift(+1) >= rsi_low)
    cond2 = (df.rsi > rsi_high) & (df.rsi.shift(+1) <= rsi_high)
    
    # Entradas
    df['position'] = 0
    
    if position_type == "short":
        df.loc[cond1, "position"] = -1
    elif position_type == "long":
        df.loc[cond2, "position"] = +1
    else:  # "both" ou qualquer outro valor padrão
        df.loc[cond1, "position"] = -1
        df.loc[cond2, "position"] = +1
    
    #Não temos posições muito cedo no dia                
    #df.loc[(df.index.to_series().dt.hour==9) & (df.index.to_series().dt.minute <= 10), 'position'] = 0
    
    # Restrição de horários
    if allowed_hours is not None:
        # Zera posição fora dos horários permitidos
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
  
    
    
    return df['position']

def pattern_rsi_trend(df, length_rsi, rsi_low, rsi_high, allowed_hours=None, position_type="both"):
    """
    Estratégia de entrada baseada na variação percentual de preços e RSI inverso.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        allowed_hours (list): Lista de horas permitidas para operar.
        position_type (str): Tipo de posição permitida: "long", "short" ou "both".
        length_rsi (int): Período para cálculo do RSI.
        rsi_low (int): Nível de sobrevenda do RSI (para entrar vendido).
        rsi_high (int): Nível de sobrecompra do RSI (para entrar comprado).
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    
    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    # Calcular a variação percentual
    df['pct_change'] = df['close'].pct_change().fillna(0)
    
    # Calcula o RSI
    df['rsi'] = df.ta.rsi(length= length_rsi).fillna(0)
    
    # Determinar posições com base no position_type e RSI (lógica inversa)
    if position_type == "long":
        df['position'] = np.where((df['pct_change'] > 0) & (df['rsi'] > rsi_high), 1, 0)
    elif position_type == "short":
        df['position'] = np.where((df['pct_change'] < 0) & (df['rsi'] < rsi_low), -1, 0)
    else:  # "both" ou qualquer outro valor padrão
        long_condition = (df['pct_change'] > 0) & (df['rsi'] > rsi_high)
        short_condition = (df['pct_change'] < 0) & (df['rsi'] < rsi_low)
        
        df['position'] = np.where(long_condition, 1, np.where(short_condition, -1, 0))
    
    # Restrição de horários
    if allowed_hours is not None:
        # Zera posição fora dos horários permitidos
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
    
    return df['position']


def pattern_rsi_anti_trend(df, length_rsi, rsi_low, rsi_high, allowed_hours=None, position_type="both"):
    """
    Estratégia de entrada baseada na variação percentual de preços e RSI  - contra tendência.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        allowed_hours (list): Lista de horas permitidas para operar.
        position_type (str): Tipo de posição permitida: "long", "short" ou "both".
        length_rsi (int): Período para cálculo do RSI.
        rsi_low (int): Nível de sobrevenda do RSI (para entrar vendido).
        rsi_high (int): Nível de sobrecompra do RSI (para entrar comprado).
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    
    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    # Calcular a variação percentual
    df['pct_change'] = df['close'].pct_change().fillna(0)
    
    # Calcula o RSI
    df['rsi'] = df.ta.rsi(length= length_rsi).fillna(0)
    
    # Determinar posições com base no position_type e RSI (lógica inversa)
    if position_type == "short":
        df['position'] = np.where((df['pct_change'] > 0) & (df['rsi'] > rsi_high), -1, 0)
    elif position_type == "long":
        df['position'] = np.where((df['pct_change'] < 0) & (df['rsi'] < rsi_low), +1, 0)
    else:  # "both" ou qualquer outro valor padrão
        long_condition = (df['pct_change'] > 0) & (df['rsi'] > rsi_high)
        short_condition = (df['pct_change'] < 0) & (df['rsi'] < rsi_low)
        
        df['position'] = np.where(long_condition, -1, np.where(short_condition, +1, 0))
    
    # Restrição de horários
    if allowed_hours is not None:
        # Zera posição fora dos horários permitidos
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
    
    return df['position']


def bb_trend(df, bb_length, std, allowed_hours=None, position_type="both"):
    """
    Estratégia baseada em Bandas de Bollinger.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        bb_length (int): Período para cálculo da média e desvio padrão.
        std (float): Número de desvios padrão para as bandas.
        allowed_hours (list): Horas que vamos executar a estratégia.
        position_type (str): Tipo de posições permitidas:
                            - "long": Apenas posições de compra (+1)
                            - "short": Apenas posições de venda (-1)
                            - "both": Ambas as posições (padrão)
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    aux = df.ta.bbands(length=bb_length, std=std)
    df[f"BBL_{bb_length}_{std}"] = aux[f"BBL_{bb_length}_{std}"]
    df[f"BBU_{bb_length}_{std}"] = aux[f"BBU_{bb_length}_{std}"]
    df[f"BBM_{bb_length}_{std}"] = aux[f"BBM_{bb_length}_{std}"]    
    
    # Inicializar a coluna de posição com zeros
    df['position'] = 0
    
    # Calculando entradas (buy/sell)
    cond1 = (df.close < df[f"BBL_{bb_length}_{std}"]) & (df.close.shift(+1) >= df[f"BBL_{bb_length}_{std}"].shift(+1))
    cond2 = (df.close > df[f"BBU_{bb_length}_{std}"]) & (df.close.shift(+1) <= df[f"BBU_{bb_length}_{std}"].shift(+1))
    
    # Aplicar as posições de acordo com o parâmetro position_type
    if position_type.lower() == "both":
        df.loc[cond1, "position"] = -1
        df.loc[cond2, "position"] = +1
    elif position_type.lower() == "long":
        df.loc[cond2, "position"] = +1
    elif position_type.lower() == "short":
        df.loc[cond1, "position"] = -1
    else:
        raise ValueError("position_type deve ser 'long', 'short' ou 'both'")
    
    # Restrição de horários
    if allowed_hours is not None:
        # Zera posição fora dos horários permitidos
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
    
    return df['position']


def bb_anti_trend(df, bb_length, std, allowed_hours=None, position_type="both"):
    """
    Estratégia baseada em Bandas de Bollinger.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        bb_length (int): Período para cálculo da média e desvio padrão.
        std (float): Número de desvios padrão para as bandas.
        allowed_hours (list): Horas que vamos executar a estratégia.
        position_type (str): Tipo de posições permitidas:
                            - "long": Apenas posições de compra (+1)
                            - "short": Apenas posições de venda (-1)
                            - "both": Ambas as posições (padrão)
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    aux = df.ta.bbands(length=bb_length, std=std)
    df[f"BBL_{bb_length}_{std}"] = aux[f"BBL_{bb_length}_{std}"]
    df[f"BBU_{bb_length}_{std}"] = aux[f"BBU_{bb_length}_{std}"]
    df[f"BBM_{bb_length}_{std}"] = aux[f"BBM_{bb_length}_{std}"]    
    
    # Inicializar a coluna de posição com zeros
    df['position'] = 0
    
    # Calculando entradas (buy/sell)
    cond1 = (df.close < df[f"BBL_{bb_length}_{std}"]) & (df.close.shift(+1) >= df[f"BBL_{bb_length}_{std}"].shift(+1))
    cond2 = (df.close > df[f"BBU_{bb_length}_{std}"]) & (df.close.shift(+1) <= df[f"BBU_{bb_length}_{std}"].shift(+1))
    
    # Aplicar as posições de acordo com o parâmetro position_type
    if position_type.lower() == "both":
        df.loc[cond1, "position"] = +1
        df.loc[cond2, "position"] = -1
    elif position_type.lower() == "short":
        df.loc[cond2, "position"] = -1
    elif position_type.lower() == "long":
        df.loc[cond1, "position"] = +1
    else:
        raise ValueError("position_type deve ser 'long', 'short' ou 'both'")
    
    # Restrição de horários
    if allowed_hours is not None:
        # Zera posição fora dos horários permitidos
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
    
    return df['position']




def macd_crossover_trend(df, fast_period, slow_period, signal_period, allowed_hours=None, position_type="both"):
    """
    Estratégia baseada no cruzamento do MACD com sua linha de sinal.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        fast_period (int): Período da média móvel rápida (ex: 12).
        slow_period (int): Período da média móvel lenta (ex: 26).
        signal_period (int): Período da linha de sinal (ex: 9).
        allowed_hours (list): Lista de horas permitidas para operar.
        position_type (str): Tipo de posição permitida: "long", "short" ou "both".
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    
    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    # Calcular MACD
    macd_result = df.ta.macd(fast=fast_period, slow=slow_period, signal=signal_period)
    df['macd'] = macd_result[f'MACD_{fast_period}_{slow_period}_{signal_period}']
    df['macd_signal'] = macd_result[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
    df['macd_hist'] = macd_result[f'MACDh_{fast_period}_{slow_period}_{signal_period}']
    
    # Preencher NaN com zeros para evitar problemas
    df['macd'] = df['macd'].fillna(0)
    df['macd_signal'] = df['macd_signal'].fillna(0)
    
    # Inicializar posições
    df['position'] = 0
    
    # Calculando entradas (buy/sell)
    cond1 = (df['macd'] < df['macd_signal']) & (df['macd'].shift(+1) > df['macd_signal'].shift(+1))
    cond2 = (df['macd'] > df['macd_signal']) & (df['macd'].shift(+1) <= df['macd_signal'].shift(+1))
    
    # Aplicar as posições de acordo com o parâmetro position_type
    if position_type.lower() == "both":
        df.loc[cond1, "position"] = -1
        df.loc[cond2, "position"] = +1
    elif position_type.lower() == "long":
        df.loc[cond2, "position"] = +1
    elif position_type.lower() == "short":
        df.loc[cond1, "position"] = -1
    else:
        raise ValueError("position_type deve ser 'long', 'short' ou 'both'")
    
    # Restrição de horários
    if allowed_hours is not None:
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
    
    return df['position']


def macd_crossover_anti_trend(df, fast_period, slow_period, signal_period, allowed_hours=None, position_type="both"):
    """
    Estratégia baseada no cruzamento do MACD com sua linha de sinal.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        fast_period (int): Período da média móvel rápida (ex: 12).
        slow_period (int): Período da média móvel lenta (ex: 26).
        signal_period (int): Período da linha de sinal (ex: 9).
        allowed_hours (list): Lista de horas permitidas para operar.
        position_type (str): Tipo de posição permitida: "long", "short" ou "both".
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    
    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    # Calcular MACD
    macd_result = df.ta.macd(fast=fast_period, slow=slow_period, signal=signal_period)
    df['macd'] = macd_result[f'MACD_{fast_period}_{slow_period}_{signal_period}']
    df['macd_signal'] = macd_result[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
    df['macd_hist'] = macd_result[f'MACDh_{fast_period}_{slow_period}_{signal_period}']
    
    # Preencher NaN com zeros para evitar problemas
    df['macd'] = df['macd'].fillna(0)
    df['macd_signal'] = df['macd_signal'].fillna(0)
    
    # Inicializar posições
    df['position'] = 0
    
    # Calculando entradas (buy/sell)
    cond1 = (df['macd'] < df['macd_signal']) & (df['macd'].shift(+1) > df['macd_signal'].shift(+1))
    cond2 = (df['macd'] > df['macd_signal']) & (df['macd'].shift(+1) <= df['macd_signal'].shift(+1))
    
    # Aplicar as posições de acordo com o parâmetro position_type
    if position_type.lower() == "both":
        df.loc[cond1, "position"] = +1
        df.loc[cond2, "position"] = -1
    elif position_type.lower() == "short":
        df.loc[cond2, "position"] = -1
    elif position_type.lower() == "long":
        df.loc[cond1, "position"] = +1
    else:
        raise ValueError("position_type deve ser 'long', 'short' ou 'both'")
    
    # Restrição de horários
    if allowed_hours is not None:
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
    
    return df['position']

def momentum_breakout(df, lookback_period, momentum_threshold, volume_factor=1.5, allowed_hours=None, position_type="both"):
    """
    Estratégia de breakout baseada em momentum e volume.
    Identifica movimentos fortes com confirmação de volume.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC e volume.
        lookback_period (int): Período para calcular o momentum (ex: 20).
        momentum_threshold (float): Threshold percentual para considerar momentum forte (ex: 0.02 para 2%).
        volume_factor (float): Fator multiplicador do volume médio para confirmação (ex: 1.5).
        allowed_hours (list): Lista de horas permitidas para operar.
        position_type (str): Tipo de posição permitida: "long", "short" ou "both".
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    
    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    # Calcular momentum (taxa de mudança)
    df['momentum'] = (df['close'] - df['close'].shift(lookback_period)) / df['close'].shift(lookback_period)
    df['momentum'] = df['momentum'].fillna(0)
    
    # Calcular volume médio
    df['avg_volume'] = df['volume'].rolling(window=lookback_period).mean()
    df['avg_volume'] = df['avg_volume'].fillna(df['volume'].mean())
    
    # Condições de volume alto
    high_volume = df['volume'] > (df['avg_volume'] * volume_factor)
    
    # Inicializar posições
    df['position'] = 0
    
    # Condições de momentum forte
    strong_up_momentum = (df['momentum'] > momentum_threshold) & high_volume
    strong_down_momentum = (df['momentum'] < -momentum_threshold) & high_volume
    
    # Aplicar posições baseadas no tipo permitido
    if position_type == "long":
        df.loc[strong_up_momentum, 'position'] = 1
    elif position_type == "short":
        df.loc[strong_down_momentum, 'position'] = -1
    else:  # "both"
        df.loc[strong_up_momentum, 'position'] = 1
        df.loc[strong_down_momentum, 'position'] = -1
    
    # Restrição de horários
    if allowed_hours is not None:
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0
    
    return df['position']