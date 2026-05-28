import pandas as pd
import numpy as np

class MimirSMCEngine:
    """
    Motor de Detecção SMC Soberano v4.0 (Vectorized by smc-toolkit logic)
    Focado em Precisão 100%: Estrutura, FVG e Liquidez de Engenharia.
    Integrado com a matemática do repositório Louisjzhao/smc-toolkit.
    """
    def __init__(self, swing_length=5):
        # swing_length equivalente ao swing_size do toolkit
        self.swing_length = swing_length
        self.liquidity_threshold = 0.50 # Tolerância para Equal Highs/Lows no Ouro

    def _calc_swing_structures_vectorized(self, df, size=None, prefix=""):
        """Implementação vetorizada que suporta estrutura principal e interna."""
        df = df.copy()
        df.columns = [c.capitalize() for c in df.columns] 
        size = size if size else self.swing_length
        
        swing_pre = pd.Series(0, index=df.index)
        future_high = df['High'].shift(-size).rolling(size).max()
        future_low = df['Low'].shift(-size).rolling(size).min()
        past_high = df['High'].rolling(size).max().shift(1)
        past_low = df['Low'].rolling(size).min().shift(1)
        
        swing_pre[(df['High'] > future_high) & (df['High'] > past_high)] = 1
        swing_pre[(df['Low'] < future_low) & (df['Low'] < past_low)] = -1
        
        swing_hl_sim = swing_pre.replace(0, np.nan).ffill().replace({1: 0, -1: 1}).fillna(0)
        swing_h_l = swing_hl_sim.diff()
        
        swing_high_level = pd.Series(np.where(swing_h_l == -1, df['High'], np.nan), index=df.index).ffill()
        swing_low_level = pd.Series(np.where(swing_h_l == 1, df['Low'], np.nan), index=df.index).ffill()
        
        bullish_bos = (df['Close'] > swing_high_level) & (df['Close'].shift(1) <= swing_high_level)
        bearish_bos = (df['Close'] < swing_low_level) & (df['Close'].shift(1) >= swing_low_level)
        
        bos = pd.Series(0, index=df.index)
        bos[bullish_bos] = 1
        bos[bearish_bos] = -1
        
        trend = bos.replace(0, np.nan).ffill().fillna(0)
        trend_shift = trend.diff()
        
        choch = pd.Series(0, index=df.index)
        choch[trend_shift.isin([-2, 2])] = trend_shift[trend_shift.isin([-2, 2])]
        
        res = pd.DataFrame(index=df.index)
        res[f'{prefix}high_level'] = swing_high_level
        res[f'{prefix}low_level'] = swing_low_level
        res[f'{prefix}bos'] = bos
        res[f'{prefix}choch'] = choch
        res[f'{prefix}trend'] = trend
        res[f'{prefix}swing_pre'] = swing_pre
        
        return res

    def analyze_market_structure(self, df):
        """Mapeia estrutura fractal: Swing (Macro) vs Internal (Micro)."""
        if len(df) < 50: # Mínimo para análise estável
            return {"trend": "NEUTRAL", "last_high": 0, "last_low": 0, "bos": [], "choch": []}
            
        # 1. Estrutura Principal (Swing - Filtro de Ruído)
        main_struct = self._calc_swing_structures_vectorized(df, size=self.swing_length, prefix="main_")
        
        # 2. Estrutura Interna (Micro - Gatilho de confirmação rápida)
        int_struct = self._calc_swing_structures_vectorized(df, size=2, prefix="int_")
        
        combined = pd.concat([df, main_struct, int_struct], axis=1)
        
        trend_val = combined['main_trend'].iloc[-1]
        int_trend_val = combined['int_trend'].iloc[-1]
        
        # Tendência composta
        trend_str = "BULL" if trend_val == 1 else "BEAR" if trend_val == -1 else "NEUTRAL"
        int_trend_str = "BULL" if int_trend_val == 1 else "BEAR" if int_trend_val == -1 else "NEUTRAL"
        
        last_high = combined['main_high_level'].iloc[-1]
        last_low = combined['main_low_level'].iloc[-1]
        
        bos_events = []
        choch_events = []
        
        recent = combined.tail(100)
        # Prioriza CHoCH interno como sinal de reversão rápida para o servidor
        for time, row in recent[recent['int_choch'] != 0].iterrows():
            choch_events.append({
                "price": row['Close'], 
                "direction": "BULL" if row['int_choch'] == 2 else "BEAR", 
                "time": time,
                "type": "INTERNAL"
            })
            
        return {
            "trend": trend_str, 
            "internal_trend": int_trend_str,
            "last_high": last_high, 
            "last_low": last_low, 
            "bos": bos_events, 
            "choch": choch_events
        }

    def get_last_wick_ratio(self, df):
        try:
            if df is None or len(df) < 1: return 0.0
            last = df.iloc[-1]
            candle_range = last['High'] - last['Low']
            if candle_range <= 0: return 0.0
            
            upper_wick = last['High'] - max(last['Open'], last['Close'])
            lower_wick = min(last['Open'], last['Close']) - last['Low']
            return (upper_wick + lower_wick) / candle_range
        except Exception:
            return 0.0

    def detect_liquidity_pools(self, df):
        """Identifica EQH/EQL usando os swings limpos vetorizados."""
        if len(df) < self.swing_length * 2: return []
        v_df = self._calc_swing_structures_vectorized(df)
        
        swings_high = v_df[v_df['Swing_Pre'] == 1]['High'].tail(10)
        swings_low = v_df[v_df['Swing_Pre'] == -1]['Low'].tail(10)
        
        pools = []
        sh_prices = swings_high.values
        sh_times = swings_high.index
        for i in range(len(sh_prices)):
            for j in range(i + 1, len(sh_prices)):
                if abs(sh_prices[i] - sh_prices[j]) <= self.liquidity_threshold:
                    pools.append({
                        "type": "EQH", 
                        "price": max(sh_prices[i], sh_prices[j]),
                        "strength": 2,
                        "time": sh_times[j]
                    })
                    
        sl_prices = swings_low.values
        sl_times = swings_low.index
        for i in range(len(sl_prices)):
            for j in range(i + 1, len(sl_prices)):
                if abs(sl_prices[i] - sl_prices[j]) <= self.liquidity_threshold:
                    pools.append({
                        "type": "EQL", 
                        "price": min(sl_prices[i], sl_prices[j]),
                        "strength": 2,
                        "time": sl_times[j]
                    })
        return pools

    def detect_fvg(self, df):
        """Identifica FVG rigorosos inspirados na lógica do smc-toolkit."""
        if len(df) < 3: return []
        df_c = df.copy()
        df_c.columns = [c.capitalize() for c in df_c.columns]
        
        # Filtro de gap real e expensão
        bar_delta = (df_c['Close'].shift(1) - df_c['Open'].shift(1)).abs()
        threshold = bar_delta.expanding().mean() * 1.5 # Expansão razoável
        
        bullish_mask = (df_c['Low'] > df_c['High'].shift(2)) & (df_c['Close'].shift(1) > df_c['High'].shift(2)) & (bar_delta > threshold)
        bearish_mask = (df_c['High'] < df_c['Low'].shift(2)) & (df_c['Close'].shift(1) < df_c['Low'].shift(2)) & (bar_delta > threshold)
        
        fvgs = []
        for time, row in df_c[bullish_mask].iterrows():
            idx = df_c.index.get_loc(time)
            top = row['Low']
            bottom = df_c['High'].iloc[idx-2]
            fvgs.append({"type": "BULL", "top": top, "bottom": bottom, "time": df_c.index[idx-1], "mitigated": False})
            
        for time, row in df_c[bearish_mask].iterrows():
            idx = df_c.index.get_loc(time)
            bottom = row['High']
            top = df_c['Low'].iloc[idx-2]
            fvgs.append({"type": "BEAR", "top": top, "bottom": bottom, "time": df_c.index[idx-1], "mitigated": False})
            
        # Avaliar mitigação nos mais recentes
        for gap in fvgs:
            gap_time = gap['time']
            future_df = df_c[df_c.index > gap_time]
            mid = (gap['top'] + gap['bottom']) / 2
            if gap['type'] == 'BULL':
                if (future_df['Low'] <= mid).any(): gap['mitigated'] = True
            else:
                if (future_df['High'] >= mid).any(): gap['mitigated'] = True
                
        # Retorna apenas não mitigados ou todos dependendo da lógica (aqui filtramos mitigados pra limpeza)
        return [g for g in fvgs if not g['mitigated']]

    def project_daily_narrative(self, df_d1):
        if len(df_d1) < 2: return {"scenario": "UNKNOWN", "bias": "NEUTRAL"}
        prev = df_d1.iloc[-2]
        pprev = df_d1.iloc[-3]
        body = abs(prev['Close'] - prev['Open'])
        candle_range = prev['High'] - prev['Low']
        if candle_range == 0: return {"scenario": "UNKNOWN", "bias": "NEUTRAL"}
        
        body_percent = body / candle_range
        upper_wick = prev['High'] - max(prev['Open'], prev['Close'])
        lower_wick = min(prev['Open'], prev['Close']) - prev['Low']
        
        swept_high = prev['High'] > pprev['High'] and prev['Close'] < pprev['High']
        swept_low = prev['Low'] < pprev['Low'] and prev['Close'] > pprev['Low']
        
        if swept_high and upper_wick / candle_range > 0.40:
            return {"scenario": "SWEEP_REVERSAL", "bias": "BEAR", "reason": "Varredura de topo D1 com forte rejeição."}
        if swept_low and lower_wick / candle_range > 0.40:
            return {"scenario": "SWEEP_REVERSAL", "bias": "BULL", "reason": "Varredura de fundo D1 com forte rejeição."}
        if body_percent > 0.60:
            direction = "BULL" if prev['Close'] > prev['Open'] else "BEAR"
            return {"scenario": "TREND_CONTINUATION", "bias": direction, "reason": "Vela de convicção (corpo > 60%). Fluxo institucional forte."}
        return {"scenario": "RETEST_WAIT", "bias": "NEUTRAL", "reason": "Indecisão ou necessidade de reteste de zona. Aguardar sweep intraday."}

    def detect_wick_demand(self, df, side="BULL"):
        """Identifica zonas de Supply/Demand baseadas em pavios de exaustão."""
        if len(df) < 5: return []
        zones = []
        lookback = df.tail(20)
        
        for time, row in lookback.iterrows():
            candle_range = row['High'] - row['Low']
            if candle_range <= 0: continue
            
            upper_wick = row['High'] - max(row['Open'], row['Close'])
            lower_wick = min(row['Open'], row['Close']) - row['Low']
            
            if side == "BULL" and (lower_wick / candle_range) > 0.4:
                zones.append({"price": row['Low'], "time": time, "type": "DEMAND"})
            elif side == "BEAR" and (upper_wick / candle_range) > 0.4:
                zones.append({"price": row['High'], "time": time, "type": "SUPPLY"})
                
        return zones

    def detect_wick_rejection(self, df, structure_price=None, side="BULL", threshold=0.35):
        if df.empty: return False
        latest = df.iloc[-1]
        candle_range = latest['High'] - latest['Low']
        if candle_range == 0: return False
        
        upper_wick = latest['High'] - max(latest['Open'], latest['Close'])
        lower_wick = min(latest['Open'], latest['Close']) - latest['Low']
        
        if side == "BULL":
            if (lower_wick / candle_range) >= threshold and latest['Close'] > latest['Low']:
                return True
        else:
            if (upper_wick / candle_range) >= threshold and latest['Close'] < latest['High']:
                return True
        return False
