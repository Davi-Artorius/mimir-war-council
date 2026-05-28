"""
FORGE SPIRIT — SMC VALIDATION GATE v1.1
=========================================
Módulo de validação determinística completa antes de convocar o Conselho.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import os

# ─── PARÂMETROS DO SISTEMA (espelho fiel do SOP v1.1) ────────────────────────
TIER1_WICK_THRESHOLD   = 0.35   # pavio M15 > 35% do range
TIER1_DOMINANCE_RATIO  = 2.0    # pavio favor >= 2x pavio oposto
TIER2_WICK_THRESHOLD   = 0.15   # threshold mínimo Tier 2 (15% do range)

# Filtro de Sangue (Regra Mimir): Máximo spread para Sniper
# Ajustado para Funding Pips ($0.14 - $0.25 normal)
MAX_SPREAD_POINTS      = 30.0   # 3 pips ($0.30 de spread no Ouro)

# Sweep: tolerância para considerar que um nível foi varrido
SWEEP_TOLERANCE_PIPS   = 0.5    # 0.5 pip = $0.05 no XAUUSD

# POI: distância máxima do preço atual ao POI para ser considerado "em zona"
POI_PROXIMITY_PIPS     = 3.0    # 3 pips de tolerância

# Estrutura: quantas velas M15 olhar para trás para validar BOS/ChoCH
STRUCTURE_LOOKBACK     = 50     # ~12 horas de M15

# FVG: quantas velas atrás ainda é considerado FVG "fresco"
FVG_MAX_AGE_CANDLES    = 20     # ~5 horas de M15

@dataclass
class ValidationResult:
    approved:          bool
    tier:              Optional[str]        # 'TIER1' | 'TIER2' | None
    direction:         Optional[str]        # 'BULL' | 'BEAR' | None
    wick_ratio:        float = 0.0
    dominance:         float = 0.0
    sweep_confirmed:   bool  = False
    sweep_level:       float = 0.0
    sweep_type:        str   = ""           # 'EQH' | 'EQL' | 'PDH' | 'PDL' | 'LAST_HIGH' | 'LAST_LOW'
    structure_aligned: bool  = False
    structure_trend:   str   = ""
    poi_hit:           bool  = False
    poi_type:          str   = ""           # 'FVG' | 'OB' | 'DEMAND' | 'SUPPLY'
    poi_level:         float = 0.0
    structure_break:   bool  = False
    break_type:        str   = ""           # 'BOS' | 'CHoCH'
    break_level:       float = 0.0
    spread_ok:         bool  = True
    mimese_tecnica:    str   = ""           # Resumo para o Oracle
    reject_reason:     str   = ""          # motivo do veto determinístico

    def to_dict(self) -> dict:
        return {
            "approved":          self.approved,
            "tier":              self.tier,
            "direction":         self.direction,
            "wick_ratio":        round(self.wick_ratio, 3),
            "dominance":         round(self.dominance, 2),
            "sweep_confirmed":   self.sweep_confirmed,
            "sweep_level":       self.sweep_level,
            "sweep_type":        self.sweep_type,
            "structure_aligned": self.structure_aligned,
            "structure_trend":   self.structure_trend,
            "poi_hit":           self.poi_hit,
            "poi_type":          self.poi_type,
            "poi_level":         self.poi_level,
            "structure_break":   self.structure_break,
            "break_type":        self.break_type,
            "break_level":       self.break_level,
            "spread_ok":         self.spread_ok,
            "mimese_tecnica":    self.mimese_tecnica,
            "reject_reason":     self.reject_reason,
        }

class SMCValidationGate:
    def __init__(self, smc_engine):
        self.engine = smc_engine

    def validate(self, df_15m, df_5m, current_price, spread_points=0.0) -> dict:
        # ── ETAPA -1: SPREAD (Filtro de Sangue)
        if spread_points > MAX_SPREAD_POINTS:
            return ValidationResult(
                approved=False, tier=None, direction=None, spread_ok=False,
                reject_reason=f"SPREAD: {spread_points} pontos (Acima do limite de {MAX_SPREAD_POINTS})"
            ).to_dict()

        # ── ETAPA 0: PAVIO
        wick_result = self._detect_wick(df_15m)
        if not wick_result:
            return ValidationResult(approved=False, tier=None, direction=None, reject_reason="PAVIO: Inexistente no M15").to_dict()

        direction = wick_result["direction"]
        wick_ratio = wick_result["wick_ratio"]
        dominance = wick_result["dominance"]
        is_tier1 = wick_result["is_tier1"]

        # ── ETAPA 1: SWEEP (OBRIGATÓRIO PARA SOBERANIA)
        sweep_result = self._detect_sweep(df_15m, current_price, direction)
        if not sweep_result["confirmed"]:
            return ValidationResult(
                approved=False, tier=None, direction=direction, 
                reject_reason="LIQUIDEZ: Nenhuma varredura (Sweep) detectada em POI Macro."
            ).to_dict()
            
        tier = "TIER1" if is_tier1 else "TIER2"

        # ── ETAPA 2: ESTRUTURA
        struct_15m = self.engine.analyze_market_structure(df_15m)
        structure_trend = struct_15m.get("trend", "NEUTRAL")
        structure_aligned = (structure_trend == direction)

        if not structure_aligned and tier == "TIER1":
            return ValidationResult(approved=False, tier=tier, direction=direction, reject_reason=f"ESTRUTURA: M15 em {structure_trend}, desalinhado.").to_dict()

        # ── ETAPA 3: POI
        poi_result = self._detect_poi(df_15m, current_price, direction)
        if not poi_result["hit"] and tier == "TIER1":
            return ValidationResult(approved=False, tier=tier, direction=direction, reject_reason="POI: Preço fora de zona de interesse válida.").to_dict()

        # ── ETAPA 4: EVENTOS DE ESTRUTURA (BOS/CHoCH) PARA O SCOUT
        structure_break = False
        break_type = ""
        break_level = 0.0
        
        # Verifica se houve CHoCH ou BOS nos últimos 2 candles
        if struct_15m.get("choch"):
            last_choch = struct_15m["choch"][-1]
            structure_break = True
            break_type = "CHoCH"
            break_level = last_choch["price"]
        elif struct_15m.get("bos"):
            last_bos = struct_15m["bos"][-1]
            structure_break = True
            break_type = "BOS"
            break_level = last_bos["price"]

        # ── GERAÇÃO DE MIMESE TÉCNICA
        mimese = f"Setup {tier} {direction}. "
        if sweep_result["confirmed"]: mimese += f"Sweep em {sweep_result['type']}. "
        if poi_result["hit"]: mimese += f"POI {poi_result['type']} mitigado. "
        mimese += f"Pavio: {wick_ratio*100:.0f}%."

        return ValidationResult(
            approved=True, tier=tier, direction=direction, wick_ratio=wick_ratio, dominance=dominance,
            sweep_confirmed=sweep_result["confirmed"], sweep_level=sweep_result.get("level", 0.0),
            sweep_type=sweep_result.get("type", ""), structure_aligned=structure_aligned,
            structure_trend=structure_trend, poi_hit=poi_result["hit"], poi_type=poi_result.get("type", ""),
            poi_level=poi_result.get("level", 0.0), structure_break=structure_break,
            break_type=break_type, break_level=break_level, mimese_tecnica=mimese
        ).to_dict()

    def _detect_wick(self, df):
        if len(df) < 2: return None
        row = df.iloc[-2]
        total = row["High"] - row["Low"]
        if total < 1e-6: return None
        bt, bb = max(row["Open"], row["Close"]), min(row["Open"], row["Close"])
        uw, lw = row["High"] - bt, bb - row["Low"]

        results = []
        if uw / total >= TIER2_WICK_THRESHOLD:
            results.append({"direction": "BEAR", "wick_ratio": uw/total, "dominance": uw/(lw+1e-9), "is_tier1": (uw/total >= TIER1_WICK_THRESHOLD)})
        if lw / total >= TIER2_WICK_THRESHOLD:
            results.append({"direction": "BULL", "wick_ratio": lw/total, "dominance": lw/(uw+1e-9), "is_tier1": (lw/total >= TIER1_WICK_THRESHOLD)})

        if not results: return None
        return max(results, key=lambda x: x["wick_ratio"])

    def _detect_sweep(self, df, current_price, direction):
        lookback = df.tail(STRUCTURE_LOOKBACK)
        levels = []
        try:
            struct = self.engine.analyze_market_structure(lookback)
            if struct.get("last_high"): levels.append({"price": struct["last_high"], "type": "LAST_HIGH"})
            if struct.get("last_low"): levels.append({"price": struct["last_low"], "type": "LAST_LOW"})
            pools = self.engine.detect_liquidity_pools(lookback)
            for p in pools: levels.append({"price": p["price"], "type": p["type"]})
            # Asian Range
            asian = df[df.index.hour < 7]
            if not asian.empty:
                levels.append({"price": asian["High"].max(), "type": "PDH"})
                levels.append({"price": asian["Low"].min(),  "type": "PDL"})
        except: pass

        # Candle anterior (Internal Candle-by-Candle Sweep para tendências)
        if len(df) >= 3:
            prev_row = df.iloc[-3]
            levels.append({"price": prev_row["High"], "type": "PREV_CANDLE_HIGH"})
            levels.append({"price": prev_row["Low"], "type": "PREV_CANDLE_LOW"})

        row = df.iloc[-2]
        tol = SWEEP_TOLERANCE_PIPS * 0.1
        for lvl in levels:
            if direction == "BEAR" and row["High"] >= lvl["price"] - tol and row["Close"] < lvl["price"]:
                return {"confirmed": True, "level": lvl["price"], "type": lvl["type"]}
            if direction == "BULL" and row["Low"] <= lvl["price"] + tol and row["Close"] > lvl["price"]:
                return {"confirmed": True, "level": lvl["price"], "type": lvl["type"]}
        return {"confirmed": False}


    def _detect_poi(self, df, current_price, direction):
        tol = POI_PROXIMITY_PIPS * 0.1
        try:
            fvgs = self.engine.detect_fvg(df)
            for fvg in fvgs[-FVG_MAX_AGE_CANDLES:]:
                if fvg["type"] == direction and (fvg["bottom"] - tol) <= current_price <= (fvg["top"] + tol):
                    return {"hit": True, "type": "FVG", "level": (fvg["top"] + fvg["bottom"])/2}
            zones = self.engine.detect_wick_demand(df, side=direction)
            for zone in zones:
                if abs(current_price - zone["price"]) <= tol * 3:
                    return {"hit": True, "type": "OB", "level": zone["price"]}
        except: pass
        return {"hit": False, "type": "", "level": 0.0}
