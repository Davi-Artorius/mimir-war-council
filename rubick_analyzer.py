"""
RUBICK WINNER ANALYZER v1.1
=============================
Analisa o backtest_results.csv e identifica as condições específicas
em que o pavio tem valor preditivo real.

CORREÇÕES v1.1:
- Timestamps em UTC corrigidos para sessões reais
- NY = 14:00-19:00 UTC (11:00-16:00 Brasília) ← FOCO DO ARQUITETO
- Londres = 08:00-13:00 UTC (05:00-10:00 Brasília)
- Análise separada NY vs global

USO:
    python rubick_analyzer.py

REQUISITO:
    backtest_results.csv gerado pelo forge_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─── CONFIGURAÇÃO ─────────────────────────────────────────────────────────────
BACKTEST_CSV = Path(
    "/home/mimir/Documentos/MIMIR/06_LABORATORIO_ESTUDOS/"
    "MIMIR_XAUUSD_SOVEREIGN/backtest_results.csv"
)

OUTPUT_REPORT = Path(
    "/home/mimir/Documentos/MIMIR/06_LABORATORIO_ESTUDOS/"
    "MIMIR_XAUUSD_SOVEREIGN/rubick_winner_map.md"
)

# ─── CARREGAMENTO ─────────────────────────────────────────────────────────────
def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERRO] Arquivo não encontrado: {path}")
        print("Execute forge_backtest.py primeiro.")
        raise SystemExit(1)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df_valid = df[df["outcome"] != "TIMEOUT"].copy()

    print(f"[RUBICK] {len(df)} trades carregados | "
          f"{len(df_valid)} válidos (excluindo timeouts)")
    return df_valid

# ─── ENRIQUECIMENTO ───────────────────────────────────────────────────────────
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour_utc"]     = df["timestamp"].dt.hour
    df["weekday_name"] = df["timestamp"].dt.day_name()

    # Sessões em UTC
    # Asia:     00:00-07:00 UTC  (21:00-04:00 Brasília)
    # Londres:  08:00-13:00 UTC  (05:00-10:00 Brasília)
    # Overlap:  13:00-14:00 UTC  (10:00-11:00 Brasília)
    # NY:       14:00-19:00 UTC  (11:00-16:00 Brasília) ← FOCO
    # Morto:    19:00-24:00 UTC  (16:00-21:00 Brasília)
    def get_session(h):
        if 14 <= h < 19: return "NY"
        if 13 <= h < 14: return "OVERLAP"
        if  8 <= h < 13: return "LONDON"
        if  0 <= h <  7: return "ASIA"
        return "DEAD"

    df["session"] = df["hour_utc"].apply(get_session)

    # Fase dentro de NY
    def get_ny_phase(row):
        if row["session"] != "NY":
            return "N/A"
        h = row["hour_utc"]
        if 14 <= h < 16: return "NY_OPEN"   # 11:00-13:00 Brasília
        if 16 <= h < 19: return "NY_MID"    # 13:00-16:00 Brasília
        return "N/A"

    df["ny_phase"] = df.apply(get_ny_phase, axis=1)

    # Faixas de pavio
    df["wick_ratio"] = df["wick_ratio"].astype(float)
    df["wick_band"]  = pd.cut(
        df["wick_ratio"],
        bins  =[0.35, 0.40, 0.50, 0.60, 1.0],
        labels=["35-40%", "40-50%", "50-60%", "60%+"]
    )

    # Faixas de dominância
    df["dominance"] = df["dominance"].astype(float)
    df["dom_band"]  = pd.cut(
        df["dominance"],
        bins  =[0, 2, 3, 5, 999],
        labels=["2-3x", "3-5x", "5-10x", "10x+"]
    )

    df["win"] = (df["outcome"] == "TP").astype(int)
    return df

# ─── ANÁLISE POR DIMENSÃO ─────────────────────────────────────────────────────
def analyze_dimension(df: pd.DataFrame, col: str,
                      label: str, baseline: float) -> pd.DataFrame:
    grp = df.groupby(col, observed=True).agg(
        total   =("win", "count"),
        wins    =("win", "sum"),
        wr      =("win", "mean"),
    ).reset_index()

    grp["wr_pct"] = (grp["wr"] * 100).round(1)
    grp["edge"]   = (grp["wr_pct"] - baseline).round(1)
    grp = grp.sort_values("wr_pct", ascending=False)

    print(f"\n{'='*60}")
    print(f"  {label}  [baseline: {baseline:.1f}%]")
    print(f"{'='*60}")
    print(f"  {'Valor':<15} {'Total':>7} {'Wins':>6} {'WR%':>7} {'Edge':>8}")
    print(f"  {'-'*55}")
    for _, r in grp.iterrows():
        edge_str = f"+{r['edge']:.1f}" if r['edge'] > 0 else f"{r['edge']:.1f}"
        flag = " ✅" if r['wr_pct'] >= 35 else (" ⚠️" if r['wr_pct'] >= 25 else " ❌")
        print(f"  {str(r[col]):<15} {r['total']:>7} {r['wins']:>6} "
              f"{r['wr_pct']:>6}% {edge_str:>7}{flag}")
    return grp

# ─── ANÁLISE CRUZADA ──────────────────────────────────────────────────────────
def cross_analysis(df: pd.DataFrame, col1: str, col2: str,
                   label: str, min_trades: int = 20):
    grp = df.groupby([col1, col2], observed=True).agg(
        total=("win", "count"),
        wins =("win", "sum"),
        wr   =("win", "mean"),
    ).reset_index()

    grp["wr_pct"] = (grp["wr"] * 100).round(1)
    grp = grp[grp["total"] >= min_trades]
    grp = grp.sort_values("wr_pct", ascending=False).head(10)

    print(f"\n{'='*65}")
    print(f"  {label}  (mín. {min_trades} trades)")
    print(f"{'='*65}")
    print(f"  {col1:<15} {col2:<15} {'Total':>7} {'Wins':>6} {'WR%':>7}")
    print(f"  {'-'*57}")
    for _, r in grp.iterrows():
        flag = " ✅" if r['wr_pct'] >= 35 else (" ⚠️" if r['wr_pct'] >= 25 else "")
        print(f"  {str(r[col1]):<15} {str(r[col2]):<15} "
              f"{r['total']:>7} {r['wins']:>6} {r['wr_pct']:>6}%{flag}")

# ─── FILTROS DE EDGE ──────────────────────────────────────────────────────────
def find_edge_filters(df: pd.DataFrame, min_wr: float = 35.0,
                      min_trades: int = 15) -> list:
    edge_filters = []
    for session in df["session"].unique():
        for tier in df["tier"].unique():
            for direction in df["direction"].unique():
                subset = df[
                    (df["session"]   == session) &
                    (df["tier"]      == tier) &
                    (df["direction"] == direction)
                ]
                if len(subset) < min_trades:
                    continue
                wr = subset["win"].mean() * 100
                if wr >= min_wr:
                    edge_filters.append({
                        "session":   session,
                        "tier":      tier,
                        "direction": direction,
                        "total":     len(subset),
                        "wr_pct":    round(wr, 1),
                    })
    return sorted(edge_filters, key=lambda x: x["wr_pct"], reverse=True)

# ─── RELATÓRIO MARKDOWN ───────────────────────────────────────────────────────
def export_markdown(df_all: pd.DataFrame, df_ny: pd.DataFrame,
                    edge_all: list, edge_ny: list, path: Path):

    baseline_all = df_all["win"].mean() * 100
    baseline_ny  = df_ny["win"].mean() * 100 if len(df_ny) > 0 else 0

    lines = [
        "# 🗺️ RUBICK WINNER MAP v1.1",
        "",
        f"> Gerado pelo Rubick Analyzer v1.1",
        f"> Total de trades válidos: {len(df_all)}",
        f"> WR baseline global: **{baseline_all:.1f}%**",
        f"> WR baseline NY: **{baseline_ny:.1f}%**",
        "",
        "---",
        "",
        "## ⏰ REFERÊNCIA DE SESSÕES",
        "",
        "| Sessão | UTC | Brasília |",
        "|--------|-----|----------|",
        "| Ásia | 00:00-07:00 | 21:00-04:00 |",
        "| Londres | 08:00-13:00 | 05:00-10:00 |",
        "| Overlap | 13:00-14:00 | 10:00-11:00 |",
        "| **NY** | **14:00-19:00** | **11:00-16:00** |",
        "| Morto | 19:00-24:00 | 16:00-21:00 |",
        "",
        "---",
        "",
        "## 🎯 FILTROS DE EDGE — NY (foco do Arquiteto)",
        "",
        "| Sessão | Tier | Direção | Total | WR% |",
        "|--------|------|---------|-------|-----|",
    ]

    if edge_ny:
        for f in edge_ny:
            lines.append(
                f"| {f['session']} | {f['tier']} | {f['direction']} "
                f"| {f['total']} | **{f['wr_pct']}%** |"
            )
    else:
        lines.append(
            "| NY | — | Nenhuma combinação atingiu 35% com amostra suficiente | — | — |"
        )

    lines += [
        "",
        "---",
        "",
        "## 🌍 FILTROS DE EDGE — TODAS AS SESSÕES",
        "",
        "| Sessão | Tier | Direção | Total | WR% |",
        "|--------|------|---------|-------|-----|",
    ]

    if edge_all:
        for f in edge_all:
            lines.append(
                f"| {f['session']} | {f['tier']} | {f['direction']} "
                f"| {f['total']} | **{f['wr_pct']}%** |"
            )
    else:
        lines.append("| — | Nenhuma combinação atingiu 35% | — | — | — |")

    lines += [
        "",
        "---",
        "",
        "## 📌 INSTRUÇÕES PARA O ORACLE",
        "",
        "```",
        "PRÉ-CONDIÇÃO OBRIGATÓRIA (antes de qualquer análise adversarial):",
        "",
        "1. Sessão ativa é NY? (14:00-19:00 UTC | 11:00-16:00 Brasília)",
        "   NÃO → VETO AUTOMÁTICO",
        "        O Arquiteto não opera fora de NY.",
        "",
        "2. Combinação session+tier+direction está no mapa com WR ≥ 35%?",
        "   NÃO → VETO RECOMENDADO (sem edge histórico nessa configuração)",
        "   SIM → Prosseguir com análise adversarial normal",
        "```",
        "",
        "---",
        "",
        "*Assinado: Rubick — O Historiador.*",
        "*3 anos de dor transformados em doutrina.*",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[RUBICK] Mapa exportado: {path}")

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RUBICK WINNER ANALYZER v1.1")
    print("  Timestamps UTC | Foco: Sessão NY (11:00-16:00 Brasília)")
    print("=" * 60)

    df_all = load_results(BACKTEST_CSV)
    df_all = enrich(df_all)
    df_ny  = df_all[df_all["session"] == "NY"].copy()

    baseline_all = df_all["win"].mean() * 100
    baseline_ny  = df_ny["win"].mean() * 100 if len(df_ny) > 0 else 0

    print(f"\n[BASELINE GLOBAL] WR: {baseline_all:.1f}%  ({len(df_all)} trades)")
    print(f"[BASELINE NY]     WR: {baseline_ny:.1f}%  ({len(df_ny)} trades)")
    print(f"[META]            Alvo: 35-45%")

    # ── ANÁLISE GLOBAL ────────────────────────────────────────────
    print("\n\n" + "█"*60)
    print("  ANÁLISE GLOBAL")
    print("█"*60)

    analyze_dimension(df_all, "session",      "📍 POR SESSÃO",         baseline_all)
    analyze_dimension(df_all, "tier",         "🎯 POR TIER",           baseline_all)
    analyze_dimension(df_all, "direction",    "📈 POR DIREÇÃO",        baseline_all)
    analyze_dimension(df_all, "weekday_name", "📅 POR DIA DA SEMANA",  baseline_all)
    analyze_dimension(df_all, "wick_band",    "📏 POR FAIXA DE PAVIO", baseline_all)
    analyze_dimension(df_all, "dom_band",     "💪 POR DOMINÂNCIA",     baseline_all)

    cross_analysis(df_all, "session",   "tier",      "🔀 SESSÃO × TIER")
    cross_analysis(df_all, "session",   "direction", "🔀 SESSÃO × DIREÇÃO")
    cross_analysis(df_all, "wick_band", "session",   "🔀 PAVIO × SESSÃO")

    # ── ANÁLISE NY ────────────────────────────────────────────────
    if len(df_ny) > 0:
        print("\n\n" + "█"*60)
        print("  ANÁLISE NY — FOCO DO ARQUITETO")
        print("  (11:00-16:00 Brasília | 14:00-19:00 UTC)")
        print("█"*60)

        analyze_dimension(df_ny, "tier",         "🎯 TIER (NY)",             baseline_ny)
        analyze_dimension(df_ny, "direction",    "📈 DIREÇÃO (NY)",          baseline_ny)
        analyze_dimension(df_ny, "weekday_name", "📅 DIA DA SEMANA (NY)",    baseline_ny)
        analyze_dimension(df_ny, "wick_band",    "📏 PAVIO (NY)",            baseline_ny)
        analyze_dimension(df_ny, "dom_band",     "💪 DOMINÂNCIA (NY)",       baseline_ny)
        analyze_dimension(df_ny, "ny_phase",     "⏱️  NY_OPEN vs NY_MID",    baseline_ny)

        cross_analysis(df_ny, "tier",     "direction", "🔀 TIER × DIREÇÃO (NY)")
        cross_analysis(df_ny, "wick_band","direction", "🔀 PAVIO × DIREÇÃO (NY)")
        cross_analysis(df_ny, "ny_phase", "direction", "🔀 FASE × DIREÇÃO (NY)")
        cross_analysis(df_ny, "ny_phase", "tier",      "🔀 FASE × TIER (NY)")
    else:
        print("\n⚠️  Nenhum trade encontrado na sessão NY.")
        print("   Verifique se o timestamp do CSV está em UTC.")

    # ── FILTROS DE EDGE ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  🏆 COMBINAÇÕES COM EDGE REAL (WR ≥ 35%)")
    print(f"{'='*60}")

    edge_all = find_edge_filters(df_all)
    edge_ny  = find_edge_filters(df_ny) if len(df_ny) > 0 else []

    print(f"\n  NY apenas:")
    if edge_ny:
        for f in edge_ny:
            print(f"  ✅ {f['session']} | {f['tier']} | "
                  f"{f['direction']}: {f['wr_pct']}% ({f['total']} trades)")
    else:
        print("  ⚠️  Nenhuma combinação atingiu 35% em NY com amostra suficiente.")
        print("     O edge virá dos filtros do Oracle/Kunkka/Spectre.")

    print(f"\n  Todas as sessões:")
    if edge_all:
        for f in edge_all:
            print(f"  ✅ {f['session']} | {f['tier']} | "
                  f"{f['direction']}: {f['wr_pct']}% ({f['total']} trades)")
    else:
        print("  ⚠️  Nenhuma combinação atingiu 35% globalmente.")
        print("     O pavio sozinho não tem edge — Oracle é o diferencial.")

    # ── EXPORT ────────────────────────────────────────────────────
    export_markdown(df_all, df_ny, edge_all, edge_ny, OUTPUT_REPORT)

    print(f"\n{'='*60}")
    print("  ANÁLISE CONCLUÍDA.")
    print(f"{'='*60}\n")
