# 📉 Mimir War Council: Sovereign Multi-Agent Intelligence

This repository showcases the core engine of the **Project Invoker**, a sovereign multi-agent system designed for high-frequency market analysis and automated decision-making in the XAUUSD (Gold) pair.

## 🏛️ Architecture Overview

The system operates on a **5-second decision loop**, utilizing an adversarial agentic architecture to ensure maximum precision and risk mitigation.

### The Decision Loop
1. **Camada 1 (Parallel Analysis):** Four specialized agents analyze data in parallel (< 2.5s).
   - **Forge Spirit:** Raw price action and liquidity sweeps.
   - **Rubick:** Geometric mimesis and fractal pattern recognition.
   - **Kunkka:** Macro correlation (DXY/US10Y).
   - **Spectre:** News events and session volatility.
2. **Camada 2 (Adversarial Validation):** The **Oracle** attempts to destroy the trade thesis using strict SMC (Smart Money Concepts) math (< 1.5s).
3. **Camada 3 (Final Execution):** The **Invoker** synthesizes the debate and executes the decision (< 1.0s).

## 🚀 Key Features

- **Asynchronous Orchestration:** High-performance Python backend managing concurrent agent calls.
- **RAG for Trading Lore:** Uses vector search to compare current market geometry with thousands of historical backtests.
- **Low Latency Communication:** Agents communicate via structured JSON on RAM Disk (`/dev/shm`) to minimize I/O overhead.
- **SMC-Toolkit Integration:** Automated validation of BOS (Break of Structure), CHoCH (Change of Character), and FVG (Fair Value Gaps).

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Framework:** FastAPI
- **AI Models:** Gemini 1.5 Pro / Flash (Rotational API keys)
- **Persistence:** PostgreSQL + Qdrant (Vector DB)
- **Automation:** Celery + Redis

## 📂 Repository Structure

- `mimir_intelligence_server.py`: The main FastAPI orchestrator.
- `smc_logic.py`: Core mathematical logic for Smart Money Concepts.
- `rubick_analyzer.py`: Vision-based pattern recognition engine.
- `forge_smc_gate.py`: Data ingestion and raw price processing.

---
*This project is part of a larger ecosystem of sovereign AI tools. For more information, visit my profile.*
