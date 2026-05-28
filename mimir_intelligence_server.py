import os
import sys
import time
import json
import io
import shutil
import threading
import asyncio
import subprocess
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import uvicorn
import concurrent.futures
import google.generativeai as genai
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image

# Infraestrutura de Grafo e Busca Vetorial
from typing import TypedDict, Annotated, Sequence, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import ccxt

# Importações locais do Realm
from smc_logic import MimirSMCEngine
from mimir_engine import MimirXAUUSDEngine
from forge_smc_gate import SMCValidationGate
from mimir_geometric_mimesis import MimirGeometricMimesis
from mimesis_reflector import MimesisReflector
from mentfx_scraper import scrape_mentfx
from sovereign_logger import SovereignLogger

class GeminiRotator:
    def __init__(self):
        # Carrega todas as chaves do .env
        self.api_keys = []
        for i in range(1, 10):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key: self.api_keys.append(key)
        
        # Fallback para a chave padrão
        if not self.api_keys and os.getenv("GEMINI_API_KEY"):
            self.api_keys.append(os.getenv("GEMINI_API_KEY"))

        self.models = [
            "models/gemini-2.5-flash", 
            "models/gemini-2.5-pro", 
            "models/gemini-3-flash-preview", 
            "models/gemini-2.0-flash"
        ]
        self.current_key_idx = 0
        self.current_model_idx = 0
        self._configure_current()

    def _configure_current(self):
        if not self.api_keys: return
        genai.configure(api_key=self.api_keys[self.current_key_idx])

    def rotate(self):
        if not self.api_keys: return
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        self._configure_current()
        add_log(f"Rotacionando Chave API (Idx: {self.current_key_idx})", "[IA]", "gold")
        
    def get_model(self):
        return genai.GenerativeModel(self.models[self.current_model_idx])

    def rotate_model(self):
        self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
        self.rotate() # Gira a chave junto com o modelo para balancear
        add_log(f"Rotacionando IA: {self.models[self.current_model_idx]}", "[IA]", "gold")
        return self.get_model()

    def call_api(self, prompt, system_instruction="Você é o Invoker.", image_path=None):
        """Interface simplificada para manter compatibilidade com o InvokerBrain."""
        model = self.get_model()
        try:
            if image_path:
                img = Image.open(image_path)
                response = model.generate_content([system_instruction, prompt, img])
            else:
                response = model.generate_content(f"{system_instruction}\n\n{prompt}")
            return response.text
        except Exception as e:
            raise e

# ==============================================================================
# REALM PATHS & CONFIGURATION
# ==============================================================================
BASE_DIR = "/home/mimir/Documentos/MIMIR"
CORE_DIR = os.path.join(BASE_DIR, "00_CORE")
CONFIG_DIR = os.path.join(CORE_DIR, "Configuracoes")
TRADING_DIR = os.path.join(BASE_DIR, "03_TRADING_SMC")
LORE_DIR = os.path.join(TRADING_DIR, "LORE_DOS_GRAFICOS")
LAB_DIR = os.path.join(BASE_DIR, "06_LABORATORIO_ESTUDOS")
SOVEREIGN_DIR = os.path.join(LAB_DIR, "MIMIR_XAUUSD_SOVEREIGN")

STATE_FILE = os.path.join(CONFIG_DIR, "sovereign_state.json")
LOG_DB_FILE = os.path.join(CONFIG_DIR, "sovereign_logs.db")
logger = SovereignLogger(LOG_DB_FILE)

# Sockets Globais
current_pending_signal = None
current_signal_result = None

ADAPTIVE_LAYER_FILE = os.path.join(CORE_DIR, "ADAPTIVE_PROMPT_LAYER.md")
GRIMORIO_FILE = os.path.join(LORE_DIR, "GRIMORIO_SMC.md")
LORE_INDEX_FILE = os.path.join(LORE_DIR, "lore_index_kael.json")

# MT5 Integration (Wine/Linux)
MT5_COMMON_FILES = "/home/mimir/MT5/drive_c/users/mimir/AppData/Roaming/MetaQuotes/Terminal/Common/Files"
MT5_LIVE_DATA = os.path.join(MT5_COMMON_FILES, "mimir_live_data.csv")
TICK_FILE = os.path.join(MT5_COMMON_FILES, "mimir_tick.txt")
SIGNAL_FILE = os.path.join(MT5_COMMON_FILES, "mimir_signal.txt")
RESULT_FILE = os.path.join(MT5_COMMON_FILES, "mimir_signal_result.json")
ACCOUNT_FILE = os.path.join(MT5_COMMON_FILES, "account_status.json")

# UI & Visualization
PUBLIC_SCREENSHOTS = os.path.join(SOVEREIGN_DIR, "web-ui/public/screenshots")
GALERIA_PATH = os.path.join(BASE_DIR, "07_ARQUIVO_HISTORICO/GALERIA_TRADES")

WAR_COUNCIL_DIR = "/dev/shm/war_council"
REPORTS_DIR = os.path.join(WAR_COUNCIL_DIR, "reports")

BRT = timezone(timedelta(hours=-3))
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DB_ID = os.getenv("NOTION_DB_ID", "19ceaa0f61ca81d3bfeece5982e02815")

# CONFIGURAÇÕES DE RISCO SOBERANO
MAX_DAILY_DRAWDOWN = 2.0
DAILY_LOSS_LOCK = False

# ==============================================================================
# GESTÃO DE ESTADO E PERSISTÊNCIA
# ==============================================================================

def load_adaptive_prompts():
    if os.path.exists(ADAPTIVE_LAYER_FILE):
        try:
            with open(ADAPTIVE_LAYER_FILE, "r") as f:
                content = f.read()
                # Extrai apenas a seção de DIRETRIZES DE VETO para manter o prompt enxuto
                if "## ⚠️ DIRETRIZES DE VETO" in content:
                    directives = content.split("## ⚠️ DIRETRIZES DE VETO")[1].split("---")[0].strip()
                    return f"\n[VETOS SPECTRE]:\n{directives}"
        except: pass
    return ""

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {}

saved_data = load_state()
state = {
    "status": "AUTO-HUNTING",
    "bias": "AUTO",
    "bias_locked_until": saved_data.get("bias_locked_until", 0),
    "ammo": min(3, saved_data.get("ammo", 3)),
    "last_ammo_reset": saved_data.get("last_ammo_reset", ""),
    "last_reported_events": "", # Trava de repetição
    "daily_pnl": saved_data.get("daily_pnl", 0.0),
    "signals": saved_data.get("signals", []),
    "logs": saved_data.get("logs", []),
    "current_price": 0.0,
    "current_spread": 0.0,
    "session": "UNKNOWN",
    "dxy_trend": "NEUTRAL",
    "us10y_trend": "NEUTRAL",
    "news": [],
    "balance": saved_data.get("balance", 50000.0),
    "equity": saved_data.get("equity", 50000.0),
    "last_trade_time": saved_data.get("last_trade_time", 0),
    "council_dialogue": [],
    "agent_status": {
        "Forge Spirit": {"status": "IDLE", "task": "Vigilância Matemática"},
        "Rubick": {"status": "IDLE", "task": "Mimesis Geométrica"},
        "Kunkka": {"status": "IDLE", "task": "Monitoramento Macro"},
        "Spectre": {"status": "IDLE", "task": "Radar de Notícias"},
        "Oracle": {"status": "IDLE", "task": "Dialética Técnica"},
        "Invoker": {"status": "IDLE", "task": "Aguardando Veredicto"}
    },
    "sentiment": saved_data.get("sentiment", {"daily": {"long": 0, "short": 0}, "intraday": {"long": 0, "short": 0}}),
    "last_briefing_date": saved_data.get("last_briefing_date", ""),
    "last_closing_date": saved_data.get("last_closing_date", ""),
    "last_planning_date": saved_data.get("last_planning_date", ""),
    "last_news_date": saved_data.get("last_news_date", ""),
    "last_hourly_council_hour": saved_data.get("last_hourly_council_hour", -1),
}

def save_state():
    global state
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Erro ao salvar: {e}")

def parse_news_time_to_datetime(news_time_str: str) -> Optional[datetime]:
    """
    Faz o parsing do horário de notícia do ForexFactory (ex: '8:30am', '1:15pm')
    e retorna o datetime correspondente na timezone de Nova York (America/New_York).
    """
    clean_str = news_time_str.strip().lower().replace(" ", "")
    if not clean_str or clean_str in ["all day", "tentative", "---", "all-day"]:
        return None
    try:
        dt_time = datetime.strptime(clean_str, "%I:%M%p").time()
        ny_tz = ZoneInfo("America/New_York")
        now_ny = datetime.now(ny_tz)
        dt_news = datetime.combine(now_ny.date(), dt_time).replace(tzinfo=ny_tz)
        return dt_news
    except Exception:
        try:
            dt_time = datetime.strptime(clean_str, "%I%p").time()
            ny_tz = ZoneInfo("America/New_York")
            now_ny = datetime.now(ny_tz)
            dt_news = datetime.combine(now_ny.date(), dt_time).replace(tzinfo=ny_tz)
            return dt_news
        except Exception:
            return None

def check_news_restriction() -> tuple[bool, str]:
    """
    Implementa as diretrizes de restrição de notícias da My Funding Pips:
    - Janela restrita de 5 minutos antes e depois de notícias USD de alto impacto.
    - 1 minuto extra de margem de segurança do Mimir (totalizando 6 minutos).
    - Exclusão: Trades abertos a mais de 5 horas do evento não sofrem restrição de fechamento.
    """
    global state
    news_list = state.get("news", [])
    if not news_list:
        return False, ""

    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny_tz)

    for news in news_list:
        if news.get("impact", "").lower() != "high":
            continue

        news_dt = parse_news_time_to_datetime(news.get("time", ""))
        if not news_dt:
            continue

        diff_minutes = (news_dt - now_ny).total_seconds() / 60.0

        # Janela de 6 minutos (5 min + 1 min precaução)
        if -6.0 <= diff_minutes <= 6.0:
            time_str = news.get("time")
            title = news.get("title")
            if diff_minutes > 0:
                reason = f"RESTRIÇÃO DE NOTÍCIAS MY FUNDING PIPS: Janela ativa. Falta(m) {diff_minutes:.1f} minuto(s) para o evento '{title}' ({time_str} NY)."
            else:
                reason = f"RESTRIÇÃO DE NOTÍCIAS MY FUNDING PIPS: Janela ativa. Ocorreu há {-diff_minutes:.1f} minuto(s) o evento '{title}' ({time_str} NY)."
            return True, reason

    return False, ""

def add_log(text, sys="[SISTEMA]", cls="white"):
    global state
    # Novo logger SQLite assíncrono
    logger.log(text, sys, cls)
    
    # Mantém lista local em memória curta
    now = datetime.now().strftime("%H:%M:%S")
    state["logs"].append({"time": now, "sys": sys, "text": text, "cls": cls})
    if len(state["logs"]) > 100: state["logs"].pop(0)
    if sys in ["[ENGINE]", "[NOTION]", "[HUD]", "[RADAR]", "[STRIKE]", "[CONSELHO]"]: save_state()


council_briefing = {
    "macro_status": "Aguardando análise...",
    "historical_mimesis": "Aguardando comparação...",
    "news_alert": "Radar limpo.",
    "last_briefing_time": 0
}

# ==============================================================================
# BUSCA VETORIAL (QDRANT)
# ==============================================================================

# Inicialização do Qdrant (Local)
try:
    qdrant = QdrantClient(host="localhost", port=6333)
    # Garante que a coleção de Lore existe
    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections]
    if "mimir_lore" not in collection_names:
        qdrant.create_collection(
            collection_name="mimir_lore",
            vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE),
        )
    add_log("Qdrant: Visão Vetorial ativada para Rubick.", "[SISTEMA]", "gold")
except Exception as e:
    add_log(f"Qdrant Offline: {e}", "[SISTEMA]", "blood")
    qdrant = None

# ==============================================================================
# INTELIGÊNCIA ARTIFICIAL (CONSELHO)
# ==============================================================================
try:
    ai_rotator = GeminiRotator()
    add_log(f"Cadeira Múltipla Ativa: {len(ai_rotator.api_keys)} chaves Gemini.", "[IA]", "gold")
except Exception as e:
    add_log(f"Erro IA: {e}", "[IA]", "blood")
    ai_rotator = None

mimesis = MimirGeometricMimesis()
reflector = MimesisReflector()

class InvokerBrain:
    def get_ai_response(self, agent_name, prompt, image_path=None, max_lines=2):
        if not ai_rotator: return f"{agent_name}: IA Offline."
        max_attempts = len(ai_rotator.api_keys) * len(ai_rotator.models)
        
        for attempt in range(max_attempts):
            try:
                # --- Sincronização de Status ---
                if agent_name in state["agent_status"]:
                    state["agent_status"][agent_name]["status"] = "PROCESSING"
                    save_state()

                # Carregamento do Grimório SMC para o Oracle
                smc_grimoire = ""
                try:
                    if os.path.exists(GRIMORIO_FILE):
                        with open(GRIMORIO_FILE, "r") as f:
                            smc_grimoire = f.read()
                except: pass

                # Definição dos Poderes do Toolkit para o Oracle
                toolkit_powers = (
                    "TOOLKIT SMC DISPONÍVEL: "
                    "1. calc_swing_structures (BOS/CHoCH Automático). "
                    "2. extract_ob_blocks (Order Blocks Reais). "
                    "3. extract_fvg (Fair Value Gaps com status de Mitigação). "
                    "4. process_smc_internal (Análise de Fractalidade)."
                )

                council_prompts = {
                    "Oracle": f"MESTRE SMC SUPREMO. Base: {smc_grimoire[:2000]}. {toolkit_powers} Analise Liquidez, Indução e Mitigação. Use os dados do Toolkit para validar cada gatilho.",
                    "Kunkka": "ESTRATEGISTA MACRO. DXY, US10Y e Correlações. Sua missão é dar o contexto de maré para o trade.",
                    "Rubick": "ANALISTA GEOMÉTRICO. Compara fractais da Lore com o gráfico atual. Busca mimese e simetria.",
                    "Invoker": "EXECUTOR SOBERANO. Recebe a convergência e decide o gatilho final. Se a Trindade divergir, aborte."
                }

                specialized_instructions = {
                    "Forge Spirit": "Batedor Técnico. Fale APENAS de Price Action bruto.",
                    "Rubick": "Analista Geométrico IMPLACÁVEL. PROIBIDO poesia. Cite ID (ex: TRADE_019) e GATILHO.",
                    "Kunkka": "Almirante Macro. Fale de DXY, US10Y Yields e Correlação. Seja seco e numérico.",
                    "Oracle": council_prompts.get("Oracle", "MESTRE SMC. Analise Liquidez, Indução e Mitigação."),
                    "Spectre": "Sentinela de Regras e Escriba. Veto em Notícias e Auditoria.",
                    "Invoker": "Executor Final. Bate o martelo."
                }

                instruction = specialized_instructions.get(agent_name, "Agente.")
                
                mimesis_context = ""
                if agent_name == "Rubick" and image_path:
                    similar_patterns = mimesis.find_similar_patterns(image_path)
                    if isinstance(similar_patterns, list):
                        mimesis_context = "\n[SIMILARIDADE]:\n" + "\n".join([f"- {os.path.basename(p['path'])} ({p['score']:.4f})" for p in similar_patterns])

                rules_str = f"[REGRAS]: RESPOSTA MÁXIMA {max_lines} LINHAS.\n\n" if max_lines else ""
                
                system_instruction = f"Você é {agent_name}. {instruction}"
                main_prompt = (
                    f"[DADOS]: Preço={state['current_price']}, Bias={state['bias']}.\n"
                    f"[BRIEFING]: Macro={council_briefing['macro_status']}, Lore={council_briefing['historical_mimesis']}.\n"
                    f"{load_adaptive_prompts() if agent_name in ['Oracle', 'Invoker'] else ''}"
                    f"{mimesis_context}\n"
                    f"{rules_str}"
                    f"Contexto: {prompt}"
                )

                response_text = ai_rotator.call_api(main_prompt, system_instruction=system_instruction, image_path=image_path)
                
                if max_lines:
                    text = " ".join([l.strip() for l in response_text.split("\n") if l.strip()][:max_lines])
                else:
                    text = response_text.strip()
                
                self.save_json_report(agent_name, text)
                return text

            except Exception as e:
                add_log(f"Erro na IA ({agent_name}): Rotacionando Modelo/Chave...", "[IA]", "blood")
                ai_rotator.rotate_model()
        return f"{agent_name}: Névoa densa."

    def save_json_report(self, agent_name, text):
        try:
            report = {"agent": agent_name, "timestamp": time.time(), "raw": text}
            file_path = os.path.join(REPORTS_DIR, f"{agent_name.lower().replace(' ', '_')}.json")
            os.makedirs(REPORTS_DIR, exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(report, f, indent=4)
        except: pass

invoker_brain = InvokerBrain()

# ==============================================================================
# CONSELHO DE GUERRA SOBERANO
# ==============================================================================

def add_council_msg(agent, text, color="gold"):
    msg = {"agent": agent, "text": text, "color": color, "time": datetime.now(BRT).strftime("%H:%M")}
    state["council_dialogue"].append(msg)
    print(f"[DEBUG] Mensagem adicionada: {agent} - {text[:30]}...")
    if len(state["council_dialogue"]) > 30: state["council_dialogue"].pop(0)

def cleanup_war_council():
    try:
        # Reset de Status Visual para o Dashboard
        for agent in state["agent_status"]:
            state["agent_status"][agent]["status"] = "IDLE"
        save_state()

        if os.path.exists(REPORTS_DIR):
            for f in os.listdir(REPORTS_DIR):
                os.remove(os.path.join(REPORTS_DIR, f))
    except: pass

# ==============================================================================
# ESCRIBA SPECTRE: AUTOMAÇÃO DE LORE E CAPTURA
# ==============================================================================
LORE_BASE_PATH = "/home/mimir/Documentos/MIMIR/03_TRADING_SMC/LORE_DOS_GRAFICOS"
from mimir_vision import MimirVision
vision = MimirVision()

active_trade_context = {
    "id": None,
    "path": None,
    "start_time": 0,
    "captured_5min": False,
    "data": {}
}

def setup_lore_folder(bias, validation_data):
    """Cria a pasta datada e serializada para o novo trade."""
    try:
        folders = [f for f in os.listdir(LORE_BASE_PATH) if f.startswith("TRADE_")]
        last_id = 0
        if folders:
            last_id = max([int(f.split("_")[1]) for f in folders])
        
        new_id = last_id + 1
        folder_name = f"TRADE_{new_id:03d}_{datetime.now().strftime('%d_%m')}"
        path = os.path.join(LORE_BASE_PATH, folder_name)
        os.makedirs(path, exist_ok=True)
        
        active_trade_context["id"] = new_id
        active_trade_context["path"] = path
        active_trade_context["start_time"] = time.time()
        active_trade_context["captured_5min"] = False
        active_trade_context["data"] = {
            'bias': bias,
            'price': state['current_price'],
            'poi': validation_data.get('poi', 0.0),
            'liq': validation_data.get('liq', 0.0),
            'target': validation_data.get('target', 0.0)
        }
        
        # 1. Captura Multi-Timeframe de Entrada (Daily, H1, M15)
        try:
            vision.capture_multi_timeframe(path)
        except Exception as ve:
            add_log(f"Aviso: Falha na visão (screenshots), mas prosseguindo com o trade. {ve}", "[VISION]", "blood")
            # Gera análise detalhada apenas no M15 (execução)
            img_m15 = os.path.join(path, "03_M15.png")
            img_hud = os.path.join(path, "01_ENTRADA_ANALISE.png")
            if os.path.exists(img_m15):
                vision.create_analysis(img_m15, img_hud, active_trade_context["data"])
            
        add_log(f"Lore Multi-TF Criada: {folder_name}", "[SPECTRE]", "gold")
        return path
    except Exception as e:
        add_log(f"Erro ao criar Lore: {e}", "[SPECTRE]", "blood")
        return None

def trigger_war_council(session_type="BATTLE", validation_data=None, user_message=None, trade_allowed=True):
    """
    Conselho de Guerra Monolítico — 1 Prompt, 1 Chamada de API.
    Todos os agentes operam no mesmo contexto, sem round-trips individuais.
    trade_allowed=False: Conselho fala mas não executa ordens (modo análise/fora de janela).
    """
    start_time = time.time()

    # Registra a mensagem do Arquiteto no chat, se houver
    if user_message:
        add_council_msg("Arquiteto", user_message, "white")

    is_auto = state.get("bias") == "AUTO"
    current_bias_context = (
        "MODO AUTO: O Conselho deve determinar o viés provável e agir."
        if is_auto else f"Bias Definido: {state['bias']}."
    )

    if validation_data is None:
        default_tier = "MANUAL" if session_type == "MANUAL" else (
            "PREDICTIVE_SESSION" if is_auto else "MANUAL_INVOCATION"
        )
        validation_data = {"tier": default_tier, "mimese_tecnica": "Análise de tendência e estrutura."}

    tier = validation_data.get("tier", "UNKNOWN")
    mimese_tecnica = validation_data.get("mimese_tecnica", "Dados Brutos.")

    # News formatadas para contexto
    news_items = state.get("news", [])
    if news_items:
        news_context = " | ".join([f"{n.get('time','?')} {n.get('title', n.get('name','?'))}" for n in news_items])
    else:
        news_context = "Sem eventos de alto impacto USD hoje."

    # Grimório SMC (contexto técnico)
    smc_grimoire = ""
    try:
        if os.path.exists(GRIMORIO_FILE):
            with open(GRIMORIO_FILE, "r") as f:
                smc_grimoire = f.read()[:1500]
    except: pass

    # Adaptive prompts (vetos do Spectre)
    adaptive = load_adaptive_prompts()

    add_log(f"Monolithic Council: {tier}", "[CONSELHO]", "cyan")

    # Blindagem de Notícias (My Funding Pips)
    is_restricted, news_reason = check_news_restriction()
    news_warning = ""
    if is_restricted:
        news_warning = f"\n[VETO CRÍTICO DE NOTÍCIAS (SPECTRE)]: RESTRIÇÃO ATIVA. {news_reason}\nO VETO É OBRIGATÓRIO. NÃO ABRIR OU FECHAR POSIÇÕES (MESA PROPRIETÁRIA MY FUNDING PIPS).\n"

    # --- PROMPT MONOLÍTICO SOBERANO ---
    user_context = f"[MENSAGEM DO ARQUITETO]: {user_message}\n\n" if user_message else ""
    daily_context = ""
    if state.get("daily_narrative"):
        dn = state["daily_narrative"]
        daily_context = f"NARRATIVA DAILY D1: Cenário={dn.get('scenario')} | Bias={dn.get('bias')} | Raciocínio={dn.get('reason')}\n"
    
    if session_type == "BATTLE":
        monolith_prompt = (
            f"CONSELHO DE GUERRA — SESSÃO BATTLE (Janela Sniper). {current_bias_context}\n"
            f"{daily_context}"
            f"{user_context}"
            f"SESSÃO: {state['session']} | XAUUSD | Preço: {state['current_price']:.2f} | Spread: {state['current_spread']:.2f}\n"
            f"ANÁLISE SMC (Scout/Forge Spirit): {mimese_tecnica}\n"
            f"GRIMÓRIO SMC: {smc_grimoire}\n"
            f"{news_warning}"
            f"{adaptive}\n\n"
            f"VOCÊ DEVE RESPONDER EXATAMENTE NO FORMATO ABAIXO (sem desvios, apenas estes dois agentes):\n"
            f"[ORACLE]: <validação SMC — Sweep/Inducement/POI em 1-2 linhas>\n"
            f"[INVOKER]: APROVADO ou VETADO | ACTION=BUY ou ACTION=SELL | SL=<preço> | TP=<preço> | LOT=<lote>\n\n"
            f"MUNIÇÃO RESTANTE: {state['ammo']}/3. SEJA IMPLACÁVEL."
        )
    elif session_type == "HOURLY_MAINTENANCE":
        monolith_prompt = (
            f"CONSELHO DE GUERRA — AUDITORIA HORÁRIA (Vigília Soberana). {current_bias_context}\n"
            f"{daily_context}"
            f"SESSÃO: {state['session']} | XAUUSD | Preço: {state['current_price']:.2f} | Spread: {state['current_spread']:.2f}\n"
            f"DXY: {state.get('dxy_trend','?')} | US10Y: {state.get('us10y_trend','?')}\n"
            f"ANÁLISE SMC (Scout/Forge Spirit): {mimese_tecnica}\n"
            f"GRIMÓRIO SMC: {smc_grimoire}\n"
            f"{news_warning}"
            f"{adaptive}\n\n"
            f"TAREFA: Realizar auditoria técnica para identificar setups sutis que o ENGINE possa ter ignorado.\n"
            f"VOCÊ DEVE RESPONDER EXATAMENTE NO FORMATO ABAIXO (apenas estes dois agentes):\n"
            f"[ORACLE]: <validação SMC em 1-2 linhas>\n"
            f"[INVOKER]: APROVADO ou VETADO | ACTION=BUY ou ACTION=SELL | SL=<preço> | TP=<preço> | LOT=<lote>\n\n"
            f"MUNIÇÃO RESTANTE: {state['ammo']}/3. SEJA O OLHO QUE TUDO VÊ."
        )
    else:
        monolith_prompt = (
            f"CONSELHO DE GUERRA — DECRETO ÚNICO. {current_bias_context}\n"
            f"{daily_context}"
            f"{user_context}"
            f"SESSÃO: {state['session']} | XAUUSD | Preço: {state['current_price']:.2f} | Spread: {state['current_spread']:.2f}\n"
            f"DXY: {state.get('dxy_trend','?')} | US10Y: {state.get('us10y_trend','?')}\n"
            f"ANÁLISE SMC (Scout/Forge Spirit): {mimese_tecnica}\n"
            f"NOTÍCIAS ALTO IMPACTO: {news_context}\n"
            f"GRIMÓRIO SMC: {smc_grimoire}\n"
            f"{news_warning}"
            f"{adaptive}\n\n"
            f"VOCÊ DEVE RESPONDER EXATAMENTE NO FORMATO ABAIXO (sem desvios):\n"
            f"[ORACLE]: <validação SMC — Sweep/Inducement/POI em 1-2 linhas>\n"
            f"[KUNKKA]: <macro DXY/US10Y apoia ou enfraquece? 1 linha>\n"
            f"[RUBICK]: <similaridade com Lore ou 'Sem precedente claro'>\n"
            f"[SPECTRE]: <veto de notícias ou 'Radar limpo'>\n"
            f"[INVOKER]: APROVADO ou VETADO | ACTION=BUY ou ACTION=SELL | SL=<preço> | TP=<preço> | LOT=<lote>\n"
            f"[MIMIR]: <sabedoria estratégica em 1 linha>\n\n"
            f"MUNIÇÃO RESTANTE: {state['ammo']}/3. SEJA IMPLACÁVEL E PRECISO."
        )

    # Feedback visual IMEDIATO — sem chamadas de API ainda
    add_council_msg("Conselho", "⚔️ Convocando o Conselho Monolítico...", "gold")
    save_state()

    # --- ÚNICA CHAMADA DE API ---
    decree_resp = invoker_brain.get_ai_response("Invoker", monolith_prompt, max_lines=None)

    # Parsing e exibição das falas individuais no chat
    agent_tags = ["ORACLE", "KUNKKA", "RUBICK", "SPECTRE", "INVOKER", "MIMIR"]
    agent_colors = {
        "ORACLE": "purple", "KUNKKA": "cyan", "RUBICK": "green",
        "SPECTRE": "white", "INVOKER": "gold", "MIMIR": "gold"
    }
    agent_display = {
        "ORACLE": "Oracle", "KUNKKA": "Kunkka", "RUBICK": "Rubick",
        "SPECTRE": "Spectre", "INVOKER": "Invoker", "MIMIR": "Mimir"
    }

    lines = decree_resp.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for tag in agent_tags:
            if line.upper().startswith(f"[{tag}]"):
                text = line[len(f"[{tag}]"):].strip().lstrip(":").strip()
                add_council_msg(
                    agent_display[tag],
                    text,
                    agent_colors.get(tag, "gold")
                )
                break

    # Parsing da decisão
    is_approved = "APROVADO" in decree_resp.upper() and (
        "ACTION=BUY" in decree_resp.upper() or "ACTION=SELL" in decree_resp.upper()
    )

    # Atualiza Bias se em modo AUTO
    if is_auto:
        invoker_line = next((l for l in lines if "[INVOKER]" in l.upper()), "")
        if "BUY" in invoker_line.upper():
            state["bias"] = "BULL"
            add_log("Conselho assumiu o comando: BULL", "[CONSELHO]", "gold")
        elif "SELL" in invoker_line.upper():
            state["bias"] = "BEAR"
            add_log("Conselho assumiu o comando: BEAR", "[CONSELHO]", "gold")
        save_state()

    if "Névoa densa" in decree_resp:
        add_log("Execução abortada por Névoa Densa na IA.", "[CONSELHO]", "blood")
        return "ABORTADO"

    # Execução de trade se aprovado — APENAS se dentro da janela Sniper
    if is_approved and trade_allowed:
        setup_lore_folder(state['bias'], validation_data)

        price = state['current_price']
        fixed_sl_pts = 700
        sl_dist = fixed_sl_pts / 100

        sl = price + sl_dist if state['bias'] == "BEAR" else price - sl_dist
        tp_dist = (fixed_sl_pts * 5) / 100
        tp = price - tp_dist if state['bias'] == "BEAR" else price + tp_dist

        lot = calculate_lot(fixed_sl_pts)
        signal_id = int(time.time() * 1000)

        detected_bias = "BULL" if "BUY" in decree_resp.upper() else "BEAR" if "SELL" in decree_resp.upper() else state['bias']
        final_action = detected_bias if is_auto else state['bias']

        new_signal = {
            "id": signal_id, "action": final_action,
            "price": price, "sl": round(sl, 2), "tp": round(tp, 2),
            "lot": lot, "timestamp": time.time()
        }

        # Limpar arquivo de resultado anterior (fallback)
        if os.path.exists(RESULT_FILE):
            try:
                os.remove(RESULT_FILE)
            except:
                pass

        try:
            if state.get('ammo', 0) <= 0:
                add_log("DISPARO ABORTADO: Munição Insuficiente (0/3).", "[INVOKER]", "blood")
                return decree_resp

            global current_pending_signal, current_signal_result
            current_signal_result = None
            current_pending_signal = new_signal

            # Injeção via arquivo (fallback/contingência)
            with open(SIGNAL_FILE, "w") as f:
                secure_action = final_action if final_action in ["BULL", "BEAR", "BUY", "SELL"] else state.get("bias", "BULL")
                if secure_action == "AUTO": secure_action = "BULL"
                f.write(f"ACTION={secure_action}\n")
                f.write(f"PRICE={price}\n")
                f.write(f"SL={sl}\n")
                f.write(f"TP={tp}\n")
                f.write(f"LOT={lot}\n")
                f.write(f"ID={signal_id}\n")
                f.flush()
                os.fsync(f.fileno())

            add_log(f"SINAL INJETADO (ID {signal_id}): {final_action} | Lote: {lot} | SL: {sl:.2f} | TP: {tp:.2f}. Aguardando handshake (Socket/Arquivo)...", "[INVOKER]", "cyan")

            # Adiciona provisoriamente o sinal à lista local
            state["signals"].append(new_signal)
            if len(state["signals"]) > 10: state["signals"].pop(0)
            save_state()

            # Espera ativa de duas vias (Socket + Arquivo Fallback)
            trade_confirmed = False
            time_start = time.time()
            timeout = 5.0

            while time.time() - time_start < timeout:
                # 1. Checa via Socket (Prioritário)
                if current_signal_result is not None:
                    status = current_signal_result.get("status")
                    msg = current_signal_result.get("message", "")
                    if status == "SUCCESS":
                        trade_confirmed = True
                        add_log(f"HANDSHAKE CONFIRMADO VIA SOCKET: Ordem executada com sucesso.", "[INVOKER]", "success")
                    else:
                        trade_confirmed = False
                        add_log(f"ORDEM CANCELADA/FALHADA NO MT5 VIA SOCKET ({status}): {msg}", "[INVOKER]", "blood")
                    break

                # 2. Checa via Arquivo (Fallback)
                if os.path.exists(RESULT_FILE):
                    try:
                        with open(RESULT_FILE, "r") as rf:
                            res_data = json.load(rf)
                            if res_data.get("id") == signal_id:
                                status = res_data.get("status")
                                msg = res_data.get("message", "")
                                if status == "SUCCESS":
                                    trade_confirmed = True
                                    add_log(f"HANDSHAKE CONFIRMADO VIA ARQUIVO: Ordem executada com sucesso.", "[INVOKER]", "success")
                                else:
                                    trade_confirmed = False
                                    add_log(f"ORDEM CANCELADA/FALHADA NO MT5 VIA ARQUIVO ({status}): {msg}", "[INVOKER]", "blood")
                                current_pending_signal = None  # Limpa o socket também
                                break
                    except Exception:
                        pass
                
                time.sleep(0.1)

            if time.time() - time_start >= timeout and not trade_confirmed:
                # Limpa sinal pendente do Socket em caso de timeout
                current_pending_signal = None
                trade_confirmed = True
                add_log("TIMEOUT DO HANDSHAKE: Nenhuma resposta (Socket/Arquivo). Consumindo munição por segurança.", "[INVOKER]", "cyan")

            if trade_confirmed:
                state['ammo'] = max(0, state['ammo'] - 1)
                save_state()
                add_log(f"MUNIÇÃO CONSUMIDA: Restante {state['ammo']}/3", "[INVOKER]", "success")
            else:
                # Remove o sinal fantasma para manter a integridade dos logs
                state["signals"] = [s for s in state["signals"] if s["id"] != signal_id]
                save_state()
                add_log("MUNIÇÃO PRESERVADA: O erro de execução evitou o gasto de munição.", "[INVOKER]", "gold")

        except Exception as e:
            add_log(f"Erro no processamento do sinal: {e}", "[INVOKER]", "blood")

    elapsed = time.time() - start_time
    if not is_approved and not trade_allowed:
        add_log(f"Conselho ouvido em modo ANÁLISE em {elapsed:.2f}s — sem ordens fora da Janela Sniper.", "[CONSELHO]", "cyan")
    elif not is_approved:
        add_log(f"Conselho reunido em {elapsed:.2f}s. Nenhum sinal aprovado neste ciclo.", "[CONSELHO]", "white")
    else:
        add_log(f"Monolithic Council finalizado em {elapsed:.2f}s", "[CONSELHO]", "gold")

    return decree_resp

def trigger_scheduled_meeting(meeting_type):
    """
    Executa as reuniões agendadas do conselho:
    - MORNING_BRIEFING (08:30): Abertura do dia.
    - EVENING_CLOSING (17:00): Fechamento do dia.
    - NIGHT_PLANNING (22:00): Planejamento de amanhã.
    """
    start_time = time.time()
    add_log(f"Iniciando reunião agendada: {meeting_type}...", "[CONSELHO]", "gold")
    
    # News formatadas para contexto
    news_items = state.get("news", [])
    if news_items:
        news_context = " | ".join([f"{n.get('time','?')} {n.get('title', n.get('name','?'))}" for n in news_items])
    else:
        news_context = "Sem eventos USD agendados."

    # Grimório SMC
    smc_grimoire = ""
    try:
        if os.path.exists(GRIMORIO_FILE):
            with open(GRIMORIO_FILE, "r") as f:
                smc_grimoire = f.read()[:1500]
    except: pass

    # Monta o prompt temático
    if meeting_type == "MORNING_BRIEFING":
        prompt = (
            f"CONSELHO DE GUERRA — REUNIÃO DE ABERTURA DIÁRIA (08:30 BRT)\n"
            f"XAUUSD | Preço Atual: {state['current_price']:.2f} | DXY: {state.get('dxy_trend','?')} | US10Y: {state.get('us10y_trend','?')}\n"
            f"NOTÍCIAS DO DIA: {news_context}\n"
            f"GRIMÓRIO SMC: {smc_grimoire}\n\n"
            f"Cada membro do conselho deve passar suas informações para o dia:\n"
            f"- Kunkka deve detalhar as notícias do dia e o impacto macro de DXY/US10Y.\n"
            f"- Rubick deve verificar a geometria atual com os fractais da Lore de Backtests e apontar semelhanças.\n"
            f"- Spectre deve relatar as sessões operacionais do dia e restrições horárias de notícias.\n"
            f"- Oracle deve relatar a análise estrutural da manhã e POIs principais.\n"
            f"- Invoker deve relatar a preparação técnica e o Bias esperado.\n"
            f"- Mimir deve dar seu conselho estratégico de sabedoria para iniciar o dia.\n\n"
            f"VOCÊ DEVE RESPONDER EXATAMENTE NO FORMATO ABAIXO:\n"
            f"[KUNKKA]: <detalhe de notícias e impacto macro>\n"
            f"[RUBICK]: <similaridade de fractais da Lore ou 'Sem precedentes claros'>\n"
            f"[SPECTRE]: <radar de notícias/sessões e restrições>\n"
            f"[ORACLE]: <análise estrutural e POIs principais>\n"
            f"[INVOKER]: <estado de preparação técnica e Bias esperado>\n"
            f"[MIMIR]: <sabedoria estratégica para o dia>"
        )
    elif meeting_type == "EVENING_CLOSING":
        # Formata os sinais/trades gerados hoje para análise
        signals_today = state.get("signals", [])
        signals_context = json.dumps(signals_today) if signals_today else "Nenhum sinal gerado hoje."
        prompt = (
            f"CONSELHO DE GUERRA — REUNIÃO DE FECHAMENTO OPERACIONAL (17:00 BRT)\n"
            f"O mercado Forex encerrou suas operações principais para nós hoje.\n"
            f"Saldo Final: {state.get('balance', 50000.0):.2f} | PnL Diário: {state.get('daily_pnl', 0.0):.2f}\n"
            f"Sinais gerados hoje: {signals_context}\n\n"
            f"Cada membro deve analisar como foi o dia e o resultado obtido. O que deu certo e o que deu errado?\n"
            f"- Oracle: Avalie a validação SMC e se as regras de entrada foram respeitadas.\n"
            f"- Kunkka: Analise o comportamento do DXY/US10Y frente ao XAUUSD.\n"
            f"- Rubick: Analise se a mimese fractal se comportou de acordo com a lore de backtests.\n"
            f"- Spectre: Avalie o impacto de notícias e sessões horárias no dia.\n"
            f"- Invoker: Analise a performance de execução e o saldo final.\n"
            f"- Mimir: Dê a sabedoria e lição de encerramento do dia.\n\n"
            f"VOCÊ DEVE RESPONDER EXATAMENTE NO FORMATO:\n"
            f"[ORACLE]: <análise técnica do dia>\n"
            f"[KUNKKA]: <análise macro do dia>\n"
            f"[RUBICK]: <análise geométrica/fractal do dia>\n"
            f"[SPECTRE]: <auditoria de notícias/sessões do dia>\n"
            f"[INVOKER]: <resumo das execuções e pnl líquido>\n"
            f"[MIMIR]: <sabedoria de fechamento do dia>"
        )
    else: # NIGHT_PLANNING (22:00)
        prompt = (
            f"CONSELHO DE GUERRA — REUNIÃO DE PLANEJAMENTO PARA AMANHÃ (22:00 BRT)\n"
            f"Preço Atual: {state['current_price']:.2f} | DXY: {state.get('dxy_trend','?')} | US10Y: {state.get('us10y_trend','?')}\n"
            f"NOTÍCIAS FUTURAS: {news_context}\n\n"
            f"Cada membro deve detalhar as projeções e expectativas para amanhã:\n"
            f"- Oracle: Zonas de liquidez e OBs que deveremos observar amanhã.\n"
            f"- Kunkka: Eventos macro e tendências DXY/US10Y projetadas.\n"
            f"- Rubick: Possível fractal que poderá se desenhar de acordo com a lore.\n"
            f"- Spectre: Alertas de notícias de alto impacto programadas.\n"
            f"- Invoker: Viés de operação proposto (BULL/BEAR) e diretrizes de munição.\n"
            f"- Mimir: Sabedoria soberana e diretriz estratégica para a noite.\n\n"
            f"VOCÊ DEVE RESPONDER EXATAMENTE NO FORMATO:\n"
            f"[ORACLE]: <zonas de interesse projetadas para amanhã>\n"
            f"[KUNKKA]: <calendário macro e tendências de amanhã>\n"
            f"[RUBICK]: <expectativa fractal ou geométrica>\n"
            f"[SPECTRE]: <alertas de eventos econômicos para amanhã>\n"
            f"[INVOKER]: <viés de operação proposto e diretrizes de munição>\n"
            f"[MIMIR]: <conselho de sabedoria soberana para a noite>"
        )

    # Feedback visual de convocação
    add_council_msg("Conselho", f"⚔️ Convocando Reunião {meeting_type}...", "gold")
    save_state()

    try:
        # Chamada de API única
        decree_resp = invoker_brain.get_ai_response("Invoker", prompt, max_lines=None)
        
        # Parsing e exibição no chat
        agent_tags = ["KUNKKA", "RUBICK", "SPECTRE", "ORACLE", "INVOKER", "MIMIR"]
        agent_colors = {
            "ORACLE": "purple", "KUNKKA": "cyan", "RUBICK": "green",
            "SPECTRE": "white", "INVOKER": "gold", "MIMIR": "gold"
        }
        agent_display = {
            "ORACLE": "Oracle", "KUNKKA": "Kunkka", "RUBICK": "Rubick",
            "SPECTRE": "Spectre", "INVOKER": "Invoker", "MIMIR": "Mimir"
        }

        lines = decree_resp.split("\n")
        for line in lines:
            line = line.strip()
            if not line: continue
            for tag in agent_tags:
                if line.upper().startswith(f"[{tag}]"):
                    text = line[len(f"[{tag}]"):].strip().lstrip(":").strip()
                    add_council_msg(
                        agent_display[tag],
                        text,
                        agent_colors.get(tag, "gold")
                    )
                    # --- AUTONOMIA DE BIAS SOBERANO ---
                    # Captura o Bias definido pelo Invoker no Briefing e aplica ao sistema
                    if tag == "INVOKER" and meeting_type == "MORNING_BRIEFING":
                        if "BULL" in text.upper() or "BUY" in text.upper() or "ALTA" in text.upper():
                            state["bias"] = "BULL"
                            state["bias_locked_until"] = time.time() + 28800 # Trava por 8 horas
                            add_log("AUTONOMIA: Conselho definiu o Bias do dia como BULLISH.", "[SISTEMA]", "gold")
                        elif "BEAR" in text.upper() or "SELL" in text.upper() or "BAIXA" in text.upper():
                            state["bias"] = "BEAR"
                            state["bias_locked_until"] = time.time() + 28800 # Trava por 8 horas
                            add_log("AUTONOMIA: Conselho definiu o Bias do dia como BEARISH.", "[SISTEMA]", "gold")
                        save_state()
                    break
        
        # Se for briefing da manhã, reseta o pnl diário e a munição
        if meeting_type == "MORNING_BRIEFING":
            state["daily_pnl"] = 0.0
            state["ammo"] = 3
            save_state()

        elapsed = time.time() - start_time
        add_log(f"Reunião {meeting_type} finalizada com sucesso em {elapsed:.2f}s", "[CONSELHO]", "gold")
    except Exception as e:
        add_log(f"Erro na reunião {meeting_type}: {e}", "[CONSELHO]", "blood")

# ==============================================================================
# MOTOR DE DADOS E SINCRONIA
# ==============================================================================
def load_mt5_data():
    if not os.path.exists(MT5_LIVE_DATA): return None, None, None, None
    for _ in range(3):
        try:
            with open(MT5_LIVE_DATA, "rb") as f:
                raw = f.read()
            
            content = None
            for encoding in ["utf-8", "latin-1", "utf-16"]:
                try:
                    content = raw.decode(encoding)
                    if "Time,Open" in content: break
                except: continue
            
            if not content: time.sleep(0.1); continue
            df = pd.read_csv(io.StringIO(content))
            df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Spread', 'Timeframe']
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            df.sort_index(inplace=True) # Garante ordem cronológica
            
            # Atualiza Spread Global
            state["current_spread"] = round(df['Spread'].iloc[-1], 4)
            
            return df[df['Timeframe']=='M5'], df[df['Timeframe']=='M15'], df[df['Timeframe']=='H1'], df[df['Timeframe']=='D1']
        except Exception as e: 
            print(f"Erro Load MT5: {e}")
            time.sleep(0.1)
    return None, None, None, None

def fetch_macro_data():
    """Atualiza DXY e US10Y a cada 60s. Silencioso — sem logs no chat."""
    try:
        dxy = yf.Ticker("DX-Y.NYB").history(period="2d", interval="1h")
        if not dxy.empty:
            state["dxy_trend"] = "BULL" if dxy['Close'].iloc[-1] > dxy['Close'].iloc[-2] else "BEAR"
        us10y = yf.Ticker("^TNX").history(period="2d", interval="1h")
        if not us10y.empty:
            state["us10y_trend"] = "BULL" if us10y['Close'].iloc[-1] > us10y['Close'].iloc[-2] else "BEAR"
    except Exception as e:
        pass  # Falhas de macro não precisam de log no chat


def fetch_news_daily():
    """
    Radar de Notícias Soberano — roda 1x/dia.
    Captura APENAS notícias de ALTO IMPACTO (vermelhas) do ForexFactory.
    Filtra apenas os eventos do DIA ATUAL.
    SEM spam no chat do Conselho.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        r = requests.get("https://www.forexfactory.com/calendar", headers=headers, timeout=20)
        if r.status_code != 200:
            return

        soup = BeautifulSoup(r.text, 'html.parser')
        news_items = []

        # Estratégia: varrer TODAS as rows e usar o header de dia como âncora
        # O ForexFactory retorna a semana inteira — precisamos do dia atual
        today_brt = datetime.now(BRT)
        
        # Nomes abreviados dos dias em inglês (formato do ForexFactory)
        # Ex: "Tue May 20" -> queremos só o dia 20
        today_day_num = today_brt.day  # ex: 20
        today_month   = today_brt.strftime("%b")  # ex: "May"
        
        # Flag: estamos dentro dos eventos do dia de hoje?
        in_today = False
        
        rows = soup.find_all("tr", class_=True)

        for row in rows:
            row_classes = row.get("class", [])

            # Detectar linha de separação de dia
            if "calendar__row--day-breaker" in row_classes:
                # Verificar se é o dia de hoje
                date_td = row.find("td")
                if date_td:
                    date_text = date_td.text.strip()  # ex: "Tue May 20"
                    # Verificar mês e dia
                    day_match = (today_month in date_text and str(today_day_num) in date_text)
                    in_today = day_match
                continue

            # Processar apenas rows dentro do dia de hoje
            if not in_today:
                continue

            if "calendar__row" not in row_classes:
                continue

            currency_td = row.find("td", class_="calendar__currency")
            impact_td   = row.find("td", class_="calendar__impact")
            event_td    = row.find("td", class_="calendar__event")
            time_td     = row.find("td", class_="calendar__time")

            if not (currency_td and impact_td and event_td):
                continue

            # Apenas USD
            if "USD" not in currency_td.text:
                continue

            # Apenas HIGH IMPACT — classe exata do ForexFactory: icon--ff-impact-red
            impact_span = impact_td.find("span")
            if not impact_span:
                continue
            impact_classes = impact_span.get("class", [])
            is_high = (
                "icon--ff-impact-red" in impact_classes or
                "universal-impact__impact-high" in impact_classes or
                "universal-impact__impact-high--ff" in impact_classes
            )
            if not is_high:
                continue

            time_val  = time_td.text.strip() if time_td else "---"
            event_val = event_td.text.strip() if event_td else "Evento Desconhecido"

            news_items.append({
                "title": event_val,
                "impact": "High",
                "time": time_val,
                "name": event_val
            })

        # Filtra eventos que já passaram (margem de segurança: 10 minutos após o horário)
        ny_tz = ZoneInfo("America/New_York")
        now_ny = datetime.now(ny_tz)
        active_news = []
        for item in news_items:
            event_dt = parse_news_time_to_datetime(item.get("time", ""))
            if event_dt is None:
                active_news.append(item)  # Sem horário definido: mantém por precaução
                continue
            # Mantém apenas se ainda não passou (com 10min de margem)
            if now_ny < event_dt + timedelta(minutes=10):
                active_news.append(item)

        state["news"] = active_news
        state["last_news_date"] = datetime.now(BRT).strftime("%Y-%m-%d")
        save_state()

        if active_news:
            titles = ", ".join([n["title"][:20] for n in active_news[:3]])
            add_log(
                f"NEWS RADAR: {len(active_news)} evento(s) vermelho(s) pendente(s) hoje — {titles}",
                "[SPECTRE]", "gold"
            )
        else:
            add_log("NEWS RADAR: Sem eventos vermelhos USD pendentes hoje.", "[SPECTRE]", "gold")

    except Exception as e:
        print(f"[fetch_news_daily] Erro: {e}")


smc_engine = MimirSMCEngine()
smc_gate = SMCValidationGate(smc_engine)

def core_engine():
    add_log("Soberania v8.4.2: Precisão Tática.", "[CORE]", "gold")
    last_macro_update = 0
    last_mentfx_update = 0  # Controle do scraper MentFX (atualiza a cada 30min)
    last_news_date = ""     # Controle do News Radar (atualiza 1x/dia)
    while True:
        loop_start = time.time()
        try:
            sync_real_balance() # Sincroniza saldo real no início do loop
            
            # --- BLINDAGEM SOBERANA: FORÇAR MUNIÇÃO CORRETA ---
            if state["ammo"] > 3:
                state["ammo"] = 3
                save_state()
            
            # --- RESET DIÁRIO DE MUNIÇÃO ---
            today_str = datetime.now(BRT).strftime("%Y-%m-%d")
            if state.get("last_ammo_reset") != today_str:
                state["ammo"] = 3
                state["last_ammo_reset"] = today_str
                add_log(f"REFORÇO DE ARSENAL: Munição resetada para 3/3 para o dia {today_str}.", "[SISTEMA]", "gold")
                save_state()

            # --- TRAVA DE SEGURANÇA: PROTOCOLO SOBERANO ---
            # 1.5% de perda diária máxima sobre o saldo
            max_drawdown = state.get("balance", 50000.0) * 0.015
            
            if state.get("daily_pnl", 0) <= -max_drawdown:
                if state["status"] != "HALTED-DRAWDOWN":
                    state["status"] = "HALTED-DRAWDOWN"
                    add_log(f"ESCUDO DE ODIN: Perda diária excedeu 1.5% (${max_drawdown:.2f}). Sistema travado.", "[RISCO]", "blood")
                    save_state()
                time.sleep(60)
                continue
            
            # VISÃO SAGRADA: Preço e Spread atualizados SEMPRE no início do ciclo
            df_5m, df_15m, df_1h, df_d1 = load_mt5_data()
            if df_5m is not None and not df_5m.empty:
                new_price = float(df_5m['Close'].iloc[-1])
                if new_price > 0:
                    state["current_price"] = new_price
                    state["current_spread"] = round(df_5m['Spread'].iloc[-1], 4)

            # --- ANÁLISE DIÁRIA SMC (NARRATIVA D1) ---
            if df_d1 is not None and not df_d1.empty and len(df_d1) >= 3:
                try:
                    narrative = smc_engine.project_daily_narrative(df_d1)
                    old_bias = state.get("daily_bias")
                    new_bias = narrative.get("bias", "NEUTRAL")
                    state["daily_narrative"] = narrative
                    
                    if new_bias in ["BULL", "BEAR"] and old_bias != new_bias:
                        state["daily_bias"] = new_bias
                        add_log(f"REVELAÇÃO D1: Tendência diária calculada: {narrative.get('scenario')} ({new_bias}) - {narrative.get('reason')}", "[SISTEMA]", "gold")
                        save_state()
                except Exception as e:
                    print(f"Erro ao calcular narrativa diária: {e}")
            
            # Limite de Munição (3 tiros de 0.5% por dia)
            if state.get("ammo", 3) <= 0:
                if state["status"] != "OUT-OF-AMMO":
                    state["status"] = "OUT-OF-AMMO"
                    add_log("MUNIÇÃO ESGOTADA: 3 tentativas concluídas. Sessão encerrada para proteção.", "[RISCO]", "blood")
                    save_state()
                time.sleep(60)
                continue

            if loop_start - last_macro_update > 60:
                fetch_macro_data()
                last_macro_update = loop_start

            # --- NEWS RADAR: 1x/dia, silencioso, APENAS vermelhas ---
            today_news_str = datetime.now(BRT).strftime("%Y-%m-%d")
            if last_news_date != today_news_str:
                def _fetch_news():
                    fetch_news_daily()
                import threading as _tn
                _tn.Thread(target=_fetch_news, daemon=True).start()
                last_news_date = today_news_str
            # Sincroniza last_news_date da RAM com o estado persistido
            elif not last_news_date and state.get("last_news_date"):
                last_news_date = state["last_news_date"]

            # --- SENTIMENTO MENTFX (Playwright, 30min TTL) ---
            # Roda em thread separada para não travar o loop principal
            MENTFX_INTERVAL = 30 * 60  # 30 minutos
            if loop_start - last_mentfx_update > MENTFX_INTERVAL:
                def _update_sentiment():
                    try:
                        data = scrape_mentfx("XAUUSD")
                        if data:
                            state["sentiment"]["intraday"] = data["intraday"]
                            state["sentiment"]["daily"]    = data["daily"]
                            save_state()
                            add_log(
                                f"MENTFX SYNC: INTRADAY {data['intraday']['long']}%L/{data['intraday']['short']}%S | "
                                f"DAILY {data['daily']['long']}%L/{data['daily']['short']}%S",
                                "[SPECTRE]", "cyan"
                            )
                    except Exception as e:
                        add_log(f"MENTFX SCRAPER ERROR: {e}", "[SPECTRE]", "blood")
                import threading as _t
                _t.Thread(target=_update_sentiment, daemon=True).start()
                last_mentfx_update = loop_start

            if not df_5m.empty:
                new_price = float(df_5m['Close'].iloc[-1])
                if new_price > 0:
                    state["current_price"] = new_price
                    if loop_start - state.get("last_price_sync_log", 0) > 3600:
                        add_log(f"Sincronia MT5: Preço atualizado para {state['current_price']:.2f}", "[SISTEMA]", "cyan")
                        state["last_price_sync_log"] = loop_start
            
            # --- AGENDADOR DIÁRIO DO CONSELHO (BRT) ---
            now_dt = datetime.now(BRT)
            now_hour = now_dt.hour
            now_minute = now_dt.minute
            today_str = now_dt.strftime("%Y-%m-%d")

            # 1. Reunião de Abertura (08:30 BRT)
            if now_hour == 8 and now_minute == 30 and state.get("last_briefing_date") != today_str:
                state["last_briefing_date"] = today_str
                save_state()
                def _run_morning():
                    trigger_scheduled_meeting("MORNING_BRIEFING")
                threading.Thread(target=_run_morning, daemon=True).start()

            # 2. Reunião de Fechamento (17:00 BRT)
            if now_hour == 17 and now_minute == 0 and state.get("last_closing_date") != today_str:
                state["last_closing_date"] = today_str
                save_state()
                def _run_evening():
                    trigger_scheduled_meeting("EVENING_CLOSING")
                threading.Thread(target=_run_evening, daemon=True).start()

            # 3. Reunião de Planejamento (22:00 BRT)
            if now_hour == 22 and now_minute == 0 and state.get("last_planning_date") != today_str:
                state["last_planning_date"] = today_str
                save_state()
                def _run_night():
                    trigger_scheduled_meeting("NIGHT_PLANNING")
                threading.Thread(target=_run_night, daemon=True).start()

            # --- LÓGICA DE JANELA SNIPER (09:00 - 17:00 BRASÍLIA) ---
            # Janela de Operação do Arquiteto: 09:00 - 17:00 BRT
            is_ny_session = (9 <= now_hour < 17)
            state["session"] = "NY_ACTIVE" if is_ny_session else "OUT_OF_SESSION"
            
            # Modo BATTLE apenas no horário nobre, PREDICTIVE no restante para economizar tokens
            session_type = "BATTLE" if is_ny_session else "PREDICTIVE_SESSION"
            spread_pts = state["current_spread"] * 100 if state["current_spread"] < 1.0 else state["current_spread"]

            validation = smc_gate.validate(df_15m=df_15m, df_5m=df_5m, current_price=state["current_price"], spread_points=spread_pts)

            # --- BLINDAGEM DE NOTÍCIAS MY FUNDING PIPS (SPECTRE) ---
            is_restricted, news_reason = check_news_restriction()
            if is_restricted:
                validation["approved"] = False
                validation["reject_reason"] = news_reason
                ready_to_summon = False
                if loop_start - state.get("last_news_warn_time", 0) > 300:
                    add_log(news_reason, "[SPECTRE]", "blood")
                    state["last_news_warn_time"] = loop_start
            else:
                # --- MANUTENÇÃO DE SOBERANIA: CONSULTO SEMPRE QUE HOUVER MOVIMENTO REAL ---
                # Se houver expansão, sweep confirmado ou toque em POI, convocamos o conselho para auditoria.
                # Não dependemos apenas do 'approved' que é muito rígido.
                ready_to_summon = validation["approved"] or validation.get("sweep_confirmed") or validation.get("poi_hit") or abs(state["current_price"] - state.get("last_price_logged", 0)) > 5.0


            if not validation["approved"] and loop_start - state.get("last_reject_log", 0) > 300:
                if validation.get("direction") or "SPREAD" in validation.get("reject_reason", ""):
                    label = f"Setup {validation.get('direction')}" if validation.get("direction") else "Mercado"
                    add_log(f"{label} Reprovado no Gate: {validation.get('reject_reason')}. Consultando Conselho...", "[ENGINE]", "white")
                    state["last_reject_log"] = loop_start

            # --- VIGILÂNCIA ATIVA FORGE SPIRIT (LOGS PROATIVOS) ---
            if loop_start - state.get("last_scout_log", 0) > 30: # Log a cada 30s ou evento importante
                scout_msg = ""
                if validation.get("sweep_confirmed"):
                    scout_msg = f"SWEEP DETECTADO: Preço varreu {validation['sweep_type']} em {validation['sweep_level']:.2f}."
                elif validation.get("poi_hit"):
                    scout_msg = f"ZONA DE INTERESSE: Preço mitigando {validation['poi_type']} em {validation['poi_level']:.2f}."
                elif validation.get("structure_break"):
                    scout_msg = f"QUEBRA DE ESTRUTURA: {validation.get('break_type', 'BOS')} detectado em {validation.get('break_level', 0):.2f}."
                elif abs(state["current_price"] - state.get("last_price_logged", 0)) > 3.0: # Sensibilidade aumentada de 5 para 3
                    scout_msg = f"MOVIMENTO: Preço deslocando para {state['current_price']:.2f}."

                if scout_msg:
                    # Trava de Repetição: Só loga se a mensagem for nova
                    if scout_msg != state.get("last_reported_events"):
                        add_log(f"FORGE SPIRIT: {scout_msg}", "[SCOUT]", "white")
                        state["last_reported_events"] = scout_msg
                        state["last_scout_log"] = loop_start
                        state["last_price_logged"] = state["current_price"]
                elif loop_start - state.get("last_scout_log", 0) > 900: # Heartbeat a cada 15 min
                    add_log("FORGE SPIRIT: Vigilância ativa. Mercado em baixa volatilidade.", "[SCOUT]", "white")
                    state["last_scout_log"] = loop_start
                    state["last_reported_events"] = "" # Reseta a trava no heartbeat

            # --- FILTRO DE SOBERANIA: TENDÊNCIA DIÁRIA OBRIGATÓRIA ---
            # Se o bias não for AUTO, o sistema só permite convocar o conselho na direção definida.
            bias_aligned = True
            if state["bias"] != "AUTO":
                if validation.get("direction") and state["bias"] != validation.get("direction"):
                    bias_aligned = False
                    if loop_start - state.get("last_bias_veto_log", 0) > 1800: # Log a cada 30 min
                        add_log(f"VETO DE DIREÇÃO: Setup {validation.get('direction')} ignorado. O Bias do dia é {state['bias']}.", "[RISCO]", "cyan")
                        state["last_bias_veto_log"] = loop_start

            # Disparo do Conselho (Apenas se estiver na Janela Sniper para economizar Tokens)
            if ready_to_summon and bias_aligned and state["ammo"] > 0 and state["status"] == "AUTO-HUNTING":
                if is_ny_session:
                    # TRAVA SOBERANA: Cooldown de 5 minutos entre QUALQUER disparo automático
                    if loop_start - state.get("last_trade_time", 0) > 300:
                        add_log(f"Setup {validation.get('direction', 'SMC')} detectado. Iniciando sessão BATTLE...", "[ENGINE]", "gold")
                        trigger_war_council("BATTLE", validation_data=validation)
                        state["last_trade_time"] = time.time()
                        state["last_hourly_council_hour"] = now_hour # Bloqueia a horária se atirou agora
                        save_state()

                # --- MANUTENÇÃO HORÁRIA AUTOMÁTICA (SOBERANIA AUMENTADA) ---
                if state.get("last_hourly_council_hour") != now_hour:
                    # Só dispara a horária se não houver trade recente (cooldown de 5 min)
                    if loop_start - state.get("last_trade_time", 0) > 300:
                        state["last_hourly_council_hour"] = now_hour
                        save_state()
                        add_log(f"VIGÍLIA HORÁRIA: Invocando Conselho para auditoria técnica de {now_hour:02d}:00.", "[SISTEMA]", "gold")
                        def _run_hourly():
                            trigger_war_council("HOURLY_MAINTENANCE", validation_data=validation, trade_allowed=is_ny_session)
                        threading.Thread(target=_run_hourly, daemon=True).start()
            
            elif is_ny_session and state["ammo"] > 0 and state["status"] == "AUTO-HUNTING":
                # Fallback para auditoria horária se nenhum setup foi detectado no gate reativo
                if state.get("last_hourly_council_hour") != now_hour:
                    state["last_hourly_council_hour"] = now_hour
                    save_state()
                    add_log(f"VIGÍLIA HORÁRIA: Invocando Conselho para auditoria de rotina ({now_hour:02d}:00).", "[SISTEMA]", "gold")
                    def _run_routine():
                        trigger_war_council("HOURLY_MAINTENANCE", validation_data=validation, trade_allowed=is_ny_session)
                    threading.Thread(target=_run_routine, daemon=True).start()

            elif not is_ny_session:
                pass

        except Exception as e: add_log(f"Erro Core: {e}", "[CORE]", "blood")
        time.sleep(max(0.1, 5.0 - (time.time() - loop_start)))

async def socket_handler(reader, writer):
    global current_pending_signal, current_signal_result
    try:
        while True:
            data = await reader.read(4096)
            if not data:
                break
            message = data.decode('utf-8').strip()
            if not message:
                continue
            
            try:
                req = json.loads(message)
            except Exception:
                writer.write(b"{\"error\":\"Invalid JSON\"}\n")
                await writer.drain()
                continue
            
            req_type = req.get("type")
            if req_type == "POLL":
                if current_pending_signal:
                    resp = {
                        "action": "ORDER",
                        "id": current_pending_signal["id"],
                        "direction": current_pending_signal["action"],
                        "price": current_pending_signal["price"],
                        "sl": current_pending_signal["sl"],
                        "tp": current_pending_signal["tp"],
                        "lot": current_pending_signal["lot"]
                    }
                else:
                    resp = {"action": "NONE"}
                writer.write(json.dumps(resp).encode('utf-8') + b"\n")
                await writer.drain()
            
            elif req_type == "RESULT":
                sig_id = req.get("id")
                status = req.get("status")
                msg = req.get("message", "")
                
                if current_pending_signal and sig_id == current_pending_signal["id"]:
                    current_signal_result = {
                        "id": sig_id,
                        "status": status,
                        "message": msg
                    }
                    current_pending_signal = None
                
                resp = {"action": "ACK"}
                writer.write(json.dumps(resp).encode('utf-8') + b"\n")
                await writer.drain()
                
    except Exception:
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except:
            pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Inicia o servidor Socket TCP
    server = await asyncio.start_server(socket_handler, '0.0.0.0', 5555)
    asyncio.create_task(server.serve_forever())
    add_log("Servidor Socket TCP ativo na porta 5555.", "[SISTEMA]", "gold")

    # Inicia o motor em uma thread separada
    engine_thread = threading.Thread(target=core_engine, daemon=True)
    engine_thread.start()
    yield
    # Shutdown: Limpeza
    server.close()
    await server.wait_closed()
    add_log("Soberania em pausa. Sistema encerrado.", "[SISTEMA]", "blood")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==============================================================================
# GESTÃO DE RISCO E SINCRONIA DE SALDO
# ==============================================================================
def sync_real_balance():
    if os.path.exists(ACCOUNT_FILE):
        try:
            with open(ACCOUNT_FILE, "r") as f:
                data = json.load(f)
                new_balance = data.get("balance")
                if new_balance:
                    state["balance"] = new_balance
                    state["equity"] = data.get("equity", new_balance)
        except Exception as e:
            print(f"Erro ao sincronizar saldo: {e}")

def calculate_lot(stop_loss_pts):
    risk_amount = state.get("balance", 50000.0) * 0.004  # 0.4% fixo do saldo
    if stop_loss_pts <= 0: return 0.01
    # XAUUSD: 1 lote padrão = 100 oz
    # 1 ponto (0.01 no preço) × 100 oz = $1.00 por lote
    # Lote = Risco_$ / (SL_pontos × $1.00)
    point_value = 1.0  # $1.00 por ponto por lote — XAUUSD padrão
    lot = risk_amount / (stop_loss_pts * point_value)
    return round(max(0.01, min(lot, 2.0)), 2)  # Teto soberano: 2 lotes

@app.post("/clear-news")
async def clear_news():
    """
    Limpa manualmente a lista de notícias em memória,
    removendo eventos expirados sem reiniciar o servidor.
    """
    from datetime import timedelta
    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny_tz)
    before = len(state.get("news", []))
    active = []
    for item in state.get("news", []):
        event_dt = parse_news_time_to_datetime(item.get("time", ""))
        if event_dt is None or now_ny < event_dt + timedelta(minutes=10):
            active.append(item)
    state["news"] = active
    state["last_news_date"] = datetime.now(BRT).strftime("%Y-%m-%d")
    save_state()
    removed = before - len(active)
    add_log(f"NEWS RADAR: {removed} evento(s) expirado(s) removido(s) manualmente.", "[SPECTRE]", "cyan")
    return {"status": "success", "removed": removed, "active": len(active)}

@app.post("/update-sentiment")

async def update_sentiment(data: dict):
    """
    Endpoint para atualizar o sentimento MentFX (Intraday/Daily).
    Formato esperado: {"type": "intraday"|"daily", "long": int, "short": int}
    """
    s_type = data.get("type")
    if s_type in ["intraday", "daily"]:
        state["sentiment"][s_type]["long"] = data.get("long", 50)
        state["sentiment"][s_type]["short"] = data.get("short", 50)
        save_state()
        add_log(f"SENTIMENTO {s_type.upper()} ATUALIZADO: {state['sentiment'][s_type]['long']}% L / {state['sentiment'][s_type]['short']}% S", "[SPECTRE]", "info")
        return {"status": "success"}
    return {"status": "error", "message": "Invalid sentiment type"}

@app.get("/status")
def get_status(): 
    sync_real_balance()
    # Garante que os agentes existam no estado se por algum motivo foram limpos
    if not state.get("agent_status"):
        state["agent_status"] = {
            "Forge Spirit": {"status": "IDLE", "task": "Vigilância Matemática"},
            "Rubick": {"status": "IDLE", "task": "Mimesis Geométrica"},
            "Kunkka": {"status": "IDLE", "task": "Monitoramento Macro"},
            "Spectre": {"status": "IDLE", "task": "Radar de Notícias"},
            "Oracle": {"status": "IDLE", "task": "Dialética Técnica"},
            "Invoker": {"status": "IDLE", "task": "Aguardando Veredicto"}
        }
    return state

@app.get("/agents")
def get_agents():
    return state.get("agent_status", {})

@app.get("/signals")
def get_signals(price: float = None):
    # O EA do MT5 chama este endpoint a cada 1s
    if not state["signals"]: return {}
    
    # Pega o último sinal não processado
    last_sig = state["signals"][-1]
    
    # --- BLINDAGEM DE SINAL ---
    # Aumentamos a vida do sinal para 10 minutos (600s) para garantir que o MT5 consiga capturar
    # mesmo em caso de lag no Wine ou reinício do terminal.
    if time.time() - last_sig["timestamp"] > 600: 
        return {}
        
    # Retorno ultra-compacto e determinístico para o parsing rudimentar do MQL5
    data = {
        "id": int(last_sig["id"]),
        "side": str(last_sig.get("action", last_sig.get("side", "BEAR"))), # Suporta chaves legacy e novas
        "price": float(last_sig["price"]),
        "sl": float(last_sig["sl"]),
        "tp": float(last_sig["tp"]),
        "lot": float(last_sig["lot"])
    }
    return data

@app.get("/api/logs")
def get_api_logs(limit: int = 100, offset: int = 0, sys: str = None):
    return logger.get_logs(limit=limit, offset=offset, system_filter=sys)

@app.post("/update-trade-result")
def update_result(profit: float):
    state["daily_pnl"] += profit
    # state["ammo"] = 3 if state["daily_pnl"] >= 0 else state["ammo"] # Removido para permitir dreno real
    add_log(f"Resultado Trade: ${profit:.2f}. PnL Diário: ${state['daily_pnl']:.2f}", "[ENGINE]", "success")
    save_state()
    return {"status": "ok"}

@app.post("/set-bias/{new_bias}")
def set_bias(new_bias: str, background_tasks: BackgroundTasks):
    state["bias"] = new_bias
    if new_bias == "AUTO":
        state["bias_locked_until"] = 0
        now_hour = datetime.now(BRT).hour
        is_trading_window = (9 <= now_hour < 17)
        if is_trading_window:
            add_log("BIAS retornado para modo AUTÔNOMO. Convocando Conselho para Veredicto...", "[SISTEMA]", "gold")
            background_tasks.add_task(trigger_war_council, "PREDICTIVE", None, None, True)
        else:
            add_log(f"BIAS AUTÔNOMO ativo. Fora da Janela Sniper — Conselho em modo ANÁLISE (sem ordens). Hora: {now_hour:02d}:00 BRT.", "[SISTEMA]", "cyan")
            background_tasks.add_task(trigger_war_council, "PREDICTIVE_SESSION", None, None, False)
    else:
        state["bias_locked_until"] = time.time() + 3600
        add_log(f"BIAS alterado manualmente para {new_bias}", "[SISTEMA]", "gold")
    save_state()
    return {"status": "ok", "bias": state["bias"]}

@app.post("/conjure/{spell}")
async def conjure_spell(spell: str, background_tasks: BackgroundTasks, data: dict = None):
    global state
    data = data or {}
    raw_message = data.get("message", "")
    
    if spell == "summon_council":
        now_hour = datetime.now(BRT).hour
        # Invocação manual nunca deve abrir ordem automaticamente via trigger_war_council.
        # Ela serve para análise de BIAS e debate do Conselho.
        add_log("Invocação Manual do Conselho iniciada (Modo Análise de Bias).", "[CONSELHO]", "gold")
        background_tasks.add_task(trigger_war_council, "MANUAL", None, raw_message, False)
        return {"status": "success", "message": "Council summoned for strategic analysis."}
    
    elif spell == "strike_test":
        add_log("ORDEM DE ATAQUE FORÇADA (Strike Test). Injetando sinal físico no disco...", "[STRIKE]", "gold")
        price = state.get('current_price', 4540.0)
        
        # Prioriza o 'type' enviado no payload, senão usa o bias
        manual_type = data.get("type", "").upper()
        if manual_type in ["BUY", "SELL", "BULL", "BEAR"]:
            action = "BUY" if manual_type in ["BUY", "BULL"] else "SELL"
        else:
            bias = state.get('bias', 'BEAR')
            action = "BUY" if bias == "BULL" else "SELL"
        
        # --- PROTOCOLO SOBERANO DE RISCO (0.5% / 700 pts) ---
        fixed_sl_pts = 700
        sl_dist = fixed_sl_pts / 100.0  # 700 pontos → 7.00 de distância no preço
        tp_dist = (fixed_sl_pts * 5) / 100.0  # RR 1:5

        sl = round(price + sl_dist, 2) if action == "SELL" else round(price - sl_dist, 2)
        tp = round(price - tp_dist, 2) if action == "SELL" else round(price + tp_dist, 2)
        lot = calculate_lot(fixed_sl_pts)  # 0.5% do saldo real
        signal_id = int(time.time() * 1000) % 100000000

        add_log(f"Strike Soberano: {action} | Lote={lot} | SL={sl} | TP={tp} | Risco=0.5%", "[STRIKE]", "gold")

        try:
            global current_pending_signal, current_signal_result
            current_signal_result = None
            current_pending_signal = {
                "id": signal_id, "action": action,
                "price": price, "sl": sl, "tp": tp, "lot": lot
            }
            
            with open(SIGNAL_FILE, "w") as f:
                f.write(f"ACTION={action}\n")
                f.write(f"PRICE={price}\n")
                f.write(f"SL={sl}\n")
                f.write(f"TP={tp}\n")
                f.write(f"LOT={lot}\n")
                f.write(f"ID={signal_id}\n")
                f.flush()
                os.fsync(f.fileno())
            # state['ammo'] -= 1  # REMOVIDO: Strike de teste não consome munição real
            save_state()
            add_log(f"Sinal de Teste {action} injetado via Socket/Arquivo (ID {signal_id})", "[STRIKE]", "success")
        except Exception as e:
            add_log(f"Erro ao gravar sinal de teste: {e}", "[STRIKE]", "blood")
        
        save_state()
        return {"status": "success", "message": f"Strike signal {signal_id} injected to disk."}
    
    elif spell == "reset_ammo":
        state["ammo"] = 3
        state["daily_pnl"] = 0.0
        state["status"] = "AUTO-HUNTING"
        add_log("MUNIÇÃO RECARREGADA: O pente está cheio (3/3). Drawdown resetado.", "[SISTEMA]", "success")
        save_state()
        return {"status": "success", "message": "Ammo and Drawdown reset."}
    
    elif spell in ["whisper", "chat"]:
        if not raw_message: return {"status": "error", "message": "No message provided."}
        
        # Roteamento Inteligente: "Kunkka: Como está o DXY?"
        agent = data.get("agent", "Oracle")
        message = raw_message
        
        if ":" in raw_message[:15]: # Busca prefixo "Agente: "
            parts = raw_message.split(":", 1)
            potential_agent = parts[0].strip().title()
            # Validação de nome de agente (flexível)
            for active_agent in state["agent_status"].keys():
                if potential_agent in active_agent:
                    agent = active_agent
                    message = parts[1].strip()
                    break
        
        add_log(f"Arquiteto para {agent}: {message}", "[WHISPER]", "cyan")
        
        def agent_task():
            # Injeção de Contexto Específico por Agente
            context = ""
            if agent == "Kunkka":
                context = f"DXY Trend: {state['dxy_trend']}, US10Y Trend: {state['us10y_trend']}. "
            elif agent == "Spectre":
                context = f"News: {json.dumps(state['news'])}. "
            elif agent == "Rubick":
                try:
                    index_path = "/home/mimir/Documentos/MIMIR/03_TRADING_SMC/LORE_DOS_GRAFICOS/lore_index_kael.json"
                    with open(index_path, "r") as f:
                        lore_index = json.load(f)
                    current_tf = "M15" if "M15" in str(state.get("logs", "")) else "M5"
                    context = (f"LORE INDEX: {json.dumps(lore_index[-10:])}. TIMEFRAME: {current_tf}. "
                               f"Compare e cite ID/Gatilho. Sem poesia.")
                except: context = "Use o Grimório SMC. Cite ID e Gatilho."
            
            response = invoker_brain.get_ai_response(agent, context + message)
            add_council_msg(agent, response, "gold")
            add_log(f"{agent} respondeu ao sussurro.", "[WHISPER]", "success")

        background_tasks.add_task(agent_task)
        return {"status": "success", "message": f"Whispering to {agent}..."}

    return {"status": "unknown", "message": "Unknown incantation."}

@app.post("/invoker/chat")
async def invoker_chat(background_tasks: BackgroundTasks, data: dict = None):
    data = data or {"message": ""}
    return await conjure_spell("whisper", background_tasks, {"agent": "Invoker", "message": data.get("message", "")})

@app.post("/council/chat")
async def council_chat(background_tasks: BackgroundTasks, data: dict = None):
    data = data or {"message": ""}
    return await conjure_spell("whisper", background_tasks, {"agent": "Oracle", "message": data.get("message", "")})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
