import requests
import json
import os
from datetime import datetime

def fetch_tradingview_news():
    """
    Radar de Spectre via TradingView (A Malícia da API Interna).
    Busca notícias de alto impacto USD.
    """
    radar_path = "/home/mimir/Documentos/MIMIR/00_CORE/Configuracoes/news_radar.json"
    
    # Datas para a consulta (Hoje e Amanhã para garantir cobertura)
    now = datetime.utcnow()
    date_from = now.strftime("%Y-%m-%dT00:00:00.000Z")
    date_to = (now + timedelta(days=2)).strftime("%Y-%m-%dT23:59:59.000Z")
    
    url = f"https://economic-calendar.tradingview.com/events?from={date_from}&to={date_to}&countries=US"
    headers = {
        "Origin": "https://www.tradingview.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            events = data.get("result", [])
            usd_red_events = []
            
            for event in events:
                # TradingView usa 'importance': 1 (High), 0 (Medium), -1 (Low)
                if event.get("importance") == 1:
                    usd_red_events.append({
                        "event": event.get("title"),
                        "time": event.get("date"), # ISO format
                        "impact": "HIGH"
                    })
            
            with open(radar_path, "w") as f:
                json.dump({
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "events": usd_red_events
                }, f, indent=4)
            
            print(f"✅ Radar de Notícias SOBERANO (TV): {len(usd_red_events)} eventos RED detectados.")
            return True
        else:
            print(f"Erro na API do TradingView: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Erro no Radar de Spectre: {e}")
        return False

from datetime import timedelta
if __name__ == "__main__":
    fetch_tradingview_news()
