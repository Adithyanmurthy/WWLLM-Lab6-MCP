"""
Experiment 2 — "Data Dashboard" Connector (MCP server)

Connects an AI to a real-time data source: the free wttr.in weather API (no key needed).
Tools:
    get_current_weather(location)   -> live conditions as structured text
    get_forecast(location)           -> 3-day forecast (today / tomorrow / day after)

Run directly (stdio transport):   python server.py
Normally it is started by app.py (Streamlit) or cli.py.
"""

from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-connector", log_level="WARNING")


def fetch(location: str) -> dict:
    """Call wttr.in's JSON API (format=j1)."""
    r = httpx.get(f"https://wttr.in/{quote(location.strip())}", params={"format": "j1"},
                  headers={"User-Agent": "mcp-lab6"}, timeout=20)
    r.raise_for_status()
    return r.json()


def place_name(data: dict, requested: str) -> str:
    """wttr.in reports the nearest weather station, which can differ from the city name
    (e.g. Tokyo -> Shikinejima), so show both."""
    try:
        a = data["nearest_area"][0]
        station, country = a["areaName"][0]["value"], a["country"][0]["value"]
    except (KeyError, IndexError):
        return requested
    if station.lower() == requested.strip().lower():
        return f"{station}, {country}"
    return f"{requested.strip().title()}, {country} (nearest station: {station})"


@mcp.tool()
def get_current_weather(location: str) -> str:
    """Get the current, real-time weather for a city or place, e.g. "Tokyo" or "Bengaluru"."""
    try:
        data = fetch(location)
        c = data["current_condition"][0]
    except Exception as e:  # network error, unknown place, API change
        return f"ERROR: could not fetch weather for '{location}': {e}"
    return "\n".join([
        f"location: {place_name(data, location)}",
        f"observed_at_utc: {c['observation_time']}",
        f"condition: {c['weatherDesc'][0]['value'].strip()}",
        f"temperature_c: {c['temp_C']}",
        f"feels_like_c: {c['FeelsLikeC']}",
        f"humidity_pct: {c['humidity']}",
        f"wind_kmph: {c['windspeedKmph']} {c['winddir16Point']}",
        f"precipitation_mm: {c['precipMM']}",
        f"cloud_cover_pct: {c['cloudcover']}",
        f"uv_index: {c['uvIndex']}",
        f"visibility_km: {c['visibility']}",
    ])


@mcp.tool()
def get_forecast(location: str) -> str:
    """Get the 3-day weather forecast for a city or place, labelled today / tomorrow /
    day after tomorrow. Use for questions about tomorrow or the coming days."""
    try:
        data = fetch(location)
    except Exception as e:
        return f"ERROR: could not fetch forecast for '{location}': {e}"
    lines = [f"location: {place_name(data, location)}"]
    labels = ["today", "tomorrow", "day after tomorrow"]
    for label, d in zip(labels, data["weather"][:3]):
        midday = d["hourly"][len(d["hourly"]) // 2]
        rain = max(int(h["chanceofrain"]) for h in d["hourly"])
        lines.append(f"{label} ({d['date']}): {midday['weatherDesc'][0]['value'].strip()}, "
                     f"min {d['mintempC']}°C / max {d['maxtempC']}°C, "
                     f"max chance of rain {rain}%")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
