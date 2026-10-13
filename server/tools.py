"""The real Python tools the model can call.

TOOL_SCHEMAS is what the model sees (xLAM format, goes into the prompt).
TOOL_FUNCS is what the gateway actually runs. Names must match.
All tools use free APIs with no key: Open-Meteo (weather) and Frankfurter (currency).
"""
import ast
import operator

import httpx

TOOL_SCHEMAS = [
    {"name": "get_weather", "description": "Get the current weather for a city.",
     "parameters": {"city": {"type": "str", "description": "City name"}}},
    {"name": "convert_currency", "description": "Convert an amount from one currency to another.",
     "parameters": {"amount": {"type": "float", "description": "Amount to convert"},
                    "from_currency": {"type": "str", "description": "3-letter source currency code, e.g. INR"},
                    "to_currency": {"type": "str", "description": "3-letter target currency code, e.g. USD"}}},
    {"name": "calculate", "description": "Evaluate a basic arithmetic expression.",
     "parameters": {"expression": {"type": "str", "description": "Expression like '23 * 47'"}}},
    {"name": "send_email", "description": "Send an email to someone.",
     "parameters": {"to": {"type": "str", "description": "Recipient email address"},
                    "subject": {"type": "str", "description": "Email subject"},
                    "body": {"type": "str", "description": "Email body"}}},
]

# Tools with side effects: the gateway asks the user to confirm before running them.
RISKY_TOOLS = {"send_email"}

WEATHER_CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
                 51: "light drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
                 80: "rain showers", 95: "thunderstorm"}


async def get_weather(city: str) -> dict:
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http:
        geo = (await http.get("https://geocoding-api.open-meteo.com/v1/search",
                              params={"name": city, "count": 1})).json()
        if not geo.get("results"):
            return {"error": f"City '{city}' not found"}
        place = geo["results"][0]
        wx = (await http.get("https://api.open-meteo.com/v1/forecast",
                             params={"latitude": place["latitude"], "longitude": place["longitude"],
                                     "current": "temperature_2m,weather_code,wind_speed_10m"})).json()
    cur = wx["current"]
    return {"city": place["name"], "country": place.get("country"),
            "temperature_c": cur["temperature_2m"], "wind_kmh": cur["wind_speed_10m"],
            "condition": WEATHER_CODES.get(cur["weather_code"], f"code {cur['weather_code']}")}


async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    src, dst = from_currency.upper(), to_currency.upper()
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http:
        r = await http.get("https://api.frankfurter.dev/v1/latest",
                           params={"amount": amount, "from": src, "to": dst})
    if r.status_code != 200:
        return {"error": f"Could not convert {src} to {dst} (HTTP {r.status_code}: {r.text[:100]})"}
    return {"amount": amount, "from": src, "to": dst, "result": r.json()["rates"][dst]}


# Safe calculator: walks the expression tree instead of eval(), so the model
# (or a user) can't sneak in code like __import__('os').system(...)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


async def calculate(expression: str) -> dict:
    try:
        return {"expression": expression, "result": _eval(ast.parse(expression, mode="eval").body)}
    except Exception as e:
        return {"error": f"Can't evaluate '{expression}': {e}"}


async def send_email(to: str, subject: str = "", body: str = "") -> dict:
    """Fake: nothing is actually sent. It exists to demo the confirmation guardrail."""
    return {"status": "sent (simulated)", "subject": subject}


TOOL_FUNCS = {"get_weather": get_weather, "convert_currency": convert_currency,
              "calculate": calculate, "send_email": send_email}
