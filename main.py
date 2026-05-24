"""
main.py — 欧洲司导行程 FastAPI 后端服务（Production-grade）
POST /api/v1/itinerary/generate — 异步 DeepSeek → PostgreSQL 持久化 → 分享链接
GET  /share/{id}            — Jinja2 动态白标渲染 → 散客浏览器
"""
import asyncio
import json
import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import quote, unquote, urlparse, urlunparse

import requests
from databases import Database
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, TemplateError, TemplateNotFound
from openai import APIError, APIConnectionError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field

from system_prompt import get_system_prompt

# ===========================================================================
# Logging
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("itinerary-engine")

# ===========================================================================
# Environment
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent

# .env 仅用于本地开发；生产环境由平台注入环境变量
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY or ANTHROPIC_API_KEY is required — check environment variables")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required — PostgreSQL connection string must be set")

# 兼容 Render / Railway 的 postgres:// 协议头
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# URL 安全编码：对密码组件中的特殊字符（# @ : / ?）进行百分号转义，
# 防止 asyncpg 底层 urlsplit 抛出 ValueError: Invalid IPv6 URL
_parts = urlparse(DATABASE_URL)
if _parts.password:
    _safe_password = quote(unquote(_parts.password), safe="")
    _safe_netloc = f"{_parts.username}:{_safe_password}@{_parts.hostname}"
    if _parts.port:
        _safe_netloc += f":{_parts.port}"
    DATABASE_URL = urlunparse(_parts._replace(netloc=_safe_netloc))

DEFAULT_BOOKING_AID = "QF_MAIN_DEFAULT_AID"

# ===========================================================================
# PostgreSQL 异步连接池（针对 Supabase PgBouncer 调优）
# ===========================================================================
database = Database(
    DATABASE_URL,
    min_size=2,             # 冷启动即保持 2 条长连接，消除首请求建连延迟
    max_size=20,            # 生产并发峰值预留
    command_timeout=30,     # 单条查询超时硬截止
)
logger.info("Database pool configured (min=2 max=20 timeout=30s)")


async def ensure_db_connected():
    """Auto-reconnect sentinel — guards against Supavisor cold-start / idle timeout.

    Uses an actual ``SELECT 1`` health ping rather than the unreliable
    ``database.is_connected`` flag.  Supavisor session pooler can report
    ``is_connected=False`` even when the underlying TCP socket is healthy,
    or vice versa — only a query proves the connection is really alive.
    """
    # Fast path: a live query means the pool is healthy
    try:
        await database.fetch_val("SELECT 1")
        return
    except Exception:
        pass  # connection truly dead — attempt reconnect below

    logger.warning("[DB_RECOVERY] Connection dead — attempting emergency reconnect...")
    try:
        await asyncio.wait_for(database.connect(), timeout=10.0)
        await database.fetch_val("SELECT 1")
        logger.info("[DB_RECOVERY] Reconnect + health ping succeeded")
    except Exception as exc:
        logger.error("[DB_RECOVERY] Auto-reconnect failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Database Service Temporarily Unavailable.",
        )


# ===========================================================================
# Async OpenAI client（全请求复用连接池）
# ===========================================================================
client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE,
    timeout=60.0,
    max_retries=2,
)
logger.info("AsyncOpenAI client initialized (base=%s, model=%s)", DEEPSEEK_BASE, DEEPSEEK_MODEL)

# ===========================================================================
# Jinja2 template engine
# ===========================================================================
TEMPLATES_DIR = BASE_DIR / "templates"
if not TEMPLATES_DIR.is_dir():
    raise RuntimeError(f"Templates directory not found: {TEMPLATES_DIR}")

_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=False,
    enable_async=False,
)
logger.info("Jinja2 environment loaded (%d template(s) in %s)",
            len(_jinja_env.list_templates()), TEMPLATES_DIR)

# ===========================================================================
# 上下文构建
# ===========================================================================
def _dot_color(stop_type: str) -> str:
    colors = {
        "attraction": "bg-gold",
        "restaurant": "bg-red-400",
        "shopping": "bg-purple-400",
        "other": "bg-gray-400",
    }
    return colors.get(stop_type, "bg-gold")


def _prepare_context(
    data: dict,
    *,
    user_id: str,
    guide_name: str | None,
    guide_wechat: str | None,
    booking_aid: str | None,
    security_mode: str,
) -> dict:
    """将 DeepSeek JSON + 请求参数组装为 Jinja2 模板上下文。"""
    itinerary = data.get("itinerary", [])
    summary = data.get("summary", {})

    days = []
    for d in itinerary:
        stops = [
            {**s, "dot_color": _dot_color(s.get("type", "attraction"))}
            for s in d.get("stops", [])
        ]

        # --- driver_parking 防御性归一化 ---
        # DeepSeek 可能返回 None / [] / 部分 key 为非 dict，强制清洗为安全 dict
        raw_parking = d.get("driver_parking")
        driver_parking: dict[str, dict] = {}
        if isinstance(raw_parking, dict):
            driver_parking = {
                str(k): v for k, v in raw_parking.items()
                if isinstance(v, dict) and v
            }

        days.append({**d, "stops": stops, "driver_parking": driver_parking})

    cities = summary.get("cities", [])
    title = " → ".join(cities) if cities else "欧洲定制行程"

    # --- master_schedule 防御性归一化 ---
    raw_schedule = data.get("master_schedule")
    master_schedule: list[dict] = []
    if isinstance(raw_schedule, list):
        master_schedule = [
            s for s in raw_schedule
            if isinstance(s, dict) and s.get("time_slot")
        ]

    return {
        "guide_name": guide_name or "",
        "guide_wechat": guide_wechat or "",
        "guide_verified": bool(guide_name),
        "user_id": user_id,
        "title": title,
        "cities": cities,
        "total_days": summary.get("total_days", len(itinerary)),
        "budget": summary.get("budget_range", "详询司导"),
        "security_mode": security_mode,
        "security_risk": data.get("security_risk") or "",
        "warning": data.get("warning") or "",
        "hotel_disclaimer": data.get("hotel_disclaimer", ""),
        "booking_aid": booking_aid or DEFAULT_BOOKING_AID,
        "master_schedule": master_schedule,
        "days": days,
    }


# ===========================================================================
# 景点图片抓取（Wikipedia REST API，免费无 Key）
# ===========================================================================
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "EuroTourHub/3.1 (https://github.com/az7809/travel; itinerary@eurotourhub.dev)",
}


def _fetch_images_for_attraction(name: str) -> list[str]:
    """抓取单个景点的 Wikipedia 高清图片 URL（最多 1 张）。

    使用 Wikipedia 的 pageimages API，返回 800px 宽的缩略图。
    Google Custom Search 可作为未来升级路径（需 GOOGLE_API_KEY + GOOGLE_CSE_ID）。
    """
    if not name:
        return []
    try:
        # Step 1: OpenSearch 找到最匹配的 Wikipedia 词条
        search_params = {
            "action": "opensearch",
            "search": name,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        }
        r = requests.get(WIKI_API, params=search_params, headers=WIKI_HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json()
        titles = data[1] if len(data) > 1 else []
        if not titles:
            return []
        title = titles[0]

        # Step 2: 获取词条主图（pageimages prop）
        img_params = {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "pithumbsize": 800,
            "format": "json",
        }
        r2 = requests.get(WIKI_API, params=img_params, headers=WIKI_HEADERS, timeout=8)
        r2.raise_for_status()
        pages = r2.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb:
                return [thumb]
        return []
    except Exception:
        return []


async def _fetch_day_images(days: list[dict]) -> dict[int, list[dict]]:
    """为每天的 Top 4 景点并发抓取代表性图片。

    每天最多 4 张，每景点取 1 张 Wikipedia 主图。
    所有请求通过 asyncio.to_thread 并发执行，不阻塞事件循环。
    """
    if not days:
        return {}

    async def _fetch_one(day_idx: int, stop_name: str, stop_idx: int) -> tuple[int, int, str] | None:
        urls = await asyncio.to_thread(_fetch_images_for_attraction, stop_name)
        if urls:
            return (day_idx, stop_idx, urls[0])
        return None

    # 每天取前 4 个景点的图片
    tasks = []
    for day_idx, day in enumerate(days):
        stops = day.get("stops", [])
        for stop_idx, stop in enumerate(stops[:4]):
            name = stop.get("name", "")
            if name:
                tasks.append(_fetch_one(day_idx, name, stop_idx))

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks, return_exceptions=True)

    day_images: dict[int, list[dict]] = {}
    for result in results:
        if isinstance(result, tuple):
            day_idx, stop_idx, url = result
            if day_idx not in day_images:
                day_images[day_idx] = []
            day_images[day_idx].append({
                "url": url,
                "name": days[day_idx]["stops"][stop_idx].get("name", ""),
            })

    return day_images


# ===========================================================================
# 报价计算引擎
# ===========================================================================
# 高住宿费国家（瑞士 / 挪威 / 冰岛）
_HIGH_ACCOMMODATION_COUNTRIES = frozenset({"Switzerland", "Norway", "Iceland"})


def _vehicle_cost_per_day(mileage_km: float) -> int:
    """单日车费：
    ≤100 km → €450
    100-350 km → €550
    >350 km → €650
    """
    if mileage_km <= 100:
        return 450
    elif mileage_km <= 350:
        return 550
    else:
        return 650


def _accommodation_cost_per_night(country: str) -> int:
    """司导住宿费/晚：
    瑞士/挪威/冰岛 → €120
    其余欧洲国家 → €100
    """
    return 120 if country in _HIGH_ACCOMMODATION_COUNTRIES else 100


def calculate_quote(days: list[dict]) -> dict:
    """根据每日里程和国家计算总报价。

    Returns:
        {
            "days": [{"day": 1, "mileage_km": 45, "country": "France",
                       "vehicle_eur": 450, "accommodation_eur": 100,
                       "day_total_eur": 550}, ...],
            "total_vehicle_eur": 1350,
            "total_accommodation_eur": 300,
            "grand_total_eur": 1650,
        }
    """
    quote_days = []
    total_vehicle = 0
    total_accommodation = 0

    for d in days:
        mileage = d.get("daily_mileage_km", 0)
        if not isinstance(mileage, (int, float)) or mileage < 0:
            mileage = d.get("driving_hours", 0) * 55  # fallback: 55 km/h 估算

        country = d.get("country", "").strip()

        vehicle = _vehicle_cost_per_day(mileage)
        accommodation = _accommodation_cost_per_night(country) if country else 100

        total_vehicle += vehicle
        total_accommodation += accommodation

        quote_days.append({
            "day": d.get("day", "?"),
            "mileage_km": round(mileage, 1),
            "country": country or "未指定",
            "vehicle_eur": vehicle,
            "accommodation_eur": accommodation,
            "day_total_eur": vehicle + accommodation,
        })

    return {
        "days": quote_days,
        "total_vehicle_eur": total_vehicle,
        "total_accommodation_eur": total_accommodation,
        "grand_total_eur": total_vehicle + total_accommodation,
    }


# ===========================================================================
# 静态停车场数据库 — 核心景点免 Token 注入
# ===========================================================================
# DeepSeek 只需输出 `{"_static": true}` 占位符，后端自动注入完整停车场数据。
# 节省每条行程约 2,000-4,000 tokens 的停车场描述开销。
STATIC_DATA_LIBRARY: dict[str, dict] = {
    # ── 巴黎 ─────────────────────────────────────────────
    "巴黎圣母院": {
        "name": "Parking Maubert Collège des Bernardins",
        "address": "17 Rue des Bernardins, 75005 Paris",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 3.5,
        "walk_to_attraction_min": 8,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。首选停车场（Indigo Notre-Dame 限高仅1.9m，V-Class无法进入）。步行沿塞纳河畔直达圣母院。",
    },
    "卢浮宫": {
        "name": "Parking Indigo Louvre Samaritaine",
        "address": "1 Place du Louvre, 75001 Paris",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 4.5,
        "walk_to_attraction_min": 5,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "地下停车场，V-Class 可入。入口在 Rue de l'Amiral de Coligny。19:00 后费率降至 €2.5/h。周二卢浮宫闭馆日停车场照常开放。",
    },
    "埃菲尔铁塔": {
        "name": "Parking Pullman Tour Eiffel",
        "address": "18 Avenue de Suffren, 75015 Paris",
        "height_limit_m": 1.9,
        "hourly_rate_eur": 5.0,
        "walk_to_attraction_min": 5,
        "large_vehicle_ok": False,
        "ztl_restricted": False,
        "notes": "⚠️ 限高1.9m，V-Class(1.92m)无法进入！替代方案：Parking Javel — 限高2.2m，步行至铁塔约15分钟，或让客户在铁塔入口下车后司导去停车。",
    },
    "凯旋门": {
        "name": "Parking Indigo Paris Kléber Trocadéro",
        "address": "65 Avenue Kléber, 75116 Paris",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 4.0,
        "walk_to_attraction_min": 6,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。建议停地下二层，一层车位偏窄。步行沿 Av Kléber 直达凯旋门。",
    },
    "奥赛博物馆": {
        "name": "Parking Indigo Paris Louvre Samaritaine",
        "address": "1 Place du Louvre, 75001 Paris",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 4.5,
        "walk_to_attraction_min": 8,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。跨塞纳河步行桥（Passerelle Léopold-Sédar-Senghor）直达奥赛。周一奥赛闭馆。",
    },
    "蒙马特": {
        "name": "Parking Anvers — Sacré-Cœur",
        "address": "7 Rue de Steinkerque, 75018 Paris",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 3.0,
        "walk_to_attraction_min": 10,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。步行上坡至圣心大教堂约10分钟（台阶较多）。注意：蒙马特高地部分巷道极窄，V-Class 切勿驶入 Rue des Saules 以北。",
    },
    "圣心大教堂": {
        "name": "Parking Anvers — Sacré-Cœur",
        "address": "7 Rue de Steinkerque, 75018 Paris",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 3.0,
        "walk_to_attraction_min": 10,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。注意蒙马特高地巷道狭窄，部分路段 V-Class 禁行。",
    },
    "先贤祠": {
        "name": "Parking Maubert Collège des Bernardins",
        "address": "17 Rue des Bernardins, 75005 Paris",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 3.5,
        "walk_to_attraction_min": 7,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。步行经 Rue Monge 直达先贤祠，沿途可经过拉丁区。",
    },
    "荣军院": {
        "name": "Parking Indigo Paris Invalides",
        "address": "23 Rue de Constantine, 75007 Paris",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 4.0,
        "walk_to_attraction_min": 3,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。最近停车场，入口醒目。",
    },

    # ── 凡尔赛 ───────────────────────────────────────────
    "凡尔赛宫": {
        "name": "Parking Versailles — Place d'Armes",
        "address": "Place d'Armes, 78000 Versailles",
        "height_limit_m": None,
        "hourly_rate_eur": 2.0,
        "walk_to_attraction_min": 3,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "露天停车场，V-Class 无忧。周一闭馆（⚠️ 行程安排在周一必须标注替代方案）。建议提前到达以避免旺季排队。",
    },

    # ── 罗马 ─────────────────────────────────────────────
    "斗兽场": {
        "name": "Parking Colosseo — Via dei Santi Quattro",
        "address": "Via dei Santi Quattro, 00184 Roma",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 3.0,
        "walk_to_attraction_min": 4,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。停车场位于斗兽场东南侧，ZTL 边界外。⛔ 注意：罗马历史中心 ZTL 管控严格，从此停车场步行至斗兽场入口仅4分钟。",
    },
    "梵蒂冈博物馆": {
        "name": "Parking Vespasiano — Vaticano",
        "address": "Via Vespasiano, 28, 00192 Roma",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 3.5,
        "walk_to_attraction_min": 5,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。位于梵蒂冈城墙外北侧，避开 ZTL。⚠️ 梵蒂冈博物馆每月最后一个周日免费（人流量井喷，建议避开）。",
    },
    "圣彼得大教堂": {
        "name": "Parking Vespasiano — Vaticano",
        "address": "Via Vespasiano, 28, 00192 Roma",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 3.5,
        "walk_to_attraction_min": 7,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。步行穿过圣彼得广场进入大教堂。进入大教堂需安检排队（约15-30分钟），建议早到。",
    },
    "许愿池": {
        "name": "Parking Ludovisi — Via Ludovisi",
        "address": "Via Ludovisi, 60, 00187 Roma",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 4.0,
        "walk_to_attraction_min": 10,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。许愿池位于 Tridente ZTL 核心区内，私家车禁入。此停车场在 ZTL 外，步行约10分钟。⚠️ 车内勿留任何可见物品（罗马砸车窗高发）。",
    },
    "万神殿": {
        "name": "Parking Ludovisi — Via Ludovisi",
        "address": "Via Ludovisi, 60, 00187 Roma",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 4.0,
        "walk_to_attraction_min": 12,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。万神殿位于罗马历史中心 ZTL 核心。步行沿 Via del Corso 直达，沿途可逛罗马精品街。",
    },
    "西班牙广场": {
        "name": "Parking Ludovisi — Via Ludovisi",
        "address": "Via Ludovisi, 60, 00187 Roma",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 4.0,
        "walk_to_attraction_min": 5,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。西班牙广场位于 ZTL 内。此停车场最近且 V-Class 可入。步行沿 Via dei Condotti 名店街直达。",
    },

    # ── 佛罗伦萨 ─────────────────────────────────────────
    "圣母百花大教堂": {
        "name": "Parcheggio Sant'Ambrogio",
        "address": "Piazza Sant'Ambrogio, 50121 Firenze",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 2.5,
        "walk_to_attraction_min": 12,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "🚫 **佛罗伦萨历史中心全境 ZTL**，私家车和大型商务车严禁进入。此停车场位于 ZTL 外围东侧，V-Class 可入。步行经 Borgo la Croce 直达大教堂。⚠️ ZTL 摄像头自动拍照罚款（€80-120），切勿侥幸驶入。",
    },
    "乌菲兹美术馆": {
        "name": "Parcheggio Sant'Ambrogio",
        "address": "Piazza Sant'Ambrogio, 50121 Firenze",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 2.5,
        "walk_to_attraction_min": 15,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "🚫 ZTL 外围停车场。步行经 Piazza della Signoria 直达乌菲兹。路程稍长但沿途全是文艺复兴街景。",
    },
    "大卫雕像": {
        "name": "Parcheggio Sant'Ambrogio",
        "address": "Piazza Sant'Ambrogio, 50121 Firenze",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 2.5,
        "walk_to_attraction_min": 12,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。学院美术馆（Galleria dell'Accademia）藏有大卫真迹，ZTL 外围停车后步行前往。",
    },
    "老桥": {
        "name": "Parcheggio Sant'Ambrogio",
        "address": "Piazza Sant'Ambrogio, 50121 Firenze",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 2.5,
        "walk_to_attraction_min": 18,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "🚫 ZTL 外围。步行较远（约18分钟），建议司导让客户在老桥附近下车后自行停车，电话约定集合点。",
    },

    # ── 威尼斯 ───────────────────────────────────────────
    "圣马可广场": {
        "name": "Garage San Marco — Piazzale Roma",
        "address": "Piazzale Roma, 30135 Venezia",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 6.0,
        "walk_to_attraction_min": 25,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "威尼斯主岛全境无机动车道路。**所有车辆必须停在 Piazzale Roma 或 Tronchetto**。V-Class 可入车库。之后乘水上巴士 Vaporetto（1号线约25分钟）直达圣马可广场。建议购买 24h 通票（€25/人）。",
    },
    "叹息桥": {
        "name": "Garage San Marco — Piazzale Roma",
        "address": "Piazzale Roma, 30135 Venezia",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 6.0,
        "walk_to_attraction_min": 25,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "同圣马可广场路线。叹息桥在圣马可广场旁步行 2 分钟。",
    },
    "里亚托桥": {
        "name": "Garage San Marco — Piazzale Roma",
        "address": "Piazzale Roma, 30135 Venezia",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 6.0,
        "walk_to_attraction_min": 20,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "乘 Vaporetto 1号线至 Rialto 站下（比圣马可少坐一站），V-Class 停 Piazzale Roma。",
    },

    # ── 米兰 ─────────────────────────────────────────────
    "米兰大教堂": {
        "name": "Parking Piazza Meda — Autosilo",
        "address": "Piazza Filippo Meda, 20121 Milano",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 4.5,
        "walk_to_attraction_min": 6,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "☢️ **米兰 Area C ZTL 工作日 7:30-19:30 全境限行**。此停车场位于 Area C 边界内，V-Class 需购买 Area C 通行票（€5/天，在烟草店 Tabacchi 或线上购买）。建议周六/周日前往可免 Area C 费。",
    },
    "最后的晚餐": {
        "name": "Parking Piazza Meda — Autosilo",
        "address": "Piazza Filippo Meda, 20121 Milano",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 4.5,
        "walk_to_attraction_min": 15,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "☢️ Area C 通行票 ¥5需购买。《最后的晚餐》位于圣玛利亚感恩教堂（Santa Maria delle Grazie），门票极度稀缺，必须提前 2-3 个月预约。",
    },

    # ── 巴塞罗那 ─────────────────────────────────────────
    "圣家堂": {
        "name": "Parking Saba Bams — Sagrada Familia",
        "address": "Carrer de Sardenya, 350, 08025 Barcelona",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 3.5,
        "walk_to_attraction_min": 3,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。圣家堂正对面，距离最近。⚠️ 巴塞罗那扒窃率极高，车内切勿遗留任何可见物品，建议司导轮流看车。",
    },
    "桂尔公园": {
        "name": "Parking BSM — Park Güell",
        "address": "Carrer d'Olot, 08024 Barcelona",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 3.0,
        "walk_to_attraction_min": 2,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。桂尔公园官方停车场，车位有限，旺季建议早到（9:00 前）。",
    },
    "巴特罗之家": {
        "name": "Parking Saba Bams — Passeig de Gràcia",
        "address": "Passeig de Gràcia, 62, 08007 Barcelona",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 3.8,
        "walk_to_attraction_min": 4,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。同在 Passeig de Gràcia 上的米拉之家（La Pedrera）步行 5 分钟可达，无需移车。",
    },
    "哥特区": {
        "name": "Parking BSM — Catedral",
        "address": "Avinguda de la Catedral, 08002 Barcelona",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 3.2,
        "walk_to_attraction_min": 2,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。位于巴塞罗那主教堂前广场地下。⚠️ 哥特区夜间部分巷道灯光昏暗，建议天黑前离场。",
    },

    # ── 阿姆斯特丹 ───────────────────────────────────────
    "梵高博物馆": {
        "name": "Q-Park Museumplein",
        "address": "Van Baerlestraat 33B, 1071 AP Amsterdam",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 5.5,
        "walk_to_attraction_min": 3,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。位于博物馆广场地下，步行至梵高博物馆、国立博物馆均5分钟内。阿姆斯特丹停车费极高（€5-7/h），属正常水平。",
    },
    "国立博物馆": {
        "name": "Q-Park Museumplein",
        "address": "Van Baerlestraat 33B, 1071 AP Amsterdam",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 5.5,
        "walk_to_attraction_min": 3,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。与梵高博物馆共用停车场，两个博物馆可一网打尽无需移车。",
    },
    "安妮之家": {
        "name": "Q-Park Bijenkorf",
        "address": "Damrak 70B, 1012 LM Amsterdam",
        "height_limit_m": 1.95,
        "hourly_rate_eur": 6.0,
        "walk_to_attraction_min": 10,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "⚠️ 限高1.95m刚好卡在 V-Class(1.92m) 边缘。建议司导现场确认高度后再入库。替代方案：Parking Oosterdok（限高2.1m），步行约15分钟。",
    },

    # ── 伦敦 ─────────────────────────────────────────────
    "大英博物馆": {
        "name": "NCP London Bloomsbury Square",
        "address": "Bloomsbury Square, London WC1A 2RJ",
        "height_limit_m": 1.98,
        "hourly_rate_eur": 8.0,
        "walk_to_attraction_min": 5,
        "large_vehicle_ok": False,
        "ztl_restricted": False,
        "notes": "⚠️ 伦敦 Congestion Charge Zone（拥堵费 £15/天）+ ULEZ（超低排放区 £12.50/天，V-Class 柴油版可能不达标）。限高1.98m 不足以容纳 V-Class(1.92m 刚好但极窄）。替代方案：在伦敦郊区换乘地铁，或选用本地司机。",
    },

    # ── 比萨 ────────────────────────────────────────────
    "比萨斜塔": {
        "name": "Parcheggio Via Pietrasantina",
        "address": "Via Pietrasantina, 56122 Pisa",
        "height_limit_m": None,
        "hourly_rate_eur": 2.0,
        "walk_to_attraction_min": 12,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "露天停车场，V-Class 无忧。斜塔所在的奇迹广场（Piazza dei Miracoli）外围最大停车场。🚫 奇迹广场全境步行区，车辆不得驶入。",
    },

    # ── 五渔村 ──────────────────────────────────────────
    "五渔村": {
        "name": "Parcheggio Stazione La Spezia Centrale",
        "address": "Piazza Medaglie d'Oro, 19122 La Spezia",
        "height_limit_m": 2.1,
        "hourly_rate_eur": 1.5,
        "walk_to_attraction_min": 5,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "🚫 **五渔村五村全境 ZTL 禁行**，外部车辆严禁驶入任何一个村子。最佳方案：V-Class 停 La Spezia 火车站停车场，全团乘火车（Cinque Terre Express，€5/单程）进入各村。买 Cinque Terre Card（€18.20/天，含火车+徒步步道）。",
    },

    # ── 圣吉米尼亚诺 ────────────────────────────────────
    "圣吉米尼亚诺": {
        "name": "Parcheggio Giubileo — San Gimignano",
        "address": "Via dei Fossi, 53037 San Gimignano",
        "height_limit_m": None,
        "hourly_rate_eur": 2.0,
        "walk_to_attraction_min": 8,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "🚫 圣吉米尼亚诺古城墙内全境 ZTL。此停车场位于城墙外南侧，V-Class 可停。步行经 Porta San Giovanni 城门进入古城。",
    },

    # ── 锡耶纳 ──────────────────────────────────────────
    "锡耶纳": {
        "name": "Parcheggio Stadio — Siena",
        "address": "Viale dei Mille, 53100 Siena",
        "height_limit_m": None,
        "hourly_rate_eur": 2.0,
        "walk_to_attraction_min": 15,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "🚫 锡耶纳历史中心全境 ZTL。此停车场位于城北，V-Class 可停。有免费自动扶梯（Escalator）直达市中心田野广场（Piazza del Campo）。",
    },

    # ── 庞贝 ────────────────────────────────────────────
    "庞贝古城": {
        "name": "Parcheggio Pompeii — Via Plinio",
        "address": "Via Plinio, 80045 Pompei",
        "height_limit_m": None,
        "hourly_rate_eur": 2.0,
        "walk_to_attraction_min": 3,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "露天停车场，V-Class 无忧。紧邻庞贝遗址 Porta Marina 主入口。旺季（6-9月）建议携带遮阳伞和水——遗址内几乎没有遮荫。",
    },

    # ── 尼斯 ────────────────────────────────────────────
    "蔚蓝海岸": {
        "name": "Parking Promenade des Anglais — Palais de la Méditerranée",
        "address": "15 Promenade des Anglais, 06000 Nice",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 3.5,
        "walk_to_attraction_min": 1,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。位于英国人漫步大道（Promenade des Anglais），海景停车场。夏季（7-8月）车位极度紧张，建议 9:00 前到达。",
    },

    # ── 马泰拉 ──────────────────────────────────────────
    "马泰拉": {
        "name": "Parcheggio Via Saragat — Matera",
        "address": "Via Giuseppe Saragat, 75100 Matera",
        "height_limit_m": None,
        "hourly_rate_eur": 2.0,
        "walk_to_attraction_min": 15,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "🚫 **马泰拉 Sassi 老城全境 ZTL 禁行区**（仅限居民车辆）。此停车场位于 Sassi 区外围，V-Class 可停。步行下坡进入 Sassi 石窟城区。建议穿防滑鞋——石窟区石板路极滑。",
    },

    # ── 阿尔贝罗贝洛 ────────────────────────────────────
    "阿尔贝罗贝洛": {
        "name": "Parcheggio Via Giuseppe Verdi",
        "address": "Via Giuseppe Verdi, 70011 Alberobello",
        "height_limit_m": None,
        "hourly_rate_eur": 1.5,
        "walk_to_attraction_min": 5,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "露天停车场，V-Class 无忧。步行 5 分钟进入 Trulli 蘑菇屋区。⚠️ Trulli 区内巷道极窄，V-Class 绝对无法进入。",
    },

    # ── 卢塞恩 ──────────────────────────────────────────
    "卢塞恩": {
        "name": "Parkhaus Altstadt — Luzern",
        "address": "Kasernenplatz, 6003 Luzern",
        "height_limit_m": 2.0,
        "hourly_rate_eur": 4.0,
        "walk_to_attraction_min": 8,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。位于卢塞恩老城入口。瑞士停车费偏高（CHF 3-5/h），此为正常水平。步行经卡佩尔廊桥（Chapel Bridge）进入老城。🇨🇭 瑞士高速需购买 Vignette（CHF 40/年）。",
    },

    # ── 因特拉肯 / 少女峰 ───────────────────────────────
    "少女峰": {
        "name": "Parking Grindelwald Terminal",
        "address": "Grundstrasse, 3818 Grindelwald",
        "height_limit_m": 2.2,
        "hourly_rate_eur": 3.0,
        "walk_to_attraction_min": 2,
        "large_vehicle_ok": True,
        "ztl_restricted": False,
        "notes": "V-Class 可入。车辆停在 Grindelwald Terminal，全团乘 Eiger Express 缆车（15分钟）至 Eigergletscher，换乘少女峰铁路（Jungfraubahn）登顶。无法开车上山。少女峰往返票 CHF 220+/人，建议提前在线购票。",
    },
}

# 反向别名索引：支持 DeepSeek 输出的各种中文名变体
_PARKING_ALIASES: dict[str, str] = {
    "巴黎圣母院内部入内导览": "巴黎圣母院",
    "巴黎圣母院内部": "巴黎圣母院",
    "Notre-Dame": "巴黎圣母院",
    "卢浮宫博物馆": "卢浮宫",
    "Louvre": "卢浮宫",
    "艾菲尔铁塔": "埃菲尔铁塔",
    "Tour Eiffel": "埃菲尔铁塔",
    "罗马斗兽场": "斗兽场",
    "Colosseum": "斗兽场",
    "Colosseo": "斗兽场",
    "Vatican Museums": "梵蒂冈博物馆",
    "Musei Vaticani": "梵蒂冈博物馆",
    "Trevi Fountain": "许愿池",
    "Fontana di Trevi": "许愿池",
    "Pantheon": "万神殿",
    "Spanish Steps": "西班牙广场",
    "Piazza di Spagna": "西班牙广场",
    "圣母百花圣殿": "圣母百花大教堂",
    "Duomo di Firenze": "圣母百花大教堂",
    "Duomo": "圣母百花大教堂",
    "Uffizi": "乌菲兹美术馆",
    "Galleria degli Uffizi": "乌菲兹美术馆",
    "大卫像": "大卫雕像",
    "学院美术馆": "大卫雕像",
    "Galleria dell'Accademia": "大卫雕像",
    "维琪奥桥": "老桥",
    "Ponte Vecchio": "老桥",
    "Piazza San Marco": "圣马可广场",
    "Bridge of Sighs": "叹息桥",
    "Ponte dei Sospiri": "叹息桥",
    "Rialto": "里亚托桥",
    "Ponte di Rialto": "里亚托桥",
    "Duomo di Milano": "米兰大教堂",
    "Milan Cathedral": "米兰大教堂",
    "Cenacolo Vinciano": "最后的晚餐",
    "The Last Supper": "最后的晚餐",
    "Sagrada Familia": "圣家堂",
    "Park Güell": "桂尔公园",
    "Casa Batlló": "巴特罗之家",
    "Gothic Quarter": "哥特区",
    "Barri Gòtic": "哥特区",
    "Van Gogh Museum": "梵高博物馆",
    "Rijksmuseum": "国立博物馆",
    "Anne Frank House": "安妮之家",
    "British Museum": "大英博物馆",
    "Torre di Pisa": "比萨斜塔",
    "Leaning Tower": "比萨斜塔",
    "Cinque Terre": "五渔村",
    "San Gimignano": "圣吉米尼亚诺",
    "Siena": "锡耶纳",
    "Pompeii": "庞贝古城",
    "庞贝": "庞贝古城",
    "Nice": "蔚蓝海岸",
    "Promenade des Anglais": "蔚蓝海岸",
    "Matera": "马泰拉",
    "Sassi di Matera": "马泰拉",
    "Alberobello": "阿尔贝罗贝洛",
    "Trulli": "阿尔贝罗贝洛",
    "Luzern": "卢塞恩",
    "Lucerne": "卢塞恩",
    "Jungfrau": "少女峰",
    "Jungfraujoch": "少女峰",
}


def _resolve_parking_key(stop_name: str) -> str | None:
    """Resolve a stop name to its canonical STATIC_DATA_LIBRARY key.

    匹配顺序：
    1. 精确匹配库 key
    2. 别名匹配
    3. 子串包含匹配（库 key 出现在 stop_name 中）
    """
    if not stop_name:
        return None
    # 1. 精确匹配
    if stop_name in STATIC_DATA_LIBRARY:
        return stop_name
    # 2. 别名匹配
    if stop_name in _PARKING_ALIASES:
        return _PARKING_ALIASES[stop_name]
    # 3. 子串包含（库 key ⊂ stop_name）
    for key in STATIC_DATA_LIBRARY:
        if key in stop_name:
            return key
    return None


def _enrich_parking_from_library(itinerary: list[dict]) -> None:
    """从静态数据库注入停车场数据，免去 DeepSeek 生成开销。

    对每个 stop，若匹配到 STATIC_DATA_LIBRARY，则将其完整停车场数据
    注入 driver_parking 字典（覆盖 `_static` 占位符）。不匹配的 stop 保留
    DeepSeek 原始输出不变。遍历结束后清理所有未被替换的 `_static` 残留。
    """
    for day in itinerary:
        stops = day.get("stops", [])
        if not stops:
            continue

        raw_parking = day.get("driver_parking")
        if not isinstance(raw_parking, dict):
            raw_parking = {}
            day["driver_parking"] = raw_parking

        # Pass 1: inject library data, tracking old keys to remove
        stale_keys: set[str] = set()
        added_keys: set[str] = set()
        for stop in stops:
            stop_name = stop.get("name", "")
            lib_key = _resolve_parking_key(stop_name)
            if lib_key is None:
                continue
            # Find any existing driver_parking key that matches this stop
            for pk in list(raw_parking.keys()):
                if pk == lib_key:
                    break
                if isinstance(raw_parking[pk], dict) and raw_parking[pk].get("_static") is True:
                    stale_keys.add(pk)
                    break
            raw_parking[lib_key] = STATIC_DATA_LIBRARY[lib_key]
            added_keys.add(lib_key)

        # Pass 2: remove stale keys (old key names replaced by canonical lib key)
        for sk in stale_keys:
            if sk not in added_keys:
                raw_parking.pop(sk, None)

        # Pass 3: remove any residual _static entries that weren't matched
        for pk in list(raw_parking.keys()):
            v = raw_parking[pk]
            if isinstance(v, dict) and v.get("_static") is True:
                del raw_parking[pk]


# ===========================================================================
# Lifespan — PostgreSQL 异步连接池生命周期（Render 冷启动容错）
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 非阻塞连接：Supavisor session pooler 冷启动可能 >15s，
    # 若阻塞则触发 Render free-tier Startup Timeout → 应用被 SIGKILL
    try:
        await asyncio.wait_for(database.connect(), timeout=10.0)
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning(
            "Database pool warmup deferred — %s. "
            "First DB-bound request will trigger lazy connect.",
            exc.__class__.__name__,
        )
    else:
        try:
            await database.fetch_val("SELECT 1")
            logger.info(
                "PostgreSQL pool ready — %s",
                DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else "connected",
            )
        except Exception:
            logger.warning("Initial health ping failed — pool may still be warming; retrying on first request")

    yield

    # 关闭阶段：优雅释放连接池
    try:
        await asyncio.wait_for(database.disconnect(), timeout=5.0)
    except Exception:
        logger.warning("Database disconnect timed out — connections will be cleaned up by OS")
    else:
        logger.info("PostgreSQL pool released")


# ===========================================================================
# FastAPI application
# ===========================================================================
app = FastAPI(
    title="欧洲司导行程引擎",
    description="WeChat → DeepSeek → 分享链接 · 全自动异步流水线",
    version="3.1.0",
    lifespan=lifespan,
)


# ===========================================================================
# Data models
# ===========================================================================
class ItineraryRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$",
        description="导游唯一标识，如 guide_01",
    )
    raw_text: str = Field(
        ..., min_length=1, max_length=10000,
        description="微信聊天混乱文本",
    )
    security_mode: Literal["standard", "concierge"] = Field(
        default="standard",
        description="standard: 不提安保; concierge: 时装周特护影子安保协议",
    )
    booking_aid: str | None = Field(
        default=None, min_length=1, max_length=32,
        description="导游个人 Booking 联盟 ID，为空则降级为平台默认 ID",
    )
    guide_name: str | None = Field(
        default=None, max_length=32,
        description="司导姓名，用于白标页面顶部和尾部渲染",
    )
    guide_wechat: str | None = Field(
        default=None, max_length=64,
        description="司导微信号，用于白标页面联系方式展示",
    )


class GenerateResponse(BaseModel):
    status: str
    share_url: str


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str


# ===========================================================================
# POST /api/v1/itinerary/generate
# ===========================================================================
@app.post(
    "/api/v1/itinerary/generate",
    response_model=GenerateResponse,
    responses={500: {"model": ErrorResponse}},
    summary="生成行程并返回分享链接",
)
async def generate_itinerary(req: ItineraryRequest, request: Request):
    logger.info("Generate request received: user_id=%s mode=%s text_len=%d",
                req.user_id, req.security_mode, len(req.raw_text))

    await ensure_db_connected()

    system_prompt = get_system_prompt(req.security_mode)

    # --- DeepSeek API 调用 ---
    try:
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.3,
            max_tokens=8192,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.raw_text},
            ],
        )
        logger.info("DeepSeek response received: model=%s usage=%s",
                    response.model, getattr(response, "usage", "N/A"))
    except RateLimitError as exc:
        logger.error("DeepSeek rate limit exceeded: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="DeepSeek API 频率限制，请稍后重试。",
        )
    except APIConnectionError as exc:
        logger.error("DeepSeek connection error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="无法连接 DeepSeek API，请检查网络或 API 端点配置。",
        )
    except APIError as exc:
        logger.error("DeepSeek API error: status=%s message=%s",
                    getattr(exc, "status_code", "?"), exc)
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek API 返回错误 (status={getattr(exc, 'status_code', '?')})。",
        )
    except Exception as exc:
        logger.error("Unexpected DeepSeek error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"DeepSeek API 调用失败: {exc}",
        )

    # --- JSON 解析 ---
    content = response.choices[0].message.content
    if not content:
        logger.error("DeepSeek returned empty content (finish_reason=%s)",
                     response.choices[0].finish_reason)
        raise HTTPException(
            status_code=500,
            detail="DeepSeek 返回空内容，可能是输入过长或模型拒绝生成。请精简聊天记录后重试。",
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error(
            "JSON parse failed at line %d col %d — raw preview (first 500 chars): %s",
            exc.lineno, exc.colno, content[:500],
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"DeepSeek 返回的内容不是合法 JSON：{exc.msg} "
                f"(line {exc.lineno}, col {exc.colno})。"
                f"原始返回前 500 字符: {content[:500]}"
            ),
        )

    # --- 验证关键字段 ---
    itinerary = data.get("itinerary")
    if not itinerary or not isinstance(itinerary, list):
        logger.error("JSON missing 'itinerary' array — keys: %s", list(data.keys())[:10])
        raise HTTPException(
            status_code=500,
            detail="行程 JSON 缺少必需的 'itinerary' 字段，请检查 System Prompt 或聊天记录输入。",
        )

    # --- 字段归一化：driver_parking / master_schedule ---
    # DeepSeek 可能返回 None / [] / 非标准结构，此处强制归一化为安全默认值，
    # 确保下游 _prepare_context() 和 Jinja2 模板不会因类型不匹配而崩溃。
    for day in itinerary:
        if not isinstance(day.get("driver_parking"), dict):
            day["driver_parking"] = {}
        else:
            day["driver_parking"] = {
                str(k): v for k, v in day["driver_parking"].items()
                if isinstance(v, dict)
            }

    if not isinstance(data.get("master_schedule"), list):
        data["master_schedule"] = []
    else:
        data["master_schedule"] = [
            s for s in data["master_schedule"] if isinstance(s, dict)
        ]

    # --- 静态数据库注入：免 Token 停车场数据 ---
    _enrich_parking_from_library(itinerary)

    # --- 将导游信息嵌入 JSON，统一落盘 ---
    structured_final = json.loads(content)
    structured_final["_guide_name"] = req.guide_name
    structured_final["_guide_wechat"] = req.guide_wechat

    # --- PostgreSQL 写入 + 生成分享链接 ---
    itinerary_id = str(uuid.uuid4())

    insert_query = """
    INSERT INTO itineraries (id, user_id, security_mode, booking_aid, structured_json)
    VALUES (:id, :user_id, :security_mode, :booking_aid, :structured_json)
    """
    await database.execute(
        query=insert_query,
        values={
            "id": itinerary_id,
            "user_id": req.user_id,
            "security_mode": req.security_mode,
            "booking_aid": req.booking_aid,
            "structured_json": json.dumps(structured_final, ensure_ascii=False),
        },
    )

    # Lightweight verification: SELECT the row we just inserted
    verify = await database.fetch_val(
        "SELECT id FROM itineraries WHERE id = :id",
        values={"id": itinerary_id},
    )
    if verify is None:
        logger.error("INSERT verification failed: id=%s not found after write", itinerary_id)
        raise HTTPException(status_code=500, detail="Database write verification failed.")
    logger.info("Itinerary stored: id=%s user_id=%s days=%d cities=%s",
                itinerary_id, req.user_id,
                len(itinerary),
                data.get("summary", {}).get("cities", []))

    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "127.0.0.1:8000")
    proto = request.headers.get("x-forwarded-proto", "https")
    share_url = f"{proto}://{host}/share/{itinerary_id}"
    logger.info("Generate complete: id=%s share_url=%s", itinerary_id, share_url)

    return GenerateResponse(status="success", share_url=share_url)


# ===========================================================================
# GET /share/{itinerary_id} — 公开分享页面
# ===========================================================================
ERROR_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>渲染错误</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-[#FAF6F1] flex items-center justify-center min-h-screen">
<div class="text-center px-4">
  <p class="text-6xl mb-4"></p>
  <h1 class="font-serif text-2xl font-bold text-[#2C2416] mb-2">页面渲染失败</h1>
  <p class="text-[#5C4B3A] text-sm">服务暂时不可用，请稍后重试或联系司导。</p>
</div>
</body>
</html>"""


@app.get(
    "/share/{itinerary_id}",
    response_class=HTMLResponse,
    summary="公开分享页面",
)
async def share_itinerary(request: Request, itinerary_id: str):
    logger.info("Share request received: id=%s", itinerary_id)

    await ensure_db_connected()

    query = "SELECT * FROM itineraries WHERE id = :id"
    record = await database.fetch_one(query=query, values={"id": itinerary_id})

    if record is None:
        logger.warning("Share 404: id=%s not found", itinerary_id)
        raise HTTPException(status_code=404, detail="行程未找到")

    # 从 JSONB 中析出导游元信息
    structured = record["structured_json"]
    if isinstance(structured, str):
        structured = json.loads(structured)

    guide_name = structured.pop("_guide_name", None)
    guide_wechat = structured.pop("_guide_wechat", None)

    # 构建模板上下文
    try:
        context = _prepare_context(
            structured,
            user_id=record["user_id"],
            guide_name=guide_name,
            guide_wechat=guide_wechat,
            booking_aid=record["booking_aid"],
            security_mode=record["security_mode"],
        )
    except Exception as exc:
        logger.error("Context preparation failed: id=%s error=%s\n%s",
                     itinerary_id, exc, traceback.format_exc())
        return HTMLResponse(content=ERROR_HTML, status_code=500)

    # 抓取每日景点代表性图片（Wikipedia API，并发非阻塞）
    try:
        day_images = await _fetch_day_images(context["days"])
        context["day_images"] = day_images
    except Exception as exc:
        logger.warning("Image fetch skipped for id=%s: %s", itinerary_id, exc)
        context["day_images"] = {}

    # 报价计算
    try:
        quote = calculate_quote(context["days"])
        low = int(quote["grand_total_eur"] * 1.1)
        high = int(quote["grand_total_eur"] * 1.2)
        context["price_range"] = f"€{low:,}–€{high:,}"
    except Exception as exc:
        logger.warning("Quote calculation skipped for id=%s: %s", itinerary_id, exc)
        context["price_range"] = None

    # Jinja2 实时渲染
    try:
        template = _jinja_env.get_template("itinerary.html")
        html = template.render(request=request, **context)
        logger.info("Share rendered: id=%s html_size=%d", itinerary_id, len(html))
        return HTMLResponse(content=html)
    except TemplateNotFound:
        logger.critical("Template 'itinerary.html' not found in %s", TEMPLATES_DIR)
        return HTMLResponse(content=ERROR_HTML, status_code=500)
    except TemplateError as exc:
        logger.error("Template render error: id=%s error=%s\n%s",
                     itinerary_id, exc, traceback.format_exc())
        return HTMLResponse(content=ERROR_HTML, status_code=500)
    except Exception as exc:
        logger.error("Unexpected render error: id=%s error=%s\n%s",
                     itinerary_id, exc, traceback.format_exc())
        return HTMLResponse(content=ERROR_HTML, status_code=500)


# ===========================================================================
# Root / Health
# ===========================================================================
@app.get("/", response_class=HTMLResponse)
async def guide_entrance_page(request: Request):
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tour Workbench</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap">
    <style>
        body { font-family: -apple-system, "Microsoft YaHei", "Segoe UI", Helvetica, Arial, sans-serif; background-color: #f6f8fa; color: #24292f; }
        .code-font { font-family: 'Fira Code', monospace; }
        .github-border { border-color: #d0d7de; }
        .github-bg-card { background-color: #ffffff; }
        .github-input { background-color: #f6f8fa; color: #24292f; }
        .github-input:focus { background-color: #ffffff; border-color: #0969da; outline: none; box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.15); }
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between antialiased">

    <main class="max-w-3xl w-full mx-auto px-4 py-8 sm:py-12 flex-1">
        <div class="github-bg-card border github-border rounded-xl p-6 sm:p-8 shadow-sm">

            <div class="border-b github-border pb-4 mb-6">
                <h2 class="text-xl font-bold text-[#1f2328]">欧洲司导行程单生成系统 <span class="text-sm font-normal text-[#57606a]">· Itinerary Generator</span></h2>
                <p class="text-sm text-[#57606a] mt-1.5">微信聊天记录 → AI 结构化编译 → 白标 H5 分享页 · Chat logs → AI compile → white-label share page</p>
            </div>

            <form id="orderForm" class="space-y-6">

                <div class="space-y-2.5">
                    <label class="block text-sm font-bold text-[#1f2328]">导游信息 (Guide Metadata)</label>
                    <input type="text" id="userId" required
                           placeholder="例如：向导Alex (微信: alex_paris / 电话: +33 6 123456)"
                           class="w-full border github-border rounded-lg px-3.5 py-3 text-base github-input transition-all">
                    <p class="text-xs text-[#57606a]">这些信息将作为动态白标直接注入到给客户浏览的网页最上方。</p>
                </div>

                <input type="hidden" id="bookingAid" value="default_platform_partner">

                <input type="hidden" id="securityMode" value="standard">

                <div class="space-y-2.5">
                    <label class="block text-sm font-bold text-[#1f2328]">客户原始意向数据 / Raw Client Inquiry</label>
                    <textarea id="rawText" required rows="7"
                              placeholder="直接将微信聊天记录或零碎想法粘贴至此。例如：5月24号去巴黎，想看巴黎圣母院内部入内讲解，下午去看埃菲尔铁塔。推荐周边步行可达、Google Map好评最高的法餐厅..."
                              class="w-full border github-border rounded-lg p-3.5 text-base github-input transition-all leading-relaxed code-font resize-none"></textarea>
                </div>

                <div class="pt-2">
                    <button type="submit" class="w-full sm:w-auto px-6 py-2.5 bg-[#2da44e] hover:bg-[#2c974b] text-white font-semibold text-sm rounded-lg border border-[rgba(27,31,36,0.15)] transition-all cursor-pointer shadow-sm">
                        生成行程单 / Generate
                    </button>
                </div>
            </form>

            <div id="resultBoard" class="mt-8 border github-border rounded-xl hidden bg-[#f6f8fa]">
                <div class="bg-[#ffffff] border-b github-border px-4 py-2.5 flex items-center justify-between rounded-t-xl">
                    <span class="text-xs font-mono text-[#57606a]">Console / stdout</span>
                    <span class="text-xs bg-[#dafbe1] text-[#1a7f37] border border-[rgba(26,127,55,0.2)] px-2.5 py-0.5 rounded-full font-bold">Success</span>
                </div>
                <div class="p-4 space-y-4">
                    <div class="p-4 bg-[#ffffff] border github-border rounded-lg relative group">
                        <span class="text-xs text-[#57606a] uppercase font-mono font-bold block">CLIENT SHARE URL:</span>
                        <p id="shareUrl" class="text-sm font-mono text-[#0969da] break-all font-bold mt-2 pr-20 select-all"></p>
                        <button onclick="copyUrl()" class="absolute right-3 top-4 text-xs bg-[#f6f8fa] hover:bg-[#eaeef2] border github-border text-[#24292f] px-3 py-1.5 rounded-md cursor-pointer transition-colors font-medium">Copy</button>
                    </div>
                    <div class="flex">
                        <a id="previewBtn" target="_blank" class="text-center px-5 py-2 bg-[#ffffff] hover:bg-[#f6f8fa] border github-border text-[#24292f] text-sm font-medium rounded-md transition-colors shadow-sm cursor-pointer">
                            View Rendered HTML
                        </a>
                    </div>
                </div>
            </div>

        </div>
    </main>

    <footer class="border-t github-border bg-[#eaeef2]/30 px-4 py-6 text-center text-xs text-[#57606a] space-y-1">
        <p class="code-font text-[11px]">Commit hash: d381505 · Environment: production</p>
        <p>&copy; 2026 EuroTourHub, Inc. Built for distributed tour guides.</p>
    </footer>

    <script>
        document.getElementById('orderForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = e.target.querySelector('button[type="submit"]');
            btn.disabled = true;
            btn.className = "w-full sm:w-auto px-6 py-2.5 bg-[#eaeef2] text-[#57606a] border github-border text-sm rounded-lg cursor-not-allowed font-mono animate-pulse";
            btn.innerText = "AI 编译中 / Compiling...";

            const payload = {
                user_id: document.getElementById('userId').value.trim(),
                raw_text: document.getElementById('rawText').value.trim(),
                security_mode: document.getElementById('securityMode').value,
                booking_aid: document.getElementById('bookingAid').value.trim()
            };

            try {
                const response = await fetch('/api/v1/itinerary/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error('Build error: Gateway responded with status ' + response.status);
                const data = await response.json();

                if (data.share_url) {
                    document.getElementById('shareUrl').innerText = data.share_url;
                    document.getElementById('previewBtn').href = data.share_url;
                    document.getElementById('resultBoard').classList.remove('hidden');
                    window.open(data.share_url, '_blank');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.className = "w-full sm:w-auto px-6 py-2.5 bg-[#2da44e] hover:bg-[#2c974b] text-white font-semibold text-sm rounded-lg border border-[rgba(27,31,36,0.15)] transition-all cursor-pointer shadow-sm";
                btn.innerText = "生成行程单 / Generate";
            }
        });

        function copyUrl() {
            const url = document.getElementById('shareUrl').innerText;
            navigator.clipboard.writeText(url);
            alert('Copied to clipboard!');
        }
    </script>
</body>
</html>""")


@app.get("/health")
async def health():
    try:
        await ensure_db_connected()
        count = await database.fetch_val("SELECT COUNT(*) FROM itineraries")
    except Exception:
        count = -1
    return {
        "status": "ok",
        "service": "itinerary-engine",
        "storage": "postgresql",
        "stored": count,
    }


# ===========================================================================
# Entrypoint
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
