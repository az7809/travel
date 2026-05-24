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
                <h2 class="text-xl font-bold text-[#1f2328]">Create New Base Itinerary</h2>
                <p class="text-sm text-[#57606a] mt-1.5">通过精简的指令自动化渲染云原生分布式客户行程单</p>
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
                    <label class="block text-sm font-bold text-[#1f2328]">散客原始意向数据 (Raw Prompt Data)</label>
                    <textarea id="rawText" required rows="7"
                              placeholder="直接将微信聊天记录或零碎想法粘贴至此。例如：5月24号去巴黎，想看巴黎圣母院内部入内讲解，下午去看埃菲尔铁塔。推荐周边步行可达、Google Map好评最高的法餐厅..."
                              class="w-full border github-border rounded-lg p-3.5 text-base github-input transition-all leading-relaxed code-font resize-none"></textarea>
                </div>

                <div class="pt-2">
                    <button type="submit" class="w-full sm:w-auto px-6 py-2.5 bg-[#2da44e] hover:bg-[#2c974b] text-white font-semibold text-sm rounded-lg border border-[rgba(27,31,36,0.15)] transition-all cursor-pointer shadow-sm">
                        Run Build Script
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
            btn.innerText = "Executing deepseek-compiler...";

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
                    document.getElementById('resultBoard').scrollIntoView({ behavior: 'smooth' });
                }
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.className = "w-full sm:w-auto px-6 py-2.5 bg-[#2da44e] hover:bg-[#2c974b] text-white font-semibold text-sm rounded-lg border border-[rgba(27,31,36,0.15)] transition-all cursor-pointer shadow-sm";
                btn.innerText = "Run Build Script";
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
