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
        days.append({**d, "stops": stops})

    cities = summary.get("cities", [])
    title = " → ".join(cities) if cities else "欧洲定制行程"

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
    version="3.0.0",
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

    # --- 将导游信息嵌入 JSON，统一落盘 ---
    structured_final = json.loads(content)
    structured_final["_guide_name"] = req.guide_name
    structured_final["_guide_wechat"] = req.guide_wechat

    # --- PostgreSQL 写入 + 生成分享链接 ---
    itinerary_id = str(uuid.uuid4())

    query = """
    INSERT INTO itineraries (id, user_id, security_mode, booking_aid, structured_json)
    VALUES (:id, :user_id, :security_mode, :booking_aid, :structured_json)
    """
    await database.execute(
        query=query,
        values={
            "id": itinerary_id,
            "user_id": req.user_id,
            "security_mode": req.security_mode,
            "booking_aid": req.booking_aid,
            "structured_json": json.dumps(structured_final, ensure_ascii=False),
        },
    )

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
    <title>欧洲全景司导云端开单工作台</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>
        body { font-family: 'Inter', system-ui, sans-serif; }
        .glow-btn:hover { box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }
    </style>
</head>
<body class="bg-neutral-950 text-neutral-100 min-h-screen flex flex-col justify-between pb-8">

    <header class="max-w-xl w-full mx-auto px-6 pt-12 text-center space-y-2">
        <div class="inline-flex items-center space-x-2 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span class="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">B2B2C Cloud Workbench v3.5</span>
        </div>
        <h1 class="text-xl font-black tracking-tight text-neutral-50">欧洲向导极速开单系统</h1>
        <p class="text-xs text-neutral-400">粘贴散客原始意向大白话，AI 自动清洗并注入高分 Google Map 餐饮底座</p>
    </header>

    <main class="max-w-xl w-full mx-auto px-6 my-auto pt-6">
        <form id="orderForm" class="bg-neutral-900 border border-neutral-800/80 rounded-2xl p-6 space-y-5 shadow-2xl">

            <div class="grid grid-cols-2 gap-4">
                <div class="space-y-1.5">
                    <label class="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">向导工号 (User ID)</label>
                    <input type="text" id="userId" required placeholder="如: Paris_Alex_666"
                           class="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-3 py-2.5 text-xs text-neutral-200 focus:outline-none focus:border-emerald-500 transition-colors">
                </div>
                <div class="space-y-1.5">
                    <label class="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">分销追踪码 (Booking AID)</label>
                    <input type="text" id="bookingAid" required placeholder="如: alex_vip_partner"
                           class="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-3 py-2.5 text-xs text-neutral-200 focus:outline-none focus:border-emerald-500 transition-colors">
                </div>
            </div>

            <div class="space-y-1.5">
                <label class="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">客户托管安保级别</label>
                <select id="securityMode" class="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-3 py-2.5 text-xs text-neutral-200 focus:outline-none focus:border-emerald-500 transition-colors">
                    <option value="standard">STANDARD (标准白标分流模式)</option>
                    <option value="concierge">CONCIERGE (高净值黑金隐私拦截模式)</option>
                </select>
            </div>

            <div class="space-y-1.5">
                <label class="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">散客原始行程意向 (微信记录/草稿)</label>
                <textarea id="rawText" required rows="5" placeholder="例如: 5月24号去巴黎，想看巴黎圣母院内部入内讲解，下午去看埃菲尔铁塔。顺便推荐附近Google Map好评最高的餐厅..."
                          class="w-full bg-neutral-950 border border-neutral-800 rounded-xl p-3 text-xs text-neutral-200 focus:outline-none focus:border-emerald-500 transition-colors leading-relaxed resize-none"></textarea>
            </div>

            <button type="submit" class="w-full glow-btn bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-bold text-xs tracking-widest py-3 rounded-xl transition-all duration-300 transform active:scale-[0.98] cursor-pointer">
                ⚡ 自动化编译并落盘生成专属行程
            </button>
        </form>

        <div id="resultBoard" class="mt-6 bg-neutral-900 border border-emerald-500/30 rounded-2xl p-5 space-y-4 hidden">
            <div class="flex items-center space-x-2 text-emerald-400">
                <span class="text-xs font-bold">🎉 行程编译成功！数据已永固落盘</span>
            </div>
            <div class="p-3 bg-neutral-950 rounded-xl border border-neutral-800 relative group">
                <p class="text-[9px] text-neutral-500 uppercase tracking-wider">散客专属动态渲染分销链接</p>
                <p id="shareUrl" class="text-xs font-mono text-emerald-400 break-all font-bold mt-1 pr-12 select-all"></p>
                <button onclick="copyUrl()" class="absolute right-3 top-4 text-[10px] bg-neutral-800 hover:bg-neutral-700 text-neutral-300 px-2 py-1 rounded cursor-pointer">复制</button>
            </div>
            <div class="flex gap-3">
                <a id="previewBtn" target="_blank" class="flex-1 text-center py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium rounded-xl transition-colors">
                    👀 司导真机预览
                </a>
            </div>
        </div>
    </main>

    <footer class="text-center text-[10px] text-neutral-600 tracking-wider">
        Powered by DeepSeek Engine & Supabase Storage Pooler. All rights reserved.
    </footer>

    <script>
        document.getElementById('orderForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = e.target.querySelector('button[type="submit"]');
            btn.disabled = true;
            btn.innerText = "⏳ 正在调集 DEEPSEEK 编译并铺设 GOOGLE MAP 高分餐厅...";

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

                if (!response.ok) throw new Error('网关响应异常');
                const data = await response.json();

                if (data.share_url) {
                    document.getElementById('shareUrl').innerText = data.share_url;
                    document.getElementById('previewBtn').href = data.share_url;
                    document.getElementById('resultBoard').classList.remove('hidden');
                    document.getElementById('resultBoard').scrollIntoView({ behavior: 'smooth' });
                }
            } catch (err) {
                alert('开单失败，请检查网络或后端环境: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerText = "⚡ 自动化编译并落盘生成专属行程";
            }
        });

        function copyUrl() {
            const url = document.getElementById('shareUrl').innerText;
            navigator.clipboard.writeText(url);
            alert('链接已成功复制到剪贴板，可直接发给散客！');
        }
    </script>
</body>
</html>""")


@app.get("/health")
async def health():
    try:
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
