"""
app.py — 欧洲司导行程全自动 Web 工作台
Streamlit: 微信聊天 → DeepSeek → JSON → HTML → Vercel 上线
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from system_prompt import get_system_prompt

# ---------------------------------------------------------------------------
# 路径 & 环境
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUTPUT_JSON = HERE / "output_itinerary.json"
OUTPUT_HTML = HERE / "index.html"
BOOKING_AFFILIATE = "https://www.booking.com/index.html?aid=YOUR_AFFILIATE_ID"

load_dotenv(HERE / ".env", override=True)
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# HTML 模板（内联自 render_html.py）
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>{title}</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
<script>
  tailwind.config = {{
    theme: {{
      extend: {{
        fontFamily: {{
          serif: ['"Playfair Display"', '"Noto Serif SC"', "serif"],
          sans: ['"Noto Sans SC"', "system-ui", "sans-serif"],
        }},
        colors: {{
          cream: "#FAF6F1",
          gold: "#C9A96E",
          ink: "#2C2416",
          warm: "#5C4B3A",
          blush: "#E8D5C4",
          alert: "#B91C1C",
          amber: "#D97706",
        }},
      }},
    }},
  }}
</script>
<style>
  .text-balance {{ text-wrap: balance; }}
  details[open] summary .chevron {{ transform: rotate(180deg); }}
  details summary::-webkit-details-marker {{ display: none; }}
  details summary::marker {{ display: none; content: ""; }}
  .glass-lock {{
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
  }}
  .card-shadow {{
    box-shadow: 0 1px 3px rgba(44,36,22,0.06), 0 4px 16px rgba(44,36,22,0.04);
  }}
  .hero-gradient {{
    background: linear-gradient(135deg, #2C2416 0%, #5C4B3A 40%, #3D3226 100%);
  }}
</style>
</head>
<body class="bg-cream font-sans text-ink antialiased">

<header class="hero-gradient text-white px-5 pt-12 pb-10 rounded-b-[2rem]">
  <p class="text-xs tracking-[0.25em] uppercase text-gold/80 mb-2 font-medium">Itineraire sur Mesure</p>
  <h1 class="font-serif text-3xl leading-tight font-semibold tracking-tight text-balance">
    {title}
  </h1>
  <div class="flex flex-wrap gap-2 mt-4">
    {city_tags}
  </div>
  <div class="flex gap-5 mt-5 text-sm text-white/70">
    <span>{total_days} 天行程</span>
    <span>{budget}</span>
  </div>
</header>

<main class="px-4 pb-20 -mt-4 relative z-10">

{security_alert}

{warning_alert}

{day_cards}

<section class="mt-8 mb-12">
  <div class="bg-white/60 backdrop-blur rounded-2xl px-5 py-5 card-shadow border border-blush/50">
    <div class="flex items-start gap-3">
      <span class="text-xl mt-0.5">📋</span>
      <div>
        <h3 class="font-serif text-base font-semibold text-ink mb-2">酒店免责声明</h3>
        <p class="text-sm text-warm leading-relaxed">
          {hotel_disclaimer}
        </p>
      </div>
    </div>
  </div>
</section>

<footer class="text-center text-xs text-warm/50 pb-8">
  <p>Genere par l'IA · Itineraire Premium</p>
  <p class="mt-1">Pre-book to unlock full roadbook with GPS & restaurant links</p>
</footer>

</main>
</body>
</html>"""

SECURITY_ALERT_TMPL = """<section class="mb-5 -mt-2">
  <div class="bg-alert/5 border border-alert/20 rounded-2xl p-4 card-shadow
              bg-gradient-to-br from-red-50 to-amber-50/50">
    <div class="flex items-start gap-3">
      <span class="text-2xl">🛡️</span>
      <div class="flex-1 min-w-0">
        <h2 class="font-serif text-base font-semibold text-alert mb-2">
          时装周级别 · 影子安保红线
        </h2>
        <p class="text-sm text-warm leading-relaxed whitespace-pre-line">
          {content}
        </p>
      </div>
    </div>
  </div>
</section>"""

CONCIERGE_SECURITY_TMPL = """<section class="mb-5 -mt-2">
  <div class="rounded-2xl p-5 card-shadow overflow-hidden
              bg-gradient-to-br from-[#1a1a2e] via-[#2C2416] to-[#0f0f0f]
              border border-gold/40">
    <div class="flex items-center gap-2 mb-3">
      <span class="text-xl">👑</span>
      <h2 class="font-serif text-base font-semibold text-gold tracking-wide">
        全包定制 · 尊贵加项
      </h2>
      <span class="ml-auto bg-gold/20 text-gold text-[10px] px-2 py-0.5 rounded-full
                   tracking-widest uppercase font-medium">
        Concierge
      </span>
    </div>
    <div class="flex items-start gap-3">
      <span class="text-2xl mt-0.5">🛡️</span>
      <div>
        <h3 class="font-serif text-sm font-semibold text-white/90 mb-2">
          时装周级别 · 影子安保协议
        </h3>
        <p class="text-sm text-white/60 leading-relaxed whitespace-pre-line">
          {content}
        </p>
        <div class="mt-3 pt-3 border-t border-white/10 flex items-center gap-2
                    text-[10px] text-gold/60 uppercase tracking-widest">
          <span>24/7 紧急响应</span>
          <span class="text-white/20">·</span>
          <span>实时位置追踪</span>
          <span class="text-white/20">·</span>
          <span>使馆直通热线</span>
        </div>
      </div>
    </div>
  </div>
</section>"""

WARNING_ALERT_TMPL = """<section class="mb-5">
  <div class="bg-amber-50 border border-amber-200 rounded-2xl p-4 card-shadow">
    <div class="flex items-start gap-3">
      <span class="text-2xl">⚠️</span>
      <div class="flex-1 min-w-0">
        <h2 class="font-serif text-base font-semibold text-amber/90 mb-2">
          行程超限警告
        </h2>
        <p class="text-sm text-warm leading-relaxed">{content}</p>
      </div>
    </div>
  </div>
</section>"""

DAY_CARD_TMPL = """<section class="mb-5">
  <div class="bg-white rounded-2xl card-shadow overflow-hidden border border-blush/30">
    <div class="flex items-center justify-between px-5 py-4 border-b border-blush/20
                bg-gradient-to-r from-white to-cream">
      <div class="flex items-center gap-3">
        <span class="font-serif text-2xl font-bold text-gold">{day}</span>
        <div>
          <p class="text-xs text-warm/60">{date}</p>
          <p class="font-serif text-lg font-semibold text-ink">{city}</p>
        </div>
      </div>
      <div class="text-right text-xs text-warm/50">
        <p>驾驶 {driving_hours}h</p>
        <p>工时 {work_hours}h</p>
      </div>
    </div>
    <div class="divide-y divide-blush/10">
      {stops_html}
    </div>
    <div class="relative mx-4 my-4 rounded-xl overflow-hidden border border-blush/20
                bg-gradient-to-br from-cream to-white">
      <div class="p-4 opacity-40 select-none pointer-events-none">
        <h4 class="font-serif text-sm font-semibold text-ink mb-2">
          🚗 精确行车路线 & GPS 坐标
        </h4>
        <p class="text-xs text-warm leading-relaxed">
          含高速出口编号、加油站定位、免费停车点、收费站避让路线。
        </p>
        <h4 class="font-serif text-sm font-semibold text-ink mt-3 mb-2">
          🍽️ 避坑餐厅 & 订位链接
        </h4>
        <p class="text-xs text-warm leading-relaxed">
          {dining_hint}
        </p>
      </div>
      <div class="absolute inset-0 glass-lock bg-white/60 flex flex-col items-center
                  justify-center gap-2 rounded-xl">
        <span class="text-2xl">🔒</span>
        <p class="font-serif text-sm font-semibold text-ink text-center px-4">
          预定后司导即送全套保姆级避坑路书
        </p>
        <p class="text-xs text-warm/70 text-center px-6">
          含精确 GPS 坐标 · 餐厅订位链接 · 高速避费路线
        </p>
      </div>
    </div>
    <div class="px-5 pb-4">
      <div class="flex items-center justify-between">
        <div>
          <p class="text-xs text-warm/50 uppercase tracking-wide">住宿推荐区域</p>
          <p class="text-sm font-medium text-ink mt-0.5">{hotel_zone}</p>
          <p class="text-xs text-warm/60 mt-0.5">{hotel_features}</p>
        </div>
        <a href="{booking_url}"
           class="flex-shrink-0 inline-flex items-center gap-1.5
                  bg-blue-600 hover:bg-blue-700 active:scale-95 transition-all
                  text-white text-xs font-medium px-4 py-2.5 rounded-full">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M3 12l2-2m0 0l7-7 7 7m-9 2v10m4-10v10"/>
          </svg>
          查看酒店
        </a>
      </div>
    </div>
  </div>
</section>"""

STOP_HTML = """              <div class="px-5 py-3.5 flex items-start gap-3">
                <div class="w-1.5 h-1.5 rounded-full {dot_color} mt-2 flex-shrink-0"></div>
                <div class="flex-1 min-w-0">
                  <div class="flex items-center justify-between gap-2">
                    <h5 class="text-sm font-medium text-ink truncate">{name}</h5>
                    <span class="text-xs text-warm/40 flex-shrink-0">{time_range}</span>
                  </div>
                  <details class="mt-1.5 group">
                    <summary class="flex items-center gap-1 text-xs text-gold cursor-pointer
                                    hover:text-gold/80 transition-colors">
                      <span>📖 司导讲解</span>
                      <svg class="w-3 h-3 chevron transition-transform duration-200"
                           fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round"
                              stroke-width="2" d="M19 9l-7 7-7-7"/>
                      </svg>
                    </summary>
                    <p class="text-xs text-warm leading-relaxed mt-2 pl-0.5">
                      {notes}
                    </p>
                  </details>
                </div>
              </div>"""


# ---------------------------------------------------------------------------
# HTML 构建函数
# ---------------------------------------------------------------------------
def _stop_type_dot(stop_type: str) -> str:
    colors = {
        "attraction": "bg-gold",
        "restaurant": "bg-red-400",
        "shopping": "bg-purple-400",
        "other": "bg-gray-400",
    }
    return colors.get(stop_type, "bg-gold")


def _fmt_time_range(arrival: str, departure: str) -> str:
    return f"{arrival} – {departure}"


def build_day_card(day_data: dict) -> str:
    day = day_data.get("day", "?")
    date = day_data.get("date", "")
    city = day_data.get("city", "")
    driving_hours = day_data.get("driving_hours", 0)
    work_hours = day_data.get("total_work_hours", 0)

    stops = day_data.get("stops", [])
    stops_parts = []
    for s in stops:
        stops_parts.append(STOP_HTML.format(
            name=s.get("name", ""),
            time_range=_fmt_time_range(s.get("arrival", ""), s.get("departure", "")),
            notes=s.get("notes", "暂无讲解"),
            dot_color=_stop_type_dot(s.get("type", "attraction")),
        ))

    meals = day_data.get("meals", [])
    dining_hint = "、".join(meals) if meals else "包含司导私藏餐厅列表及预订链接"

    hotel_zone = day_data.get("hotel_zone", "")
    hotel_features = " · ".join(day_data.get("hotel_features", []))
    if hotel_features:
        hotel_features = f"设施需求：{hotel_features}"

    return DAY_CARD_TMPL.format(
        day=day,
        date=date,
        city=city,
        driving_hours=driving_hours,
        work_hours=work_hours,
        stops_html="\n".join(stops_parts),
        dining_hint=dining_hint,
        hotel_zone=hotel_zone,
        hotel_features=hotel_features,
        booking_url=BOOKING_AFFILIATE,
    )


def build_security_alert(security_text: str | None, security_mode: str = "standard") -> str:
    if not security_text:
        return ""
    if security_mode == "concierge":
        return CONCIERGE_SECURITY_TMPL.format(content=security_text)
    # 标准安全模式不渲染安保卡片（即使用了旧版 SYSTEM_PROMPT 也隐藏）
    return ""


def build_warning_alert(warning_text: str | None) -> str:
    if not warning_text:
        return ""
    return WARNING_ALERT_TMPL.format(content=warning_text)


def build_city_tags(cities: list[str]) -> str:
    tags = []
    for c in cities:
        tags.append(
            f'<span class="bg-white/15 backdrop-blur text-xs px-3 py-1 rounded-full '
            f'border border-white/10">{c}</span>'
        )
    return "\n    ".join(tags) if tags else ""


def build_title(meta: dict) -> str:
    summary = meta.get("summary", {})
    cities = summary.get("cities", [])
    if cities:
        return " → ".join(cities)
    return "欧洲定制行程"


def render_html(data: dict, security_mode: str = "standard") -> str:
    itinerary = data.get("itinerary", [])
    summary = data.get("summary", {})

    security_alert = build_security_alert(data.get("security_risk"), security_mode)
    warning_alert = build_warning_alert(data.get("warning"))
    day_cards = "\n".join(build_day_card(d) for d in itinerary)
    city_tags = build_city_tags(summary.get("cities", []))
    title = build_title(data)
    total_days = summary.get("total_days", len(itinerary))
    budget = summary.get("budget_range", "详询司导")
    hotel_disclaimer = data.get("hotel_disclaimer", "")

    return HTML_TEMPLATE.format(
        title=title,
        city_tags=city_tags,
        total_days=total_days,
        budget=budget,
        security_alert=security_alert,
        warning_alert=warning_alert,
        day_cards=day_cards,
        hotel_disclaimer=hotel_disclaimer,
    )


# ---------------------------------------------------------------------------
# DeepSeek API 调用
# ---------------------------------------------------------------------------
def call_deepseek(raw_text: str, mode: str, security_mode: str = "standard") -> dict | None:
    """调用 DeepSeek API 生成行程 JSON，失败返回 None。"""
    if not DEEPSEEK_API_KEY:
        st.error("未设置 DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY")
        return None

    # 行程偏好注入
    mode_modifiers = {
        "标准行程模式": "",
        "深度慢游模式": "🚶 行程偏好：深度慢游。减少每日景点数量至2-3个，增加单个景点的停留时长，优先选择小众深度体验而非热门打卡点。节奏放慢，留出充足自由探索时间。",
        "网红打卡模式": "📸 行程偏好：网红打卡模式。每日安排4-5个高颜值出片点，优先选择社交媒体热门地点，标注最佳拍照时间和角度。节奏紧凑，覆盖尽可能多的标志性场景。",
    }
    modifier = mode_modifiers.get(mode, "")
    user_message = raw_text + "\n\n" + modifier if modifier else raw_text

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE)
    system_prompt = get_system_prompt(security_mode)

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            temperature=0.3,
            max_tokens=8192,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:
        st.error(f"API 调用失败：{exc}")
        return None

    content = response.choices[0].message.content
    if not content:
        st.error("API 返回空内容")
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        st.error(f"JSON 解析失败：{exc}")
        st.code(content[:1000], language="json")
        return None


# ---------------------------------------------------------------------------
# Vercel 全自动部署
# ---------------------------------------------------------------------------
def deploy_to_vercel() -> str | None:
    """静默运行 vercel --prod --yes，返回线上 URL。"""
    try:
        result = subprocess.run(
            "vercel --prod --yes",
            capture_output=True,
            text=True,
            timeout=180,
            shell=True,
            cwd=str(HERE),
        )
    except FileNotFoundError:
        st.error("未找到 vercel 命令，请先执行 `npm install -g vercel`")
        return None
    except subprocess.TimeoutExpired:
        st.error("Vercel 部署超时（>3分钟），请检查网络或手动部署")
        return None

    combined = result.stdout + "\n" + result.stderr

    # 优先取 Aliased URL（短域名），其次取 Production URL
    aliased = re.search(r"Aliased\s+(https://[\w.-]+\.vercel\.app)", combined)
    if aliased:
        return aliased.group(1)

    production = re.search(r"Production\s+(https://[\w.-]+\.vercel\.app)", combined)
    if production:
        return production.group(1)

    st.error(f"未能从 Vercel 输出中解析到线上地址")
    with st.expander("部署输出详情"):
        st.code(combined[-2000:], language="text")
    return None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="欧洲司导行程工作台",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 注入暗色科技感 CSS
st.markdown("""
<style>
    /* 全局暗色基调 */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #16213e 100%);
    }
    /* 侧边隐藏 */
    [data-testid="stSidebar"] { display: none; }
    /* 主内容区 */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    /* 大按钮 */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #C9A96E 0%, #e6c98e 100%);
        color: #1a1a2e;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 0.75rem;
        transition: all 0.2s;
        letter-spacing: 0.05em;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(201,169,110,0.3);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    /* 文本输入框 */
    textarea, .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e0e0e0 !important;
        border-radius: 0.75rem !important;
        font-size: 0.95rem !important;
    }
    /* Select 下拉 */
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 0.75rem !important;
    }
    /* 成功消息 */
    .stAlert[data-baseweb="notification"] {
        border-radius: 0.75rem;
    }
    /* 标签 */
    label, .stMarkdown, .stCaption {
        color: #c0c0c0 !important;
    }
    h1, h2, h3 {
        color: #C9A96E !important;
    }
    /* 代码块 */
    .stCodeBlock {
        background: rgba(0,0,0,0.3) !important;
        border: 1px solid rgba(201,169,110,0.2) !important;
        border-radius: 0.75rem !important;
    }
    /* 输入框 label */
    .stTextArea label, .stSelectbox label {
        color: #a0a0a0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    /* 分割线 */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- 标题 ----
st.markdown("""
<h1 style="font-size:2rem; margin-bottom:0.25rem;">
    <span style="color:#C9A96E;">🧳</span> 欧洲司导行程工作台
</h1>
<p style="color:#888; font-size:0.85rem; margin-top:0;">
    WeChat → DeepSeek → HTML → Vercel · 全自动流水线
</p>
""", unsafe_allow_html=True)

st.divider()

# ---- 双栏布局 ----
left, right = st.columns([0.45, 0.55], gap="large")

# ========================= 左栏：输入端 =========================
with left:
    st.markdown("### 📥 输入区")

    raw_text = st.text_area(
        "粘贴微信聊天记录",
        height=240,
        placeholder="小李啊，我们一共5个人，7月中旬打算去欧洲。巴黎肯定要去，我老婆要看铁塔...",
        label_visibility="visible",
    )

    mode = st.selectbox(
        "行程风格",
        ["标准行程模式", "深度慢游模式", "网红打卡模式"],
        index=0,
    )

    security_mode = st.selectbox(
        "安全级别",
        ["标准安全", "时装周特护模式"],
        index=0,
        help="标准安全：不提安保内容。时装周特护：DeepSeek 注入影子安保协议，H5 渲染为全包定制尊贵加项卡片。",
    )
    # 转为内部 key
    security_key = "concierge" if security_mode == "时装周特护模式" else "standard"

    st.markdown("<br>", unsafe_allow_html=True)

    go_btn = st.button("⚡ 智能洗稿并全自动发布", type="primary")

    # 会话状态初始化
    if "vercel_url" not in st.session_state:
        st.session_state.vercel_url = None
    if "html_content" not in st.session_state:
        st.session_state.html_content = None
    if "itinerary_data" not in st.session_state:
        st.session_state.itinerary_data = None
    if "last_raw_text" not in st.session_state:
        st.session_state.last_raw_text = ""
    if "last_security_mode" not in st.session_state:
        st.session_state.last_security_mode = "standard"

# ========================= 右栏：输出端 =========================
with right:
    st.markdown("### 📤 发布结果")

    url_placeholder = st.empty()
    copy_placeholder = st.empty()
    preview_placeholder = st.empty()

    # 已发布过的结果直接展示
    if st.session_state.vercel_url:
        url_placeholder.markdown(f"""
        <div style="background:rgba(201,169,110,0.1); border:1px solid rgba(201,169,110,0.3);
                    border-radius:0.75rem; padding:1rem; text-align:center;">
            <p style="color:#888; font-size:0.8rem; margin:0 0 0.25rem 0;">🔗 线上地址</p>
            <a href="{st.session_state.vercel_url}" target="_blank"
               style="color:#C9A96E; font-size:1.2rem; font-weight:700;
                      text-decoration:none; word-break:break-all;">
                {st.session_state.vercel_url}
            </a>
        </div>
        """, unsafe_allow_html=True)

        # 一键复制
        copy_placeholder.markdown(f"""
        <button onclick="navigator.clipboard.writeText('{st.session_state.vercel_url}')"
                style="width:100%; background:rgba(255,255,255,0.06); color:#c0c0c0;
                       border:1px solid rgba(255,255,255,0.1); border-radius:0.5rem;
                       padding:0.5rem; cursor:pointer; font-size:0.85rem; margin:0.5rem 0;">
            📋 一键复制链接
        </button>
        """, unsafe_allow_html=True)

        # 内嵌 H5 预览
        if st.session_state.html_content:
            with preview_placeholder.expander("📱 内嵌 H5 预览", expanded=False):
                st.components.v1.html(
                    st.session_state.html_content,
                    height=700,
                    scrolling=True,
                )
    else:
        url_placeholder.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                    border-radius:0.75rem; padding:2rem; text-align:center; color:#666;">
            <p style="font-size:1.5rem; margin:0;">⏳</p>
            <p style="margin:0.5rem 0 0 0; font-size:0.9rem;">等待输入并执行...</p>
        </div>
        """, unsafe_allow_html=True)


# ========================= 按钮逻辑 =========================
if go_btn:
    if not raw_text.strip():
        st.error("请先粘贴微信聊天记录")
    else:
        with st.status("⚡ 全自动流水线运行中...", expanded=True) as status:
            # Step 1: DeepSeek 生成行程 JSON
            status.update(label="📡 Step 1/4: 调用 DeepSeek API 生成行程...")
            data = call_deepseek(raw_text.strip(), mode, security_key)
            if data is None:
                status.update(label="❌ 失败于 Step 1", state="error")
                st.stop()

            # 持久化 JSON
            OUTPUT_JSON.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary = data.get("summary", {})
            days = summary.get("total_days", len(data.get("itinerary", [])))
            cities = " → ".join(summary.get("cities", []))
            status.update(label=f"✅ Step 1/4 完成: {days}天行程 · {cities}")

            # Step 2: 渲染 HTML
            status.update(label="🎨 Step 2/4: 渲染 H5 网页...")
            html = render_html(data, security_key)
            OUTPUT_HTML.write_text(html, encoding="utf-8")
            st.session_state.html_content = html
            st.session_state.itinerary_data = data
            st.session_state.last_raw_text = raw_text.strip()
            st.session_state.last_security_mode = security_key
            status.update(label=f"✅ Step 2/4 完成: index.html 已生成 ({len(html):,} 字符)")

            # Step 3: Vercel 部署
            status.update(label="🚀 Step 3/4: 部署至 Vercel...")
            url = deploy_to_vercel()
            if url is None:
                status.update(label="⚠️ Step 3/4: Vercel 部署失败，但本地文件已就绪", state="error")
                st.session_state.vercel_url = None
            else:
                st.session_state.vercel_url = url
                status.update(label=f"🌐 Step 3/4 完成: {url}")

            # Step 4: 完成
            status.update(label="🎉 Step 4/4: 全流程完成！", state="complete")

        # 刷新右侧面板
        st.rerun()
