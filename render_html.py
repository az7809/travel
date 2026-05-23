"""
render_html.py — output_itinerary.json → 精美移动端 H5 网页

自动读取行程 JSON，渲染为小红书/微信适配的欧洲极简摩登 H5。
Tailwind CSS CDN，零依赖，一键输出 index.html。
"""

import json
import sys
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
INPUT_JSON = HERE / "output_itinerary.json"
OUTPUT_HTML = HERE / "index.html"

# Booking 分销链接（替换 YOUR_AFFILIATE_ID 即可上线）
BOOKING_AFFILIATE = "https://www.booking.com/index.html?aid=YOUR_AFFILIATE_ID"


# ---------------------------------------------------------------------------
# HTML 模板
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

<!-- ====== Hero Header ====== -->
<header class="hero-gradient text-white px-5 pt-12 pb-10 rounded-b-[2rem]">
  <p class="text-xs tracking-[0.25em] uppercase text-gold/80 mb-2 font-medium">Itinéraire sur Mesure</p>
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

<!-- ====== Security Alert ====== -->
{security_alert}

<!-- ====== Warning ====== -->
{warning_alert}

<!-- ====== Day Cards ====== -->
{day_cards}

<!-- ====== Hotel Disclaimer ====== -->
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

<!-- ====== Footer ====== -->
<footer class="text-center text-xs text-warm/50 pb-8">
  <p>Généré par l'IA · Itinéraire Premium</p>
  <p class="mt-1">Pre-book to unlock full roadbook with GPS & restaurant links</p>
</footer>

</main>
</body>
</html>"""

# ---------------------------------------------------------------------------
# 子模板片段
# ---------------------------------------------------------------------------

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

DAY_CARD_TMPL = """<!-- Day {day}: {city} -->
<section class="mb-5">
  <div class="bg-white rounded-2xl card-shadow overflow-hidden border border-blush/30">
    <!-- Day header -->
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

    <!-- Stops -->
    <div class="divide-y divide-blush/10">
      {stops_html}
    </div>

    <!-- Locked: Driving routes & insider tips -->
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
      <!-- Frosted glass overlay -->
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

    <!-- Hotel + Booking -->
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
# 构建函数
# ---------------------------------------------------------------------------

def _stop_type_dot(stop_type: str) -> str:
    """根据停留类型返回不同颜色的标记点."""
    colors = {
        "attraction": "bg-gold",
        "restaurant": "bg-red-400",
        "shopping": "bg-purple-400",
        "other": "bg-gray-400",
    }
    return colors.get(stop_type, "bg-gold")


def _fmt_time_range(arrival: str, departure: str) -> str:
    return f"{arrival} – {departure}"


def build_day_card(day_data: dict, booking_url: str | None = None) -> str:
    """渲染单天行程卡片 HTML."""
    day = day_data.get("day", "?")
    date = day_data.get("date", "")
    city = day_data.get("city", "")
    driving_hours = day_data.get("driving_hours", 0)
    work_hours = day_data.get("total_work_hours", 0)

    # 停留点
    stops = day_data.get("stops", [])
    stops_parts = []
    for s in stops:
        stops_parts.append(STOP_HTML.format(
            name=s.get("name", ""),
            time_range=_fmt_time_range(s.get("arrival", ""), s.get("departure", "")),
            notes=s.get("notes", "暂无讲解"),
            dot_color=_stop_type_dot(s.get("type", "attraction")),
        ))

    # 餐厅线索（用于模糊遮罩下露出一点内容）
    meals = day_data.get("meals", [])
    dining_hint = "、".join(meals) if meals else "包含司导私藏餐厅列表及预订链接"

    # 酒店
    hotel_zone = day_data.get("hotel_zone", "")
    hotel_features = " · ".join(day_data.get("hotel_features", []))
    if hotel_features:
        hotel_features = f"设施需求：{hotel_features}"

    url = booking_url or BOOKING_AFFILIATE
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
        booking_url=url,
    )


def build_security_alert(security_text: str | None, security_mode: str = "standard") -> str:
    if not security_text:
        return ""
    if security_mode == "concierge":
        return CONCIERGE_SECURITY_TMPL.format(content=security_text)
    return SECURITY_ALERT_TMPL.format(content=security_text)


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
    """从 JSON 中提取合适的页面标题."""
    summary = meta.get("summary", {})
    cities = summary.get("cities", [])
    if cities:
        return " → ".join(cities)
    return "欧洲定制行程"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def render_html(data: dict, booking_aid: str | None = None, security_mode: str = "standard") -> str:
    """渲染完整 H5 HTML，返回字符串。可传入自定义 Booking affiliate ID 和安全模式。"""
    itinerary = data.get("itinerary", [])
    summary = data.get("summary", {})

    booking_url = f"https://www.booking.com/index.html?aid={booking_aid}" if booking_aid else BOOKING_AFFILIATE

    security_alert = build_security_alert(data.get("security_risk"), security_mode)
    warning_alert = build_warning_alert(data.get("warning"))

    day_cards = "\n".join(build_day_card(d, booking_url) for d in itinerary)

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


def main() -> int:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not INPUT_JSON.exists():
        print(f"错误：找不到 {INPUT_JSON}", file=sys.stderr)
        print("请先运行 generate_itinerary.py 生成行程 JSON。", file=sys.stderr)
        return 1

    try:
        data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"错误：JSON 解析失败 — {exc}", file=sys.stderr)
        return 1

    # 提取数据
    itinerary = data.get("itinerary", [])
    summary = data.get("summary", {})

    # 构建各区域
    security_alert = build_security_alert(data.get("security_risk"))
    warning_alert = build_warning_alert(data.get("warning"))

    day_cards = "\n".join(build_day_card(d) for d in itinerary)

    city_tags = build_city_tags(summary.get("cities", []))
    title = build_title(data)
    total_days = summary.get("total_days", len(itinerary))
    budget = summary.get("budget_range", "详询司导")
    hotel_disclaimer = data.get("hotel_disclaimer", "")

    # 渲染完整 HTML
    html = HTML_TEMPLATE.format(
        title=title,
        city_tags=city_tags,
        total_days=total_days,
        budget=budget,
        security_alert=security_alert,
        warning_alert=warning_alert,
        day_cards=day_cards,
        hotel_disclaimer=hotel_disclaimer,
    )

    # 写入文件
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ 已生成：{OUTPUT_HTML}")
    print(f"   行程 {total_days} 天 · {len(itinerary)} 张日卡片")

    if security_alert:
        print("   🛡️  安保红线卡片已注入")
    if warning_alert:
        print("   ⚠️  超限警告卡片已注入")

    return 0


if __name__ == "__main__":
    sys.exit(main())
