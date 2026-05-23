"""
Integration smoke test against production Render deployment.
"""
import json
import sys
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "https://travel-gqru.onrender.com"
TIMEOUT = 120  # Render free tier cold-start can take 30-60s

OK = 0
FAIL = 0


def check(step: str, condition: bool, detail: str = ""):
    global OK, FAIL
    if condition:
        OK += 1
        print(f"  PASS  {step}")
    else:
        FAIL += 1
        print(f"  FAIL  {step}  —  {detail}")
    return condition


# ===== 1. Health =====
print("=== 1. GET /health ===")
try:
    r = requests.get(f"{BASE}/health", timeout=TIMEOUT)
    health = r.json()
    ok = check("status=200", r.status_code == 200, f"got {r.status_code}")
    if ok:
        check("storage=postgresql", health.get("storage") == "postgresql", str(health))
        check("stored >= 0", isinstance(health.get("stored"), int), str(health))
        print(f"       DB records: {health.get('stored')}  storage: {health.get('storage')}")
except Exception as e:
    check("health request", False, str(e))

# ===== 2. Generate itinerary =====
print("\n=== 2. POST /api/v1/itinerary/generate ===")
payload = {
    "user_id": "ci_test_007",
    "raw_text": (
        "5天意大利深度游，罗马进米兰出。"
        "必须打卡：斗兽场、梵蒂冈博物馆、许愿池、万神殿。"
        "第三天去佛罗伦萨看大卫雕像和圣母百花大教堂，乌菲兹美术馆。"
        "第四天威尼斯圣马可广场、叹息桥、坐贡多拉。"
        "第五天米兰大教堂、最后的晚餐。"
        "全程4星酒店，预算中高档，每天一顿米其林推荐餐厅。"
    ),
    "security_mode": "standard",
    "guide_name": "Marco",
    "guide_wechat": "marco_italia",
    "booking_aid": "MARCO_AID_2026",
}

try:
    r = requests.post(
        f"{BASE}/api/v1/itinerary/generate",
        json=payload,
        timeout=TIMEOUT,
    )
    gen = r.json()
    ok = check("status=200", r.status_code == 200, f"got {r.status_code}: {gen}")
    if ok:
        share_url = gen.get("share_url", "")
        check("has share_url", bool(share_url), str(gen))
        check("status=success", gen.get("status") == "success", str(gen))
        print(f"       share_url: {share_url}")
        itinerary_id = share_url.rsplit("/", 1)[-1] if share_url else None
except Exception as e:
    check("generate request", False, str(e))
    share_url = None
    itinerary_id = None

# ===== 3. Share page =====
if itinerary_id:
    print(f"\n=== 3. GET {share_url} ===")
    try:
        r = requests.get(share_url, timeout=30)
        html = r.text
        ok = check("status=200", r.status_code == 200, f"got {r.status_code}")
        if ok:
            check("HTML contains Marco", "Marco" in html)
            check("HTML contains 意大利", "意大利" in html or "罗马" in html)
            check("HTML contains Booking CTA", "booking.com" in html.lower())
            check("HTML contains title tag", "<title>" in html)
            print(f"       HTML size: {len(html)} bytes")
    except Exception as e:
        check("share request", False, str(e))

# ===== 4. 404 for bogus UUID =====
print("\n=== 4. GET /share/00000000-0000-0000-0000-000000000000 ===")
try:
    r = requests.get(
        f"{BASE}/share/00000000-0000-0000-0000-000000000000",
        timeout=30,
    )
    check("status=404", r.status_code == 404, f"got {r.status_code}")
except Exception as e:
    check("404 request", False, str(e))

# ===== Summary =====
print(f"\n{'='*40}")
total = OK + FAIL
print(f"Results: {OK}/{total} passed, {FAIL}/{total} failed")
sys.exit(0 if FAIL == 0 else 1)
