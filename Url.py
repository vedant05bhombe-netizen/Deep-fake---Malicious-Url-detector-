from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncio
import os
from urllib.parse import urlparse
import base64
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_SAFE_BROWSING_KEY = os.getenv("GOOGLE_API_KEY", "AQ.Ab8RN6IqePfU-JYBNOaWXPQHVP1xeKPgC_VgaTe9CSLxrTXSHQ")
VIRUSTOTAL_KEY           = os.getenv("VIRUSTOTAL_API_KEY", "d8ef969fdc7e512560f533305ffef870c3daaba39bd8247fb0256d5c89d0705e")
URLSCAN_KEY              = os.getenv("URLSCAN_API_KEY", "019c7e41-8553-7515-8ed4-2aec091658dc")

# ──────────────────────────────────────────────
# TRUSTED DOMAIN WHITELIST
# These return SAFE instantly — no API calls made
# ──────────────────────────────────────────────
TRUSTED_DOMAINS = {
    # Search
    "google.com", "www.google.com", "google.co.in",
    "bing.com", "www.bing.com",
    "duckduckgo.com", "www.duckduckgo.com",
    "yahoo.com", "www.yahoo.com",
    # Social
    "facebook.com", "www.facebook.com", "m.facebook.com",
    "instagram.com", "www.instagram.com",
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
    "linkedin.com", "www.linkedin.com",
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "pinterest.com", "www.pinterest.com",
    "tiktok.com", "www.tiktok.com",
    "snapchat.com", "www.snapchat.com",
    "threads.net", "www.threads.net",
    # Messaging
    "web.whatsapp.com", "whatsapp.com",
    "web.telegram.org", "telegram.org",
    "discord.com", "www.discord.com", "canary.discord.com",
    "slack.com", "app.slack.com",
    "signal.org", "www.signal.org",
    # Google Services
    "gmail.com", "mail.google.com",
    "drive.google.com", "docs.google.com", "sheets.google.com",
    "slides.google.com", "forms.google.com", "meet.google.com",
    "calendar.google.com", "maps.google.com", "photos.google.com",
    "play.google.com", "accounts.google.com", "myaccount.google.com",
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
    # Microsoft
    "microsoft.com", "www.microsoft.com",
    "office.com", "www.office.com",
    "outlook.com", "www.outlook.com", "outlook.live.com",
    "live.com", "hotmail.com",
    "github.com", "www.github.com", "gist.github.com",
    "azure.microsoft.com", "portal.azure.com",
    "onedrive.live.com", "teams.microsoft.com",
    # Apple
    "apple.com", "www.apple.com",
    "icloud.com", "www.icloud.com",
    "store.apple.com", "support.apple.com",
    # Amazon
    "amazon.com", "www.amazon.com", "amazon.in", "www.amazon.in",
    "aws.amazon.com", "console.aws.amazon.com",
    "prime.amazon.com",
    # Shopping
    "flipkart.com", "www.flipkart.com",
    "ebay.com", "www.ebay.com",
    "etsy.com", "www.etsy.com",
    "shopify.com", "www.shopify.com",
    "meesho.com", "www.meesho.com",
    "myntra.com", "www.myntra.com",
    "ajio.com", "www.ajio.com",
    # Payments
    "paypal.com", "www.paypal.com",
    "stripe.com", "www.stripe.com", "dashboard.stripe.com",
    "razorpay.com", "www.razorpay.com",
    "paytm.com", "www.paytm.com",
    "phonepe.com", "www.phonepe.com",
    "pay.google.com",
    # Streaming
    "netflix.com", "www.netflix.com",
    "spotify.com", "www.spotify.com", "open.spotify.com",
    "primevideo.com", "www.primevideo.com",
    "hotstar.com", "www.hotstar.com",
    "twitch.tv", "www.twitch.tv",
    "crunchyroll.com", "www.crunchyroll.com",
    "disneyplus.com", "www.disneyplus.com",
    # Dev / Tech
    "stackoverflow.com", "www.stackoverflow.com",
    "stackexchange.com",
    "npmjs.com", "pypi.org",
    "gitlab.com", "bitbucket.org",
    "vercel.com", "netlify.com", "heroku.com", "render.com",
    "cloudflare.com", "digitalocean.com",
    "replit.com", "codesandbox.io", "codepen.io",
    "figma.com", "notion.so", "postman.com",
    # News
    "bbc.com", "www.bbc.com", "bbc.co.uk",
    "cnn.com", "www.cnn.com",
    "nytimes.com", "theguardian.com", "reuters.com",
    "ndtv.com", "www.ndtv.com",
    "timesofindia.com", "thehindu.com",
    "hindustantimes.com", "indianexpress.com",
    # Education
    "wikipedia.org", "en.wikipedia.org",
    "khanacademy.org", "coursera.org",
    "udemy.com", "edx.org",
    "medium.com", "substack.com", "quora.com",
    # AI
    "openai.com", "chat.openai.com",
    "claude.ai", "www.claude.ai",
    "anthropic.com", "www.anthropic.com",
    "gemini.google.com",
    "huggingface.co",
    # Banking India
    "sbi.co.in", "onlinesbi.sbi.co.in",
    "hdfcbank.com", "netbanking.hdfcbank.com",
    "icicibank.com", "axisbank.com",
    "kotak.com",
    # Misc
    "zoom.us", "dropbox.com",
    "canva.com", "wordpress.com",
    "wix.com", "typeform.com",
    "airtable.com", "trello.com",
    "asana.com", "hubspot.com",
    "salesforce.com",
}


class URLRequest(BaseModel):
    url: str


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return url


def vt_url_id(url: str) -> str:
    return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


def is_trusted(domain: str) -> bool:
    if domain in TRUSTED_DOMAINS:
        return True
    # Also match subdomains e.g. news.bbc.com → bbc.com
    parts = domain.split(".")
    for i in range(1, len(parts)):
        if ".".join(parts[i:]) in TRUSTED_DOMAINS:
            return True
    return False


# ──────────────────────────────────────────────
# Scanner 1 — Google Safe Browsing
# ──────────────────────────────────────────────
async def scan_google(url: str, client: httpx.AsyncClient) -> dict:
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_SAFE_BROWSING_KEY}"
    payload = {
        "client": {"clientId": "sentinel-url-detector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        res = await client.post(endpoint, json=payload, timeout=8)
        data = res.json()
        matches = data.get("matches", [])
        if matches:
            threat_types = list({m.get("threatType", "UNKNOWN") for m in matches})
            return {"source": "Google Safe Browsing", "safe": False, "threat_types": threat_types, "details": f"Matched {len(matches)} threat(s): {', '.join(threat_types)}"}
        return {"source": "Google Safe Browsing", "safe": True, "threat_types": [], "details": "No threats found"}
    except Exception as e:
        return {"source": "Google Safe Browsing", "safe": None, "error": str(e)}


# ──────────────────────────────────────────────
# Scanner 2 — VirusTotal (lookup only, no sleep)
# ──────────────────────────────────────────────
async def scan_virustotal(url: str, client: httpx.AsyncClient) -> dict:
    headers = {"x-apikey": VIRUSTOTAL_KEY}
    url_id = vt_url_id(url)
    try:
        res = await client.get(f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers, timeout=5)
        if res.status_code == 404:
            await client.post("https://www.virustotal.com/api/v3/urls", headers=headers, data={"url": url}, timeout=5)
            return {"source": "VirusTotal", "safe": None, "details": "Submitted — no cached result yet"}
        data = res.json()
        attrs = data.get("data", {}).get("attributes", {})
        stats = attrs.get("last_analysis_stats", {})
        malicious  = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        total      = sum(stats.values()) if stats else 0
        return {
            "source": "VirusTotal",
            "safe": malicious == 0 and suspicious == 0,
            "malicious": malicious,
            "suspicious": suspicious,
            "total_engines": total,
            "details": f"{malicious} malicious, {suspicious} suspicious out of {total} engines",
        }
    except Exception as e:
        return {"source": "VirusTotal", "safe": None, "error": str(e)}


# ──────────────────────────────────────────────
# Scanner 3 — URLScan fast (cached search only)
# ──────────────────────────────────────────────
async def scan_urlscan_fast(url: str, client: httpx.AsyncClient) -> dict:
    headers = {"API-Key": URLSCAN_KEY}
    try:
        res = await client.get(f"https://urlscan.io/api/v1/search/?q=page.url:{url}&size=1", headers=headers, timeout=3)
        data = res.json()
        results = data.get("results", [])
        if results:
            verdict = results[0].get("verdicts", {}).get("overall", {})
            malicious = verdict.get("malicious", False)
            score = verdict.get("score", 0)
            return {"source": "URLScan.io", "safe": not malicious, "score": score, "details": f"Cached · Score: {score}/100"}
        return {"source": "URLScan.io", "safe": None, "details": "No cached result"}
    except Exception as e:
        return {"source": "URLScan.io", "safe": None, "error": str(e)}


# ──────────────────────────────────────────────
# Scanner 3 — URLScan full (submit + poll)
# ──────────────────────────────────────────────
async def scan_urlscan_full(url: str, client: httpx.AsyncClient) -> dict:
    headers = {"API-Key": URLSCAN_KEY, "Content-Type": "application/json"}
    try:
        submit = await client.post("https://urlscan.io/api/v1/scan/", headers=headers, json={"url": url, "visibility": "public"}, timeout=10)
        if submit.status_code not in (200, 201):
            return {"source": "URLScan.io", "safe": None, "error": f"Submission failed: {submit.status_code}"}
        result_url = submit.json().get("api")
        if not result_url:
            return {"source": "URLScan.io", "safe": None, "error": "No result URL"}
        for _ in range(3):
            await asyncio.sleep(6)
            poll = await client.get(result_url, timeout=10)
            if poll.status_code == 200:
                data = poll.json()
                verdicts = data.get("verdicts", {}).get("overall", {})
                malicious = verdicts.get("malicious", False)
                score = verdicts.get("score", 0)
                tags = verdicts.get("tags", [])
                screenshot = data.get("task", {}).get("screenshotURL", "")
                return {"source": "URLScan.io", "safe": not malicious, "score": score, "tags": tags, "screenshot": screenshot, "details": f"Score: {score}/100 · Tags: {', '.join(tags) if tags else 'none'}"}
        return {"source": "URLScan.io", "safe": None, "error": "Scan timed out"}
    except Exception as e:
        return {"source": "URLScan.io", "safe": None, "error": str(e)}


# ──────────────────────────────────────────────
# Aggregate verdict
# ──────────────────────────────────────────────
def aggregate_verdict(results: list) -> dict:
    danger_count  = sum(1 for r in results if r.get("safe") is False)
    unknown_count = sum(1 for r in results if r.get("safe") is None)
    safe_count    = sum(1 for r in results if r.get("safe") is True)

    if danger_count >= 2:
        verdict, risk = "DANGEROUS", "HIGH"
    elif danger_count == 1:
        verdict, risk = "SUSPICIOUS", "MEDIUM"
    elif unknown_count >= 3:
        verdict, risk = "UNKNOWN", "UNKNOWN"
    else:
        verdict, risk = "SAFE", "LOW"

    threat_types = []
    for r in results:
        threat_types.extend(r.get("threat_types", []))
        if r.get("malicious", 0) > 0:
            threat_types.append("MALWARE")

    return {
        "verdict": verdict,
        "risk": risk,
        "danger_count": danger_count,
        "safe_count": safe_count,
        "unknown_count": unknown_count,
        "threat_types": list(set(threat_types)),
    }


# ──────────────────────────────────────────────
# ENDPOINT 1 — /scan/url
# For website use — full deep scan
# ──────────────────────────────────────────────
@app.post("/scan/url")
async def scan_url(body: URLRequest):
    url = normalize_url(body.url)
    domain = extract_domain(url)
    if not domain:
        raise HTTPException(status_code=400, detail="Invalid URL")

    async with httpx.AsyncClient() as client:
        google_res, vt_res = await asyncio.gather(
            scan_google(url, client),
            scan_virustotal(url, client),
        )
        urlscan_res = await scan_urlscan_full(url, client)

    results = [google_res, vt_res, urlscan_res]
    summary = aggregate_verdict(results)

    return {
        "url": url,
        "domain": domain,
        "scanned_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "scanners": results,
    }


# ──────────────────────────────────────────────
# ENDPOINT 2 — /scan/extension
# For extension auto-scan — fast, 1-3s
# Step 1: Whitelist check → instant if trusted
# Step 2: All 3 APIs in parallel, 2s timeout each
# ──────────────────────────────────────────────
@app.post("/scan/extension")
async def scan_extension(body: URLRequest):
    url = normalize_url(body.url)
    domain = extract_domain(url)
    if not domain:
        raise HTTPException(status_code=400, detail="Invalid URL")

    # Instant whitelist check — 0ms
    if is_trusted(domain):
        return {
            "url": url,
            "domain": domain,
            "scanned_at": datetime.utcnow().isoformat() + "Z",
            "whitelisted": True,
            "summary": {
                "verdict": "SAFE",
                "risk": "LOW",
                "danger_count": 0,
                "safe_count": 1,
                "unknown_count": 0,
                "threat_types": [],
            },
            "scanners": [{"source": "Whitelist", "safe": True, "details": f"{domain} is a trusted domain"}],
        }

    # Unknown domain — run all 3 in parallel with hard 2s timeout
    async with httpx.AsyncClient() as client:
        async def google_safe():
            try:
                return await asyncio.wait_for(scan_google(url, client), timeout=2.0)
            except:
                return {"source": "Google Safe Browsing", "safe": None, "error": "Timeout"}

        async def vt_safe():
            try:
                return await asyncio.wait_for(scan_virustotal(url, client), timeout=2.0)
            except:
                return {"source": "VirusTotal", "safe": None, "error": "Timeout"}

        async def urlscan_safe():
            try:
                return await asyncio.wait_for(scan_urlscan_fast(url, client), timeout=2.0)
            except:
                return {"source": "URLScan.io", "safe": None, "error": "Timeout"}

        google_res, vt_res, urlscan_res = await asyncio.gather(
            google_safe(), vt_safe(), urlscan_safe()
        )

    results = [google_res, vt_res, urlscan_res]
    summary = aggregate_verdict(results)

    return {
        "url": url,
        "domain": domain,
        "scanned_at": datetime.utcnow().isoformat() + "Z",
        "whitelisted": False,
        "summary": summary,
        "scanners": results,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sentinel-url-detector"}