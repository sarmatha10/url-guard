"""URL feature extraction + human-readable risk indicators."""
import math
import re
from collections import Counter
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = ["login", "signin", "verify", "secure", "account", "update", "bank",
                       "wallet", "free", "bonus", "confirm", "password", "suspend", "invoice",
                       "payment", "prize", "claim", "unlock", "recover", "kyc"]
BRANDS = ["paypal", "amazon", "google", "facebook", "microsoft", "apple", "netflix", "instagram",
          "whatsapp", "sbi", "hdfc", "icici", "paytm", "flipkart", "linkedin", "dropbox"]
RISKY_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "zip", "country",
              "support", "loan", "icu", "buzz", "cam", "rest"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "cutt.ly",
              "rb.gy", "shorturl.at"}
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length", "num_dots", "num_hyphens",
    "num_digits", "num_at", "num_double_slash", "num_question", "num_equals", "num_percent",
    "num_underscore", "subdomain_count", "has_ip", "risky_tld", "has_punycode", "is_shortener",
    "is_https", "nonstandard_port", "keyword_count", "brand_misuse", "typosquat",
    "entropy", "digit_ratio", "host_hyphens", "host_digits",
]


def normalize(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def split_host(host: str):
    """Return (subdomain_parts, registered_domain_label, tld, registered_domain)."""
    parts = host.split(".")
    if len(parts) < 2:
        return [], host, "", host
    tld = parts[-1]
    if len(parts) >= 3 and parts[-2] in SECOND_LEVEL and len(tld) == 2:
        return parts[:-3], parts[-3], tld, ".".join(parts[-3:])
    return parts[:-2], parts[-2], tld, ".".join(parts[-2:])


def registered_domain(url: str) -> str:
    try:
        host = (urlparse(normalize(url)).hostname or "").lower()
    except ValueError:
        host = ""
    return split_host(host)[3]


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract(url: str) -> dict:
    full = normalize(url)
    try:
        p = urlparse(full)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:                      # malformed URLs (bad port / bracket) are common in real data
        p, host, port = urlparse("http://invalid"), "", None
    path, query = p.path or "", p.query or ""
    subs, reg_label, tld, _ = split_host(host)
    lower = full.lower()
    letters = sum(c.isalpha() for c in full)
    digits = sum(c.isdigit() for c in full)

    is_ip = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host)) or bool(re.fullmatch(r"0x[0-9a-f]+", host))
    kw = sum(k in lower for k in SUSPICIOUS_KEYWORDS)
    outside = ".".join(subs).lower() + "/" + path.lower() + "?" + query.lower()
    brand_misuse = int(
    any(
        b in outside and b not in reg_label
        for b in BRANDS
    )
    or any(
        b in reg_label and (
            "-" in reg_label or any(k in reg_label for k in SUSPICIOUS_KEYWORDS)
        )
        for b in BRANDS
    )
)
    typo = 0 if is_ip else int(any(0 < edit_distance(reg_label, b) <= 2 and len(reg_label) >= 5 for b in BRANDS))

    return {
        "url_length": len(full), "host_length": len(host), "path_length": len(path),
        "query_length": len(query), "num_dots": full.count("."), "num_hyphens": full.count("-"),
        "num_digits": digits, "num_at": full.count("@"),
        "num_double_slash": full[8:].count("//"), "num_question": full.count("?"),
        "num_equals": full.count("="), "num_percent": full.count("%"),
        "num_underscore": full.count("_"), "subdomain_count": len(subs),
        "has_ip": int(is_ip), "risky_tld": int(tld in RISKY_TLDS),
        "has_punycode": int("xn--" in host), "is_shortener": int(host in SHORTENERS),
        "is_https": int(p.scheme == "https"),
        "nonstandard_port": int(port not in (None, 80, 443)),
        "keyword_count": kw, "brand_misuse": brand_misuse, "typosquat": typo,
        "entropy": round(entropy(full), 4), "digit_ratio": round(digits / max(letters, 1), 4),
        "host_hyphens": host.count("-"), "host_digits": sum(c.isdigit() for c in host),
    }


def vectorize(url: str):
    f = extract(url)
    return [f[n] for n in FEATURE_NAMES]


def explain(url: str) -> list:
    """Human-readable risk indicators: [{text, severity}] severity: high|medium|low|good."""
    f, out = extract(url), []
    add = lambda t, s: out.append({"text": t, "severity": s})

    if f["has_ip"]: add("Uses a raw IP address instead of a domain name", "high")
    if f["num_at"]: add("Contains '@' - browsers ignore everything before it, used to hide the real host", "high")
    if f["brand_misuse"]: add("A well-known brand name appears in the subdomain/path of an unrelated domain (impersonation)", "high")
    if f["typosquat"]: add("Domain looks like a misspelling of a popular brand (typosquatting)", "high")
    if f["has_punycode"]: add("Punycode (xn--) domain - can hide look-alike characters", "high")
    if f["risky_tld"]: add("Uses a top-level domain frequently abused for scams", "medium")
    if f["is_shortener"]: add("URL shortener hides the real destination", "medium")
    if f["keyword_count"] >= 2: add(f"Contains {f['keyword_count']} phishing-style keywords (login, verify, secure...)", "medium")
    elif f["keyword_count"] == 1: add("Contains a phishing-style keyword", "low")
    if f["subdomain_count"] >= 3: add(f"Many subdomains ({f['subdomain_count']}) - common in deceptive URLs", "medium")
    if f["host_hyphens"] >= 3: add(f"Hostname has {f['host_hyphens']} hyphens", "medium")
    if f["nonstandard_port"]: add("Uses a non-standard port", "medium")
    if f["url_length"] > 75: add(f"Unusually long URL ({f['url_length']} characters)", "low")
    if f["num_percent"] >= 3: add("Heavy percent-encoding, may be obfuscating content", "low")
    if f["entropy"] > 4.6: add("High character randomness (looks auto-generated)", "low")
    if not f["is_https"]: add("Does not use HTTPS", "low")
    else: add("Uses HTTPS (note: phishing sites also use it)", "good")
    if all(o["severity"] == "good" for o in out):
        add("No suspicious URL patterns detected", "good")
    return out
