"""ADIM 18 — URL ve dosya adı güvenliği testleri.

Bağımsız çalışır:
    .venv\\Scripts\\python.exe tests\\test_url_safety.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from academic_fetcher import AcademicFetcher, is_safe_public_url, safe_ref_token  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


# --- URL güvenliği ---------------------------------------------------------

blocked = {
    "http://127.0.0.1:8501/": "loopback",
    "http://localhost/admin": "localhost",
    "http://10.0.0.5/gizli": "özel ağ (10.x)",
    "http://192.168.1.1/": "özel ağ (192.168.x)",
    "http://172.16.5.4/": "özel ağ (172.16.x)",
    "http://169.254.169.254/latest/meta-data/": "link-local (bulut metadata)",
    "file:///C:/Windows/win.ini": "file şeması",
    "javascript:alert(1)": "javascript şeması",
    "ftp://example.com/x.pdf": "ftp şeması",
    "https://[::1]/": "IPv6 loopback",
}

for url, reason in blocked.items():
    check(f"reddet: {reason}", is_safe_public_url(url) is False, url)

allowed = [
    "https://8.8.8.8/dosya.pdf",
    "http://1.1.1.1/x",
]
for url in allowed:
    check(f"izin ver: {url}", is_safe_public_url(url) is True, url)

check("boş URL reddedilir", is_safe_public_url("") is False)
check("None reddedilir", is_safe_public_url(None) is False)

# --- Dosya adı temizliği ---------------------------------------------------

cases = [
    "../../etc/passwd",
    "..\\..\\windows\\system32",
    "ref 12/../../x",
    "APA_7",
]
for raw in cases:
    got = safe_ref_token(raw)
    ok = (
        got != ""
        and "/" not in got
        and "\\" not in got
        and ".." not in got
        and len(got) <= 40
    )
    check(f"ref_id temizle: {raw!r}", ok, got)

check("APA_7 değişmez", safe_ref_token("APA_7") == "APA_7", safe_ref_token("APA_7"))
check("boş ref_id -> ref", safe_ref_token("") == "ref")
check("uzun ref_id kısaltılır", len(safe_ref_token("a" * 100)) <= 40)

# --- Varsayılanlar ---------------------------------------------------------

f = AcademicFetcher()
check("Sci-Hub varsayılan KAPALI", f.enable_scihub is False, str(f.enable_scihub))

passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print(f"\n{passed}/{len(RESULTS)} test geçti")
raise SystemExit(0 if passed == len(RESULTS) else 1)
