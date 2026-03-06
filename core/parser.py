"""
Parses raw Japanese string data from the SUUMO CSVs into usable Python types.

Key patterns handled
--------------------
- Rent: "13.4万円" → 134_000.0   |  "12000円" → 12_000.0
- Address → ward/city: "東京都中央区銀座１" → "中央区"
- Access:  "東京メトロ有楽町線/新富町駅 歩2分 | …" → list of dicts
"""

from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# Rent / fee parsing
# ---------------------------------------------------------------------------

_MAN_RE = re.compile(r"([\d.]+)\s*万円")   # e.g. "13.4万円"
_YEN_RE = re.compile(r"([\d,]+)\s*円")     # e.g. "12000円" or "12,000円"


def parse_rent(s: str | None) -> float | None:
    """
    Convert a Japanese rent string to a float in yen.

    >>> parse_rent("13.4万円")
    134000.0
    >>> parse_rent("12000円")
    12000.0
    >>> parse_rent(None) is None
    True
    """
    if not s or not isinstance(s, str):
        return None

    s = s.strip()

    m = _MAN_RE.search(s)
    if m:
        return float(m.group(1)) * 10_000

    m = _YEN_RE.search(s)
    if m:
        return float(m.group(1).replace(",", ""))

    return None


def parse_fee(s: str | None) -> float:
    """Same as parse_rent but returns 0 when the value is missing/unparseable.

    >>> parse_fee("12000円")
    12000.0
    >>> parse_fee(None)
    0
    """
    val = parse_rent(s)
    return val if val is not None else 0


# ---------------------------------------------------------------------------
# Address → area extraction
# ---------------------------------------------------------------------------

_WARD_RE = re.compile(r"東京都(.{1,4}区)")         # 東京都XX区
_CITY_RE = re.compile(r"(?:埼玉県|千葉県|神奈川県)(.{1,5}市)")  # XX市


def extract_area(addr: str | None) -> str | None:
    """
    Pull the ward (区) or city (市) from a full address string.

    >>> extract_area("東京都中央区銀座１")
    '中央区'
    >>> extract_area("埼玉県さいたま市大宮区")
    'さいたま市'
    >>> extract_area(None) is None
    True
    """
    if not addr or not isinstance(addr, str):
        return None

    m = _WARD_RE.search(addr)
    if m:
        return m.group(1)

    m = _CITY_RE.search(addr)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# Access string → station list
# ---------------------------------------------------------------------------

_ACCESS_RE = re.compile(r"/(.+?)駅\s*歩(\d+)分")


def parse_access(s: str | None) -> list[dict]:
    """
    Parse a pipe-delimited access string into a list of station dicts.

    >>> parse_access("東京メトロ有楽町線/新富町駅 歩2分 | 東京メトロ日比谷線/東銀座駅 歩6分")
    [{'station': '新富町', 'walk_min': 2}, {'station': '東銀座', 'walk_min': 6}]
    >>> parse_access(None)
    []
    """
    if not s or not isinstance(s, str):
        return []

    results = []
    for part in s.split("|"):
        m = _ACCESS_RE.search(part)
        if m:
            results.append({
                "station": m.group(1).strip(),
                "walk_min": int(m.group(2)),
            })
    return results
