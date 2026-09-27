"""Name / address normalization. Owner: Person 2.  Contract: docs/CONTRACT.md §3.

Row-wise, stateless, deterministic; never branch on the country value (France is unseen in train).
EDA findings to handle (see outputs/eda_report.txt):
  - junk prefixes/wrappers: "-- ", "<< ", "#", "[...]"
  - names given as domains/handles: "aimsons.com", "@barretocardiology"
  - legal suffix variants: Pvt/Private, Ltd/Limited, Inc, LLC/L.L.C., PLLC, SARL, SAS ...
  - "&" vs "and", case differences (S2 addresses are UPPERCASE)
  - Indic-script names (Devanagari, Telugu, Bengali, Tamil, Malayalam) in S2/S3 India
  - address: Rd/Road, St/Street, full state name vs code, "NULL" tokens, component reordering

Unicode: Python's `\\w` does not match Indic vowel signs / viramas (categories Mn/Mc), so punctuation is
removed with a translate table over Unicode categories P*/S*, and token regexes use whitespace
lookarounds instead of `\\b`. Non-Latin scripts are never transliterated.
"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa

_STR = pd.ArrowDtype(pa.string())
_SLICE = 200_000  # rows per internal slice: bounds peak memory of the Python-object intermediates

# Latin diacritics (U+0300-036F only: Indic vowel signs live in their own blocks) and format chars (ZWNJ ...).
_FOLD = {i: None for i in range(0x300, 0x370)}
_PUNCT = {}
for _i in range(sys.maxunicode + 1):
    _cat = unicodedata.category(chr(_i))
    if _cat == "Cf":
        _FOLD[_i] = None
    elif _cat[0] in "PS":
        _PUNCT[_i] = " "
del _i, _cat


def _fold(s: pd.Series) -> pd.Series:
    """casefold, drop Latin accents and format chars; Indic text keeps its marks."""
    return s.str.casefold().str.normalize("NFKD").str.translate(_FOLD).str.normalize("NFC")


def _words(xs) -> str:
    return "|".join(sorted({re.escape(x) for x in xs}, key=len, reverse=True))


# ---- names ----
_TLD = r"(?:com|net|org|in|co|io|biz|info|fr|online|store|shop)"
_WEB_TAIL = re.compile(r"\s*\|\s*(?:https?://)?www\.\S*\s*$")
_ID_TAG = re.compile(r"[\[(]\s*(?:#|id\s*:?)\s*\d+\s*[\])]")
_HANDLE = re.compile(r"^[^\w@]*@[^\s@]+$")
_DOMAIN = re.compile(rf"^(?:https?://)?(?:www\.)?[^\s@/]+(?:\.[^\s@/.]+)*\.{_TLD}/?$")
_LEAD_JUNK = re.compile(r"^[\W_]+")
_MS = re.compile(r"^m\s*/\s*s\.?\s+")
_DOMAIN_PARTS = re.compile(rf"^(?:https?://)?(?:www\.)|(?:\.{_TLD})+/?$")
_DOTTED = re.compile(r"(?<!\w)(?:[a-z]\.)+[a-z]\.?(?!\w)")  # l.l.c. -> llc, p.c. -> pc
_SPACES = re.compile(r"\s+")

# Unambiguous abbreviations: never a meaningful word, removed anywhere in the name.
_LEGAL_ANY = ["llc", "pllc", "llp", "pvt", "ltd", "inc"]
# Real words or short codes: removed only as a trailing run.
_LEGAL_END = _LEGAL_ANY + [
    "private", "limited", "corp", "corporation", "incorporated", "co", "company", "lp", "pc", "plc",
    "gmbh", "ag", "aktiengesellschaft", "sa", "sarl", "sas", "sasu", "eurl", "snc", "bv", "nv", "pte", "pty",
    # Indic "private" / "limited" / short forms / "LLP", as they appear in the data
    "प्राइवेट", "लिमिटेड", "प्रा", "लि", "एलएलपी",
    "ప్రైవేట్", "లిమిటెడ్", "ఎల్ఎల్పీ",
    "ಪ್ರೈವೇಟ್", "ಲಿಮಿಟೆಡ್", "ಎಲ್ಎಲ್ಪಿ",
    "பிரைவேட்", "லிமிடெட்", "எல்எல்பி",
    "প্রাইভেট", "লিমিটেড", "এলএলপি",
    "પ્રાઇવેટ", "લિમિટેડ", "પ્રા", "લિ", "એલએલપી",
    "പ്രൈവറ്റ്", "ലിമിറ്റഡ്",
    "ପ୍ରାଇଭେଟ୍", "ଲିମିଟେଡ୍",
    "ਪ੍ਰਾਈਵੇਟ", "ਲਿਮਟਿਡ", "ਪ੍ਰਾ", "ਲਿ",
]
_LEGAL_END = _fold(pd.Series(_LEGAL_END, dtype=object)).tolist()
_RE_LEGAL_ANY = re.compile(rf"(?<!\S)(?:{_words(_LEGAL_ANY)})(?!\S)")
_RE_LEGAL_END = re.compile(rf"(?:\s+(?:{_words(_LEGAL_END)}))+$")
_EDGE_AND = re.compile(r"^(?:and\s+)+|(?:\s+and)+$")
_INDIC = re.compile(r"[ऀ-ൿ]")

# ---- addresses ----
_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "jammu & kashmir": "jk", "ladakh": "la",
    "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
}
# Whole comma-separated component only, so "New Delhi" or "Washington Street" stay intact.
_RE_STATE = re.compile(rf"(?:^|(?<=,))\s*({_words(_STATES)})\s*(?=,|$)")
_NULL = re.compile(r"(?<![^\s,])<?\s*(?:null|none|nan|n/a)\s*>?(?![^\s,])")
_HOUSE = re.compile(
    r"(?:^|,)\s*(?:(?:house|h|door|plot|flat|shop|bldg|building)\s*\.?\s*)?(?:(?:no|number)\s*\.?\s*|#+\s*)?"
    r"[-:]?\s*0*([1-9]\d*[a-z]?(?:\s*[/-]\s*\d+[a-z]?)*)(?=[\s,.]|$)"
)
_ABBR = {
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "boulevard": "blvd", "court": "ct", "place": "pl", "highway": "hwy", "parkway": "pkwy", "circle": "cir",
    "terrace": "ter", "square": "sq", "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "saint": "st", "mount": "mt", "fort": "ft",
    "number": "no", "near": "nr", "opposite": "opp", "sector": "sec",
}
_RE_ABBR = re.compile(rf"(?<!\S)({_words(_ABBR)})(?!\S)")


def _clean_spaces(s: pd.Series) -> pd.Series:
    return s.str.replace(_SPACES, " ", regex=True).str.strip()


def _names(raw: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    s = _fold(raw)
    s = s.str.replace(_WEB_TAIL, "", regex=True).str.replace(_ID_TAG, " ", regex=True).str.strip()
    is_handle = s.str.match(_HANDLE)
    s = s.str.replace(_LEAD_JUNK, "", regex=True).str.replace(_MS, "", regex=True)
    is_domain = is_handle | s.str.match(_DOMAIN)
    s = s.str.replace(_DOMAIN_PARTS, "", regex=True)
    s = s.str.replace(_DOTTED, lambda m: m.group().replace(".", ""), regex=True)
    s = _clean_spaces(s.str.replace("&", " and ", regex=False).str.translate(_PUNCT))
    stripped = _clean_spaces(s.str.replace(_RE_LEGAL_ANY, " ", regex=True))
    stripped = stripped.str.replace(_RE_LEGAL_END, "", regex=True).str.replace(_EDGE_AND, "", regex=True)
    s = stripped.where(stripped != "", s)  # a name made only of legal words keeps them
    script = np.where(s.str.contains(_INDIC), "indic", np.where(s.str.contains("[a-z]"), "latin", "other"))
    return s, is_domain.to_numpy(bool), script


def _addresses(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = _fold(raw).str.replace(_NULL, " ", regex=True)
    s = s.str.replace(_RE_STATE, lambda m: " " + _STATES[m.group(1)], regex=True)
    house = s.str.extract(_HOUSE, expand=False).fillna("").str.replace(_SPACES, "", regex=True)
    s = _clean_spaces(s.str.translate(_PUNCT))
    s = s.str.replace(_RE_ABBR, lambda m: _ABBR[m.group(1)], regex=True)
    return s, house


def _as_text(col: pd.Series) -> pd.Series:
    return col.astype(object).where(col.notna(), "").astype(str).astype(object)


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with config.NORM_COLUMNS added; same index and row order, raw columns untouched."""
    parts = {c: [] for c in ("name_norm", "name_is_domain", "name_script", "addr_norm", "house_no")}
    for i in range(0, len(df), _SLICE):
        names = _as_text(df["business_name"].iloc[i:i + _SLICE])
        addrs = _as_text(df["business_address"].iloc[i:i + _SLICE])
        name, dom, script = _names(names)
        addr, house = _addresses(addrs)
        for c, v in zip(parts, (name, dom, script, addr, house)):
            parts[c].append(np.asarray(v, dtype=object) if c != "name_is_domain" else v)
        del names, addrs, name, dom, script, addr, house

    def cat(c, dtype):
        v = np.concatenate(parts.pop(c)) if len(df) else np.array([], dtype=object)
        return pd.Series(v, index=df.index, dtype=dtype)

    name_norm, addr_norm = cat("name_norm", _STR), cat("addr_norm", _STR)
    return df.assign(
        name_norm=name_norm,
        addr_norm=addr_norm,
        name_is_domain=cat("name_is_domain", bool),
        name_script=cat("name_script", _STR),
        addr_empty=(addr_norm == "").to_numpy(bool),
        house_no=cat("house_no", _STR),
    )
