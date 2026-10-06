"""Country-agnostic text normalisation for business names and addresses.

Long forms are canonicalised to short tokens (street->st, saint->st, limited->ltd)
so the same rules behave sensibly for any country label, including ones never
seen in training. Country-specific tables (US / India state codes) are only an
enrichment; unknown countries fall through to the generic path.
"""
import re
import unicodedata

from unidecode import unidecode

INDIC_RE = re.compile(r"[ऀ-෿]")
RAW_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ɏऀ-෿‌‍]+")
ALIAS_RE = re.compile(
    r"\s+(?:formerly\s+known\s+as|formerly|f/k/a|fka|a/k/a|aka|d/b/a|dba:?|t/a|trading\s+as)\s+",
    re.I,
)
URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|org\.in|com|net|org|in|co|biz|info|us|fr|io)\b",
    re.I,
)
PHONE_RE = re.compile(r"\+?\d[\d \-]{6,}\d")
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th|er|e|eme)$")
DIGITS_RE = re.compile(r"\d+")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})

NAME_CANON = {
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "cos": "co", "llc": "llc", "llp": "llp", "lp": "lp",
    "pllc": "pllc", "pc": "pc", "plc": "plc", "ltda": "ltd",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "scop": "scop", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "and": "and", "et": "and", "und": "and", "y": "and",
    "intl": "international", "international": "international",
    "svcs": "services", "svc": "services", "service": "services", "services": "services",
    "assoc": "associates", "assocs": "associates", "associates": "associates",
    "mfg": "manufacturing", "tech": "technology", "technologies": "technology",
    "technology": "technology", "ent": "enterprises", "enterprise": "enterprises",
    "bros": "brothers", "dept": "department", "univ": "university", "hosp": "hospital",
    "ctr": "center", "centre": "center", "shri": "sri", "shree": "sri", "sree": "sri",
    "st": "st", "saint": "st", "sainte": "st", "ste": "st", "mt": "mount", "ft": "fort",
    "ets": "etablissements", "etablissement": "etablissements",
}
LEGAL = {
    "ltd", "pvt", "corp", "inc", "co", "llc", "llp", "lp", "pllc", "pc", "plc", "sarl",
    "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "gmbh", "ag", "bv", "nv", "opc",
}
NAME_STOP = {"the", "and", "of", "a", "an", "du", "de", "des", "la", "le", "les", "l", "d", "en", "for", "at"}

ADDR_CANON = {
    "street": "st", "str": "st", "st": "st", "road": "rd", "rd": "rd", "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr", "drv": "dr", "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd", "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl", "plaza": "plz", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "circle": "cir", "cir": "cir", "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq", "expressway": "expy",
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "appartement": "apt",
    "appt": "apt", "building": "bldg", "bldg": "bldg", "batiment": "bldg", "bat": "bldg",
    "floor": "fl", "fl": "fl", "flr": "fl", "etage": "fl", "room": "rm", "rm": "rm",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "nord": "n", "sud": "s", "est": "e", "ouest": "w",
    "mount": "mount", "mt": "mount", "fort": "fort", "ft": "fort", "saint": "st", "sainte": "st",
    "rue": "rue", "r": "rue", "impasse": "imp", "imp": "imp", "chemin": "ch", "ch": "ch",
    "allee": "all", "all": "all", "route": "rte", "rte": "rte", "quai": "quai", "qu": "quai",
    "residence": "res", "res": "res", "lieu": "lieu", "dit": "dit", "cours": "crs", "crs": "crs",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "behind": "bhd",
    "number": "no", "no": "no", "nos": "no", "house": "h", "h": "h", "hno": "h",
    "sector": "sec", "sec": "sec", "nagar": "nagar", "ngr": "nagar", "marg": "marg",
    "colony": "col", "col": "col", "society": "soc", "soc": "soc", "complex": "cmplx",
    "cmplx": "cmplx", "industrial": "ind", "ind": "ind", "estate": "est", "extension": "ext",
    "ext": "ext", "main": "main", "cross": "crs", "post": "po", "po": "po", "office": "off",
    "village": "vill", "vill": "vill", "vpo": "vill", "district": "dist", "dist": "dist",
    "taluka": "tal", "tal": "tal", "tehsil": "tal", "city": "city", "town": "town",
    "township": "twp", "twp": "twp", "county": "cnty", "cnty": "cnty", "unit": "unit",
    "pmb": "pmb", "box": "box",
    "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata", "bangalore": "bengaluru",
    "poona": "pune", "baroda": "vadodara", "gurgaon": "gurugram", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysore": "mysuru", "benares": "varanasi", "allahabad": "prayagraj",
    "simla": "shimla", "pondicherry": "puducherry", "mangalore": "mangaluru", "belgaum": "belagavi",
    "shimoga": "shivamogga", "gulbarga": "kalaburagi", "hubli": "hubballi", "vizag": "visakhapatnam",
    "calicut": "kozhikode", "orissa": "odisha", "uttaranchal": "uttarakhand", "cawnpore": "kanpur",
}
ADDR_DROP = {"no", "the", "of", "de", "du", "des", "la", "le", "les", "l", "d", "and", "et"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld", "andaman and nicobar islands": "an",
}
IN_CODE_ALIASES = {"ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "tl": "tg"}
# French departments -> their region, so "Nord" and "Hauts-de-France" canonicalise to
# the same token (records alternate between the two, like US state codes vs names).
FR_REGIONS = {
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf", "aisne": "hdf", "oise": "hdf", "somme": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq", "dordogne": "naq", "landes": "naq", "lot et garonne": "naq",
    "pyrenees atlantiques": "naq", "charente": "naq", "charente maritime": "naq", "correze": "naq", "creuse": "naq",
    "deux sevres": "naq", "vienne": "naq", "haute vienne": "naq",
    "pays de la loire": "pdl", "loire atlantique": "pdl", "maine et loire": "pdl", "mayenne": "pdl", "sarthe": "pdl",
    "vendee": "pdl",
    "ile de france": "idf", "paris": "idf", "auvergne rhone alpes": "ara", "provence alpes cote d azur": "pac",
    "occitanie": "occ", "grand est": "ges", "bretagne": "bre", "normandie": "nor", "bourgogne franche comte": "bfc",
    "centre val de loire": "cvl", "corse": "cor",
}

_TRANSLIT_NAME = {}
_TRANSLIT_ADDR = {}


def set_translit(name_dict, addr_dict):
    _TRANSLIT_NAME.clear()
    _TRANSLIT_NAME.update(name_dict or {})
    _TRANSLIT_ADDR.clear()
    _TRANSLIT_ADDR.update(addr_dict or {})


def raw_tokens(s):
    """Tokenise keeping Indic combining marks attached (\\w would split them)."""
    s = unicodedata.normalize("NFKC", s).replace(".", "").replace("'", "").replace("’", "")
    return RAW_TOKEN_RE.findall(s)


def latin(tok):
    return unidecode(tok).lower()


def romanize(tok):
    """Fallback romanisation of an Indic token, squeezed to reduce vowel noise."""
    r = re.sub(r"[^a-z0-9]", "", unidecode(tok).lower())
    return re.sub(r"(.)\1+", r"\1", r)


def _map_token(tok, table):
    if INDIC_RE.search(tok):
        hit = table.get(tok)
        if hit is None and table is _TRANSLIT_ADDR:
            hit = _TRANSLIT_NAME.get(tok)
        return hit.split() if hit else [romanize(tok)]
    t = latin(tok)
    t = re.sub(r"[^a-z0-9]", "", t)
    return [t] if t else []


def _deleet(t):
    if t.isalpha() or t.isdigit() or ORDINAL_RE.match(t):
        return t
    if any(c.isalpha() for c in t) and any(c.isdigit() for c in t):
        return t.translate(LEET)
    return t


def _name_tokens(s):
    toks = []
    for rt in raw_tokens(s.replace("&", " and ").replace("+", " ")):
        for t in _map_token(rt, _TRANSLIT_NAME):
            t = _deleet(t)
            toks.append(NAME_CANON.get(t, t))
    return toks


def normalize_name(raw):
    """Return dict of name representations."""
    s = unicodedata.normalize("NFKC", raw or "").strip()
    parts = [p.strip() for p in s.split("|")]
    main, extras = parts[0], parts[1:]
    domains = []
    for e in extras:
        m = URL_RE.search(e)
        if m:
            domains.append(m.group(1))
    has_phone = bool(PHONE_RE.search(main))
    main = PHONE_RE.sub(" ", main)
    is_domain = False
    m = URL_RE.search(main)
    if m:
        is_domain = True
        main = URL_RE.sub(lambda mm: " " + mm.group(1).replace("-", " ") + " ", main)
    alias_parts = ALIAS_RE.split(main)
    alt_toks = _name_tokens(alias_parts[-1]) if len(alias_parts) > 1 else []
    brand_toks = _name_tokens(alias_parts[0]) if len(alias_parts) > 1 else []
    full_toks = brand_toks + alt_toks if len(alias_parts) > 1 else _name_tokens(main)
    core = [t for t in full_toks if t not in LEGAL and t not in NAME_STOP]
    alt_core = [t for t in alt_toks if t not in LEGAL and t not in NAME_STOP]
    legal = sorted({t for t in full_toks if t in LEGAL})
    digits = [t for t in full_toks if t.isdigit()]
    return {
        "n_full": " ".join(full_toks),
        "n_core": " ".join(core),
        "n_alt": " ".join(alt_core),
        "n_brand": " ".join(brand_toks),
        "n_concat": "".join(t for t in core if not t.isdigit()),
        "n_legal": " ".join(legal),
        "n_digits": " ".join(digits),
        "n_dom": " ".join(d.lower() for d in domains),
        "f_domain": is_domain,
        "f_alias": len(alias_parts) > 1,
        "f_phone": has_phone,
        "f_nonlatin": bool(INDIC_RE.search(s)),
    }


def normalize_address(raw, country):
    s = unicodedata.normalize("NFKC", raw or "").strip()
    if not s:
        return {"a_full": "", "a_nums": "", "a_house": "", "a_tail": "", "f_addr_nonlatin": False}
    comps = [c for c in s.split(",") if c.strip()]
    toks_by_comp = []
    for c in comps:
        ct = []
        for rt in raw_tokens(c.replace("#", " ").replace("/", " / ").replace("-", " ")):
            ct.extend(_map_token(rt, _TRANSLIT_ADDR))
        toks_by_comp.append([t for t in ct if t and t != "/"])
    cl = (country or "").strip().lower()
    table, codes, aliases = None, None, {}
    if cl in ("us", "usa", "united states"):
        table, codes = US_STATES, US_CODES
    elif cl == "india":
        table, codes, aliases = IN_STATES, IN_CODES, IN_CODE_ALIASES
    elif cl == "france":
        table, codes = FR_REGIONS, set()
    if table is not None:
        for ci, ct in enumerate(toks_by_comp):
            words = [t for t in ct if not t.isdigit()]
            key = " ".join(words)
            code = table.get(key)
            if code is None and len(words) == 1 and aliases.get(key, key) in codes:
                code = aliases.get(key, key)
            if code is not None:
                toks_by_comp[ci] = ["state_" + code] + [t for t in ct if t.isdigit()]
    flat = [t for c in toks_by_comp for t in c]
    out, nums = [], []
    for t in flat:
        if t.isdigit():
            nums.append(t.lstrip("0") or "0")
            out.append(nums[-1])
            continue
        if any(ch.isdigit() for ch in t):
            if not ORDINAL_RE.match(t):
                for d in DIGITS_RE.findall(t):
                    nums.append(d.lstrip("0") or "0")
            out.append(t)
            continue
        t = ADDR_CANON.get(t, t)
        if t in ADDR_DROP:
            continue
        out.append(t)
    tail = toks_by_comp[-1] if toks_by_comp else []
    return {
        "a_full": " ".join(out),
        "a_nums": " ".join(nums),
        "a_house": nums[0] if nums else "",
        "a_tail": " ".join(tail),
        "f_addr_nonlatin": bool(INDIC_RE.search(s)),
    }


US_CODES = set(US_STATES.values())
IN_CODES = set(IN_STATES.values())
