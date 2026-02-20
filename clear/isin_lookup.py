import json
import re
import time
import urllib.parse
import urllib.request

COMPANIES_PATH = "/workspaces/new-PPK/clear/companies.txt"
OUTPUT_PATH = "/workspaces/new-PPK/clear/isin_output.txt"

# Stooq tickers for selected Polish listings.
STOOQ_TICKERS = {
    "Alior Bank SA": "alr",
    "Allegro.eu SA": "ale",
    "Asseco Poland SA": "acp",
    "Bank Millennium SA": "mil",
    "Bank Pekao SA": "peo",
    "Benefit Systems SA": "bft",
    "Budimex SA": "bdx",
    "CCC SA": "ccc",
    "CD Projekt SA": "cdr",
    "Cyber_Folks SA": "cfs",
    "DataWalk SA": "dat",
    "Dino Polska SA": "dnp",
    "Grupa Kęty SA": "kty",
    "Grupa Pracuj SA": "gpp",
    "ING Bank Śląski SA": "ing",
    "KGHM Polska Miedź SA": "kgh",
    "Kruk SA": "kru",
    "LPP SA": "lpp",
    "mBank SA": "mbk",
    "Mirbud SA": "mrb",
    "Orange Polska SA": "opl",
    "Orlen SA": "pkn",
    "Pepco Group N.V.": "pco",
    "PGE Polska Grupa Energetyczna SA": "pge",
    "PKO Bank Polski SA": "pko",
    "PKP Cargo SA": "pkp",
    "Powszechny Zakład Ubezpieczeń SA": "pzu",
    "Rainbow Tours SA": "rbt",
    "Render Cube SA": "rcu",
    "Selvita SA": "slv",
    "Shoper SA": "sho",
    "Synektik SA": "snt",
    "Torpol SA": "tor",
    "Vercom SA": "vrc",
    "XTB SA": "xtb",
}

STOOQ_PROFILE_URL = "https://stooq.pl/q/p/?s={ticker}"
WIKIDATA_SEARCH_URL = (
    "https://www.wikidata.org/w/api.php?"
    "action=wbsearchentities&format=json&language={lang}&limit=5&search={query}"
)
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

ISIN_RE = re.compile(r"<td[^>]*>\s*ISIN\s*</td>\s*<td[^>]*>([^<]+)</td>", re.IGNORECASE)
LEGAL_SUFFIXES = [
    " s.a.",
    " s.a",
    " sa",
    " inc.",
    " inc",
    " corp.",
    " corp",
    " ltd.",
    " ltd",
    " plc",
    " n.v.",
    " s.p.a.",
    " ag",
    " se",
    " company",
    " co.",
    " co",
    " group",
    " holdings",
    " holding",
]
ALIASES = {
    "JP Morgan Chase & Co.": ["JPMorgan Chase", "JPMorgan Chase & Co."],
    "CrowdStrike Holdings Inc": ["CrowdStrike", "CrowdStrike Holdings"],
    "Waste Management Inc": ["Waste Management", "Waste Management, Inc."],
    "InPost SA": ["InPost"],
    "Münchener Rück AG": ["Munich Re"],
    "Zabka Group SA": ["Zabka Group", "Żabka Group"],
    "Render Cube SA": ["Render Cube"],
    "Diagnostyka SA": ["Diagnostyka"],
    "Compremum SA": ["Compremum"],
    "Vercom SA": ["Vercom"],
    "Shoper SA": ["Shoper"],
    "Synektik SA": ["Synektik"],
    "Selvita SA": ["Selvita"],
    "LVMH SA": ["LVMH"],
    "Grenergy Renovables SA": ["Grenergy"],
    "Oracle Corp.": ["Oracle"],
    "Meta Platforms Inc": ["Meta Platforms", "Facebook"],
    "Applied Materials Inc": ["Applied Materials"],
    "Broadcom Inc": ["Broadcom"],
    "Amazon.com Inc": ["Amazon"],
    "Intel Corp.": ["Intel"],
    "Microsoft Corp.": ["Microsoft"],
    "Nvidia Corp.": ["NVIDIA"],
}


def http_get(url, headers=None, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def fetch_stooq_isin(ticker):
    url = STOOQ_PROFILE_URL.format(ticker=urllib.parse.quote(ticker))
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": f"privacy={int(time.time())}",
    }
    html = http_get(url, headers=headers)
    match = ISIN_RE.search(html)
    return match.group(1).strip() if match else None


def fetch_wikidata_isin(company):
    for query in build_queries(company):
        for lang in ("en", "pl"):
            search_url = WIKIDATA_SEARCH_URL.format(
                query=urllib.parse.quote(query),
                lang=lang,
            )
            search_json = json.loads(http_get(search_url, headers={"User-Agent": "Mozilla/5.0"}))
            results = search_json.get("search", [])
            for result in results:
                qid = result.get("id")
                if not qid:
                    continue
                entity_url = WIKIDATA_ENTITY_URL.format(qid=qid)
                entity_json = json.loads(http_get(entity_url, headers={"User-Agent": "Mozilla/5.0"}))
                entity = entity_json.get("entities", {}).get(qid, {})
                claims = entity.get("claims", {})
                isin_claims = claims.get("P946", [])
                if not isin_claims:
                    continue
                datavalue = isin_claims[0].get("mainsnak", {}).get("datavalue", {})
                isin = datavalue.get("value")
                if isin:
                    return isin
    return None


def build_queries(company):
    base = company.strip()
    cleaned = base.replace("_", " ").replace("&", "and")

    candidates = [base, cleaned]
    candidates.extend(ALIASES.get(company, []))
    lowered = cleaned.lower().strip()
    for suffix in LEGAL_SUFFIXES:
        if lowered.endswith(suffix):
            stripped = cleaned[: -len(suffix)].strip(" ,.")
            candidates.append(stripped)
            candidates.append(stripped.replace(".", ""))

    candidates.append(cleaned.replace(".", ""))
    candidates.append(cleaned.replace(",", ""))

    seen = set()
    unique = []
    for candidate in candidates:
        key = candidate.lower()
        if candidate and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def load_companies(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    companies = load_companies(COMPANIES_PATH)
    results = []

    for company in companies:
        isin = None
        ticker = STOOQ_TICKERS.get(company)
        if ticker:
            try:
                isin = fetch_stooq_isin(ticker)
            except Exception:
                isin = None

        if not isin:
            try:
                isin = fetch_wikidata_isin(company)
            except Exception:
                isin = None

        time.sleep(0.2)

        results.append((company, isin or ""))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        for company, isin in results:
            handle.write(f"{company};{isin}\n")

    for company, isin in results:
        print(f"{company};{isin}")


if __name__ == "__main__":
    main()
