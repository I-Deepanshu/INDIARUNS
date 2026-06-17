#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Senior AI Engineer Candidate Ranker

Ranks 100,000 candidates for a Senior AI Engineer role.
Two-layer approach:
  1. Relevance score  — title (gate) + career evidence + skills + experience + location + education
  2. Availability multiplier — behavioral signals (activity, response rate, notice period)

Design principle: title is the primary anti-honeypot gate.  A Marketing Manager or
Java Developer never reaches the top 100 regardless of how many AI skills they listed.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints: CPU only, no network, < 5 min, < 16 GB RAM, stdlib + math only.
"""

import argparse
import csv
import heapq
import itertools
import json
import math
import sys
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 15)
TOP_N = 100

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD SETS
# All keyword matching uses _norm() first: lowercase + hyphens → spaces.
# So "re-ranking" in a description becomes "re ranking" and matches "re ranking".
# ─────────────────────────────────────────────────────────────────────────────

# Explicit title → score mapping.
# NOTE: default floor (0.28) is applied ONLY when NO key matches.
#       Explicit low scores (e.g. 0.10) are honoured over the floor.
TITLE_SCORES = {
    # Tier A — perfect JD fit (all score 1.00)
    "ai engineer":               1.00,
    "ml engineer":               1.00,
    "machine learning engineer": 1.00,
    "applied scientist":         1.00,
    "applied ml":                1.00,
    "applied ai":                1.00,
    "search engineer":           1.00,
    "ranking engineer":          1.00,
    "retrieval engineer":        1.00,
    "nlp engineer":              1.00,
    "nlp scientist":             1.00,
    "information retrieval":     1.00,
    "senior ml":                 1.00,
    "senior ai":                 1.00,
    "senior machine learning":   1.00,
    # Tier A — seniority variants
    "staff ml":                  1.00,
    "staff ai":                  1.00,
    "staff machine learning":    1.00,
    "principal ml":              1.00,
    "principal ai":              1.00,
    "lead ml":                   1.00,
    "lead ai":                   1.00,
    "lead machine learning":     1.00,
    # Tier A — modern AI titles
    "generative ai":             1.00,
    "llm engineer":              1.00,
    "rag engineer":              1.00,
    "vector search engineer":    1.00,
    "recommendation engineer":   1.00,
    "ai research engineer":      0.95,
    # Tier B — strong adjacent
    "senior data scientist":     0.80,
    "data scientist":            0.72,
    "research engineer":         0.65,
    "junior ml":                 0.62,
    "junior ai":                 0.62,
    "research scientist":        0.50,
    # Tier C — neutral; career must corroborate
    "software engineer":         0.42,
    "backend engineer":          0.40,
    "senior engineer":           0.40,
    "platform engineer":         0.38,
    "data engineer":             0.36,
    "full stack":                0.32,
    "analytics engineer":        0.30,
    # Tier D — marginal; title heavily penalises even with good career
    "devops engineer":           0.15,
    "cloud engineer":            0.15,
    "qa engineer":               0.12,
    "frontend engineer":         0.08,
    "java developer":            0.08,
    ".net developer":            0.08,
    "mobile developer":          0.08,
    "data analyst":              0.20,
}

# Hard floor for clear non-tech / honeypot roles
TITLE_NEGATIVE = {
    # HR / people
    "hr manager", "hr business partner", "hrbp",
    "human resources", "human resource",
    "talent acquisition", "technical recruiter", "recruiter",
    # Marketing / sales / BD
    "marketing manager", "content writer", "content manager", "content strategist",
    "digital marketer", "seo specialist", "growth manager",
    "sales executive", "sales manager", "business development",
    # Finance / legal
    "accountant", "financial analyst", "finance manager",
    "legal counsel", "compliance manager",
    # Design / creative
    "graphic designer", "ui designer", "ux designer",
    "product designer", "visual designer",
    # Non-software engineering
    "civil engineer", "mechanical engineer", "electrical engineer",
    "electronics engineer", "manufacturing engineer",
    # Operations / management
    "operations manager", "project manager", "program manager",
    "it manager", "it director",
    "customer support", "customer service", "customer success",
    # Product (non-technical)
    "product manager",
    # Mobile / non-AI dev
    "android developer", "ios developer",
    # Supply chain / logistics
    "supply chain", "logistics manager",
    # Generic non-tech
    "network engineer", "system administrator",
}

# Production AI evidence in role descriptions (matched after hyphen normalisation)
PRODUCTION_KEYWORDS = {
    # ── Explicit retrieval / vector tools ──
    "vector search", "vector retrieval", "vector database", "vector db",
    "semantic search", "hybrid search", "dense retrieval", "sparse retrieval",
    "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "elasticsearch", "opensearch", "solr",
    "ann index", "approximate nearest neighbor", "hnsw",
    "embedding retrieval", "embedding search",
    # ── Ranking / recommendation (specific phrases only) ──
    "ranking system", "recommendation system", "recommender system",
    "recommendation engine", "recommendation pipeline",
    "collaborative filtering", "matrix factorization",
    "re ranking",           # normalised "re-ranking"
    "reranking", "reranker", "two tower",
    "search ranking", "learning to rank", "ltr",
    "gradient boosted re ranking",
    "click through rate", "click through",
    # ── Embeddings / sentence models ──
    "sentence transformer", "text embedding", "embedding model",
    "bge", "e5 model", "openai embedding",
    # ── NLP production work (vocabulary from actual descriptions) ──
    "nlp pipeline", "sentiment analysis", "document classification",
    "transformer based", "distilbert",
    "natural language processing",
    # ── Evaluation frameworks ──
    "ndcg", "mrr", "mean average precision",
    "precision@", "recall@",
    "offline evaluation", "online a/b", "a/b test",
    "evaluation framework", "ranking metric",
    # ── Production deployment signals ──
    "deployed to production", "production deployment",
    "queries per month", "queries per second", "qps",
    "serving latency", "at scale", "real users",
    "mlops", "model serving", "inference pipeline",
    "retrieval augmented", "rag based", "rag pipeline",
    "bm25",
    # ── Fine-tuning ──
    "fine tun", "lora", "qlora", "peft",
    "llama", "mistral", "instruction tuning",
    # ── Predictive ML shipped to users ──
    "predictive modeling", "churn prediction", "demand prediction",
    "ranking model", "scoring model",
    # ── Modern AI stack (post-2023) ──
    "vector store", "vector index", "dense vector", "sparse vector",
    "hybrid retrieval", "dense retrieval pipeline",
    "cross encoder", "bi encoder", "late interaction",
    "colbert", "splade", "bgem3", "bge m3",
    "generative ai", "gen ai pipeline",
    "rag system", "rag architecture", "agentic",
    "chunk", "chunking strategy", "context window",
    "knowledge graph retrieval", "graph rag",
    "trec", "beir benchmark",
}

ADJACENT_KEYWORDS = {
    "llm", "large language model", "language model",
    "transformers", "hugging face", "huggingface",
    "pytorch", "tensorflow", "scikit learn", "xgboost",
    "gradient boosting", "lightgbm", "random forest",
    "nlp", "natural language",
    "machine learning", "deep learning",
    "feature engineering", "model training", "model deployment",
    "python", "spark", "data pipeline",
    "model evaluation", "model monitoring",
    "predictive model",
}

RESEARCH_SIGNALS = {
    "phd research", "academic lab", "university lab",
    "arxiv", "conference paper", "research internship",
    "research assistant", "thesis project",
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra",
    "mphasis", "hexaware", "mindtree",
    "hcl technologies", "hcl tech", "hcl",   # "hcl" alone misses when stored as abbreviation
    "l&t infotech", "ltimindtree", "lti mindtree", "lti",
    "niit technologies", "niit tech",
    "persistent systems", "persistent",
    "oracle financial", "oracle india",
}

CORE_SKILL_KEYWORDS = {
    "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "elasticsearch", "opensearch", "solr",
    "vector search", "vector retrieval", "semantic search",
    "hybrid search", "bm25",
    "embeddings", "sentence transformer", "sentence-transformer",
    "text embedding", "bge", "e5",
    "ndcg", "mrr", "ranking", "retrieval", "reranker",
    "recommendation system", "recommender",
    "information retrieval", "learning to rank",
    "python", "pytorch", "tensorflow",
    "llm", "rag", "fine-tuning", "fine tuning",
}

NICE_SKILL_KEYWORDS = {
    "lora", "qlora", "peft", "transformers",
    "hugging face", "huggingface",
    "xgboost", "gradient boosting", "mlops", "model serving",
    "a/b testing", "nlp", "natural language",
    "lightgbm", "scikit-learn", "scikit learn",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25, "intermediate": 0.60,
    "advanced": 0.85, "expert": 1.00,
}

PREFERRED_CITY_SCORES = {
    "pune": 1.00, "noida": 1.00,
    "new delhi": 0.88, "delhi": 0.88,
    "gurgaon": 0.88, "gurugram": 0.88, "ncr": 0.85,
    "hyderabad": 0.82, "mumbai": 0.82,
    "bangalore": 0.78, "bengaluru": 0.78,
    "chennai": 0.72, "kolkata": 0.68, "ahmedabad": 0.68,
    "jaipur": 0.65, "kochi": 0.63, "trivandrum": 0.63,
    "coimbatore": 0.62, "nagpur": 0.62, "indore": 0.62,
    "bhubaneswar": 0.62,
}

EDUCATION_TIER = {
    "tier_1": 1.00, "tier_2": 0.75,
    "tier_3": 0.52, "tier_4": 0.32, "unknown": 0.42,
}

RELEVANT_FIELDS = {
    # Specific enough to not match unrelated degrees
    "computer science", "information technology", "information systems",
    "software engineering", "software technology",
    "artificial intelligence", "machine learning", "data science",
    "mathematics", "statistics", "computational",
    # Deliberately excluded: "cs" (matches economics/physics/logistics),
    # "engineering" (matches civil/mechanical engineering),
    # "electrical"/"electronics" (hardware, not AI)
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text) -> str:
    """Lowercase + replace hyphens and underscores with spaces."""
    return (text or "").lower().replace("-", " ").replace("_", " ")


def _hits(text: str, keywords: set) -> int:
    """Count keyword hits. Short keywords (≤4 chars) require word boundaries
    to avoid false positives: 'llm' in 'fulfillment', 'rag' in 'storage', etc."""
    t = _norm(text)
    padded = " " + t + " "
    count = 0
    for kw in keywords:
        if len(kw) <= 4:
            count += 1 if (" " + kw + " ") in padded else 0
        else:
            count += 1 if kw in t else 0
    return count


def _kw_match(kw: str, name: str) -> bool:
    """Match a keyword against a (already-normed) skill name.
    Short keywords require whole-word match to prevent 'rag' hitting 'storage'."""
    if len(kw) <= 4:
        return (" " + kw + " ") in (" " + name + " ")
    return kw in name


def _days_since(date_str) -> int:
    if not date_str:
        return 999
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
        return max((REFERENCE_DATE - d).days, 0)
    except (ValueError, TypeError):
        return 999


# ─────────────────────────────────────────────────────────────────────────────
# SCORING COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def score_title(profile: dict, career: list) -> float:
    """
    Returns a title relevance score in [0, 1].
    NEGATIVE titles (honeypots) get 0.03 unless a prior AI role overrides to 0.20.
    EXPLICIT low-score titles (java developer, frontend, etc.) are honoured:
        the default floor (0.28) only applies when NO keyword matches.
    """
    raw = _norm(profile.get("current_title") or "")
    hl  = _norm(profile.get("headline") or "")

    # 1. Hard negative gate
    if any(neg in raw for neg in TITLE_NEGATIVE):
        has_prior_ai = any(
            any(pos in _norm(r.get("title") or "") for pos in [
                "ml engineer", "ai engineer", "nlp", "search engineer",
                "ranking", "data scientist", "applied scientist",
            ]) and (r.get("duration_months") or 0) >= 12
            for r in career
        )
        return 0.20 if has_prior_ai else 0.03

    # 2. Match explicit title keywords; NO default floor applied yet
    matched = None
    for kw, sc in TITLE_SCORES.items():
        if kw in raw:
            matched = max(matched, sc) if matched is not None else sc

    # 3. Apply default floor ONLY when nothing matched (truly unknown title)
    # 0.18 is conservative: unknown titles are treated as likely non-AI until proven otherwise
    best = matched if matched is not None else 0.18

    # 4. Small headline boost
    for kw in ["ai engineer", "ml engineer", "nlp", "search", "retrieval", "ranking"]:
        if kw in hl:
            best = min(1.0, best + 0.04)

    return best


def score_career_history(career: list) -> tuple:
    """Returns (career_score, consulting_fraction, production_role_count)."""
    total_months = cons_months = 0
    role_data = []

    for role in career:
        months = max(int(role.get("duration_months") or 0), 0)
        if months == 0:
            continue
        desc    = (role.get("description") or "")
        rtitle  = (role.get("title") or "")
        company = _norm(role.get("company") or "")
        combined = desc + " " + rtitle

        total_months += months
        if any(f in company for f in CONSULTING_FIRMS):
            cons_months += months

        prod_hits = _hits(combined, PRODUCTION_KEYWORDS)
        adj_hits  = _hits(combined, ADJACENT_KEYWORDS)
        res_hits  = _hits(combined, RESEARCH_SIGNALS)

        if prod_hits >= 4:     rs = 1.00
        elif prod_hits >= 2:   rs = min(0.55 + 0.10 * prod_hits, 1.0)
        elif prod_hits == 1:   rs = 0.38   # single hit is a weak signal; was 0.50
        elif adj_hits >= 4:    rs = 0.32   # adjacent-only; was 0.40
        elif adj_hits >= 2:    rs = 0.18   # was 0.25
        elif adj_hits >= 1:    rs = 0.10   # was 0.15
        else:                  rs = 0.04

        if res_hits >= 2 and prod_hits == 0:
            rs *= 0.40

        # Title corroboration within this specific role
        rt = _norm(rtitle)
        if any(pos in rt for pos in [
            "ml engineer", "ai engineer", "nlp", "search engineer",
            "ranking", "data scientist", "applied scientist",
        ]):
            rs = min(1.0, rs + 0.12)

        role_data.append((rs, months))

    if not role_data or total_months == 0:
        return 0.0, 0.0, 0

    career_score = sum(s * m for s, m in role_data) / total_months
    prod_roles   = sum(1 for s, _ in role_data if s >= 0.45)
    cons_frac    = cons_months / total_months

    # Consulting penalty
    if cons_frac >= 0.90:    career_score *= 0.05
    elif cons_frac >= 0.70:  career_score *= 0.30
    elif cons_frac >= 0.50:  career_score *= 0.60

    return career_score, cons_frac, prod_roles


def score_skills(skills: list, assessments: dict, career_score: float) -> float:
    """Skill depth with corroboration multiplier (anti-honeypot)."""
    vals = []
    for sk in skills:
        name = _norm(sk.get("name") or "")
        prof = sk.get("proficiency") or "beginner"
        end  = max(int(sk.get("endorsements") or 0), 0)
        dur  = max(int(sk.get("duration_months") or 0), 0)

        is_core = any(_kw_match(kw, name) for kw in CORE_SKILL_KEYWORDS)
        is_nice = any(_kw_match(kw, name) for kw in NICE_SKILL_KEYWORDS)
        if not (is_core or is_nice):
            continue

        w   = 1.0 if is_core else 0.50
        p   = PROFICIENCY_WEIGHT.get(prof, 0.25)
        df  = min(dur / 36.0, 1.0) ** 0.5
        ef  = min(math.log1p(end) / math.log1p(50), 1.0)

        bonus = 0.0
        for ak, av in assessments.items():
            if _norm(ak) in name or name in _norm(ak):
                v = float(av)
                bonus = 0.20 if v >= 70 else (0.10 if v >= 50 else 0.0)
                break

        trust = p * 0.50 + df * 0.30 + ef * 0.20
        vals.append(w * (trust + bonus))

    if not vals:
        return 0.0

    vals.sort(reverse=True)
    top5 = vals[:5]
    rest = vals[5:]
    raw  = (sum(top5) / len(top5)) * 0.80 + \
           ((sum(rest) / len(rest)) * 0.20 if rest else 0.0)
    raw  = min(raw, 1.0)

    # Corroboration: shrink skills score when career evidence is thin
    # Thresholds raised so more candidates need real career evidence to carry their skill score
    if career_score < 0.08:    raw *= 0.10   # was < 0.06 → ×0.12
    elif career_score < 0.22:  raw *= 0.40   # was < 0.18 → ×0.45
    elif career_score < 0.38:  raw *= 0.68   # was < 0.32 → ×0.72

    return min(raw, 1.0)


def score_experience(years: float) -> float:
    if years < 0:
        return 0.0
    center = 7.0
    sigma  = 2.5 if years < center else 3.5
    s = math.exp(-0.5 * ((years - center) / sigma) ** 2)
    if years < 3.0:    s *= 0.20
    elif years > 15.0: s *= 0.60
    return s


def score_location(profile: dict, signals: dict) -> float:
    loc      = _norm(profile.get("location") or "")
    country  = _norm(profile.get("country") or "")
    relocate = bool(signals.get("willing_to_relocate", False))

    base = 0.28
    if "india" in country:
        base = 0.62
        # Take the BEST matching city, not the first (a location string can name
        # multiple cities or contain city names as substrings — max avoids the
        # bug where dict iteration order decides the winner).
        for city, cs in PREFERRED_CITY_SCORES.items():
            if city in loc:
                base = max(base, cs)

    if relocate:
        base = base + (0.75 - base) * 0.75 if base < 0.70 else min(1.0, base + 0.05)

    return min(base, 1.0)


def score_education(edu: list) -> float:
    if not edu:
        return 0.42
    best = 0.0
    for e in edu:
        tier  = e.get("tier") or "unknown"
        field = _norm(e.get("field_of_study") or "")
        deg   = _norm(e.get("degree") or "")
        ts = EDUCATION_TIER.get(tier, 0.42)
        if any(f in field for f in RELEVANT_FIELDS): ts = min(1.0, ts + 0.10)
        if any(d in deg for d in ["m.tech", "m.s.", "m.e.", "ph.d", "phd"]): ts = min(1.0, ts + 0.05)
        best = max(best, ts)
    return best


def score_availability(signals: dict) -> float:
    open_f = 1.0 if signals.get("open_to_work_flag", False) else 0.28

    days = _days_since(signals.get("last_active_date") or "")
    if days <= 7:      recency = 1.00
    elif days <= 30:   recency = 0.92
    elif days <= 90:   recency = 0.72
    elif days <= 180:  recency = 0.45
    else:              recency = 0.18

    # BUG FIX: use explicit None checks — `0.0 or default` and `0 or default`
    # would replace legitimate zero values (0% response rate, 0-day notice) with
    # the default, giving wrong scores to immediately-available candidates.
    _rr_raw = signals.get("recruiter_response_rate")
    rr = max(0.0, min(float(_rr_raw if _rr_raw is not None else 0.30), 1.0))

    _nd_raw = signals.get("notice_period_days")
    notice = int(_nd_raw) if _nd_raw is not None else 90
    if notice <= 0:      ns = 1.00   # immediately available
    elif notice <= 30:   ns = 1.00
    elif notice <= 60:   ns = 0.72
    elif notice <= 90:   ns = 0.45
    elif notice <= 120:  ns = 0.25
    else:                ns = 0.10

    _ir_raw = signals.get("interview_completion_rate")
    ir = max(0.0, min(float(_ir_raw if _ir_raw is not None else 0.50), 1.0))

    github = signals.get("github_activity_score", -1)
    gb = min(float(github) / 100.0, 1.0) * 0.12 if (github is not None and github >= 0) else 0.0

    return min(open_f * 0.25 + recency * 0.25 + rr * 0.20 + ns * 0.20 + ir * 0.10 + gb, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(candidate: dict) -> tuple:
    profile  = candidate.get("profile") or {}
    career   = candidate.get("career_history") or []
    skills   = candidate.get("skills") or []
    edu      = candidate.get("education") or []
    signals  = candidate.get("redrob_signals") or {}
    assess   = signals.get("skill_assessment_scores") or {}

    career_s, cons_frac, prod_roles = score_career_history(career)
    title_s  = score_title(profile, career)

    # Title is the PRIMARY gate (65%), career evidence is secondary (35%).
    # Raised title weight from 0.60 → 0.65 so non-AI titles can't be rescued
    # purely by dense production keywords in career descriptions.
    tc_score = title_s * 0.65 + career_s * 0.35

    skill_s  = score_skills(skills, assess, career_s)
    years    = float(profile.get("years_of_experience") or 0)
    exp_s    = score_experience(years)
    loc_s    = score_location(profile, signals)
    edu_s    = score_education(edu)

    relevance = (
        tc_score * 0.45 +
        skill_s  * 0.20 +
        exp_s    * 0.15 +
        loc_s    * 0.12 +
        edu_s    * 0.08
    )

    avail = score_availability(signals)
    final = relevance * (0.60 + 0.40 * avail)

    # Hard ceiling: confirmed honeypot titles (title_s == 0.03, no prior AI career)
    # cannot exceed 0.10 regardless of skill stuffing or availability signals.
    if title_s <= 0.03:
        final = min(final, 0.10)

    comps = {
        "title_s": title_s, "career_s": career_s, "tc": tc_score,
        "skill_s": skill_s, "exp_s": exp_s,
        "loc_s": loc_s, "edu_s": edu_s,
        "avail": avail, "relevance": relevance,
        "cons_frac": cons_frac, "prod_roles": prod_roles,
        "years": years,
    }
    return final, comps


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _top_skills(skills: list, n: int = 3) -> list:
    scored = []
    for sk in skills:
        name = _norm(sk.get("name") or "")
        if any(_kw_match(kw, name) for kw in CORE_SKILL_KEYWORDS) or \
           any(_kw_match(kw, name) for kw in NICE_SKILL_KEYWORDS):
            p = PROFICIENCY_WEIGHT.get(sk.get("proficiency") or "beginner", 0.25)
            e = math.log1p(int(sk.get("endorsements") or 0))
            scored.append((p * e, sk.get("name") or name))
    scored.sort(reverse=True)
    return [nm for _, nm in scored[:n]]


def _career_kws_found(career: list, n: int = 2) -> list:
    best, best_kws = 0, []
    for role in career:
        desc  = role.get("description") or ""
        found = [kw for kw in PRODUCTION_KEYWORDS if kw in _norm(desc)]
        if len(found) > best:
            best, best_kws = len(found), found[:n]
    return [k.strip() for k in best_kws]


def generate_reasoning(candidate: dict, comps: dict, rank: int) -> str:
    profile  = candidate.get("profile") or {}
    career   = candidate.get("career_history") or []
    skills   = candidate.get("skills") or []
    signals  = candidate.get("redrob_signals") or {}

    title    = profile.get("current_title") or "Candidate"
    company  = profile.get("current_company") or ""
    loc      = profile.get("location") or ""
    country  = profile.get("country") or ""
    years    = comps["years"]
    notice   = int(signals.get("notice_period_days") or 90)
    rr       = float(signals.get("recruiter_response_rate") or 0.3)
    days_in  = _days_since(signals.get("last_active_date") or "")
    github   = signals.get("github_activity_score", -1)
    title_s  = comps["title_s"]
    cons_f   = comps["cons_frac"]
    prod_r   = comps["prod_roles"]

    top_sk  = _top_skills(skills, 3)
    kws     = _career_kws_found(career, 2)
    loc_str = loc if "india" in (country or "").lower() else f"{loc}, {country}"

    # ── Sentence 1: lead with strongest positive signal ────────────────────
    if title_s >= 0.90:
        s1 = f"{title} with {years:.1f} years"
        if company:
            s1 += f" at {company}"
        if kws:
            s1 += f"; career descriptions reference {' and '.join(kws)}"
        elif prod_r >= 1:
            s1 += f"; {prod_r} role(s) with production ML/AI evidence"
        if top_sk:
            s1 += f"; key skills: {', '.join(top_sk)}"

    elif title_s >= 0.60:
        s1 = f"{title} ({years:.1f} yrs) with ML/AI background"
        if kws:
            s1 += f"; career evidence includes {' and '.join(kws)}"
        if top_sk:
            s1 += f"; relevant skills: {', '.join(top_sk)}"

    elif title_s >= 0.30:
        s1 = f"{title} ({years:.1f} yrs) — adjacent technical role"
        if kws:
            s1 += f"; career references {' and '.join(kws)}"
        elif top_sk:
            s1 += f"; AI-adjacent skills ({', '.join(top_sk)}) need career verification"
        else:
            s1 += "; limited production ML evidence in descriptions"

    else:
        s1 = f"Title ({title}) is not a direct match for this AI Engineering role"
        if kws:
            s1 += f"; career mentions {' and '.join(kws)}"
        else:
            s1 += "; career history lacks retrieval/ranking production evidence"

    # ── Sentence 2: availability + honest gaps, tone by rank ──────────────
    parts = []

    if notice <= 30:
        parts.append(f"notice {notice}d (sub-30, JD preference)")
    elif notice <= 60:
        parts.append(f"notice {notice}d (buyout feasible)")
    else:
        parts.append(f"notice {notice}d (above JD threshold)")

    if rr >= 0.70:
        parts.append(f"response rate {rr:.0%} (high)")
    elif rr >= 0.40:
        parts.append(f"response rate {rr:.0%}")
    else:
        parts.append(f"response rate {rr:.0%} (reachability risk)")

    if days_in <= 14:
        parts.append("active within last 2 weeks")
    elif days_in <= 60:
        parts.append(f"active {days_in}d ago")
    else:
        parts.append(f"inactive {days_in}d (engagement concern)")

    if loc_str.strip():
        parts.append(f"based in {loc_str}")

    if cons_f >= 0.90:
        parts.append("entire career at IT-services/consulting (JD disqualifier)")
    elif cons_f >= 0.60:
        parts.append(f"{cons_f:.0%} consulting background (partial concern)")

    if github is not None and github > 30:
        parts.append(f"GitHub score {int(github)}/100")

    tone_map = [
        (1, 10, "Strong overall fit"),
        (11, 25, "Good fit"),
        (26, 50, "Moderate fit"),
        (51, 75, "Partial fit"),
        (76, 100, "Borderline inclusion"),
    ]
    tone = next(t for lo, hi, t in tone_map if lo <= rank <= hi)
    s2 = f"{tone}: {'; '.join(parts[:4])}."

    reasoning = s1.rstrip(". ") + ". " + s2
    return reasoning[:420] + "..." if len(reasoning) > 420 else reasoning


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION — strictly decreasing, no ties
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scores(ranked: list) -> list:
    """
    ranked is already sorted by descending raw score (ties broken by cid asc).
    Returns the same list with monotonically decreasing scores 0.9900 → 0.1000.
    No ties → validator tie-break check always passes.
    """
    n = len(ranked)
    result = []
    for i, (cid, _, comps, cand) in enumerate(ranked):
        norm = round(0.9900 - (0.8900 / max(n - 1, 1)) * i, 4)
        result.append((cid, norm, comps, cand))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for a Senior AI Engineer role."
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    print(f"Scoring: {args.candidates}", file=sys.stderr)

    if args.candidates.endswith(".gz"):
        import gzip
        opener = lambda: gzip.open(args.candidates, "rt", encoding="utf-8")
    else:
        opener = lambda: open(args.candidates, "r", encoding="utf-8")

    # min-heap keyed by (score, counter, cid) so heap[0] = LOWEST-scoring candidate
    # in the current top-N.  When a new candidate beats heap[0], evict the worst
    # and keep the new one.  counter breaks score ties without comparing dicts.
    heap = []
    _counter = itertools.count()
    processed = errors = 0

    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            cid = cand.get("candidate_id") or ""
            if not cid:
                errors += 1
                continue

            score, comps = compute_score(cand)
            entry = (score, next(_counter), cid, cand, comps)

            if len(heap) < TOP_N:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:   # beats the current worst in top-N
                heapq.heapreplace(heap, entry)

            processed += 1
            if processed % 10000 == 0:
                print(f"  {processed:,} done…", file=sys.stderr)

    print(f"Scored {processed:,} ({errors} errors).", file=sys.stderr)

    if not heap:
        print("ERROR: no candidates scored.", file=sys.stderr)
        sys.exit(1)

    # Sort descending by score; tie-break by cid ascending
    top_list = sorted(heap, key=lambda x: (-x[0], x[2]))
    ranked   = [(cid, score, comps, cand) for score, _, cid, cand, comps in top_list]

    normalized = normalize_scores(ranked)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, (cid, score, comps, cand) in enumerate(normalized, start=1):
            reasoning = generate_reasoning(cand, comps, rank_pos)
            writer.writerow([cid, rank_pos, f"{score:.4f}", reasoning])

    print(f"Written: {len(normalized)} rows → {args.out}", file=sys.stderr)

    print("\nTop 15:", file=sys.stderr)
    for i, (cid, score, comps, cand) in enumerate(normalized[:15], 1):
        prof = cand.get("profile") or {}
        print(
            f"  #{i:2d}  {cid}  "
            f"title={prof.get('current_title','?')!r:35s}  "
            f"yrs={comps['years']:4.1f}  "
            f"ts={comps['title_s']:.2f}  "
            f"cs={comps['career_s']:.2f}  "
            f"av={comps['avail']:.2f}",
            file=sys.stderr
        )


if __name__ == "__main__":
    main()
