# Redrob India Runs Data & AI Challenge — Deepanshu

Deterministic candidate ranker for the Senior AI Engineer role.
Ranks 100,000 candidates in ~70 seconds on CPU with zero external dependencies.

## Run

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## How it works

Two-layer scoring:

```
final = relevance × (0.60 + 0.40 × availability)
```

**Relevance** (six components):

| Component | Weight | Key idea |
|---|---|---|
| Title + Career | 45% | Title is the primary anti-honeypot gate; career keyword scan corroborates |
| Skill depth | 20% | Proficiency × duration × endorsements, shrinks when career evidence is thin |
| Experience years | 15% | Gaussian centred at 7 years (JD ideal: 6–8) |
| Location | 12% | India ≥ 0.62; Pune/Noida = 1.00 |
| Education | 8% | Tier lookup + STEM field bonus |

**Availability multiplier** (swings final score ±40%):
`open_to_work`, `last_active_date` recency, `recruiter_response_rate`,
`notice_period_days`, `interview_completion_rate`, `github_activity_score`

### Anti-honeypot design

The dataset contains ~55K candidates with AI skills but non-AI titles
(HR Managers, Accountants, etc.). Three defences:

1. **Title gate** — TITLE_NEGATIVE set → score 0.03 regardless of skill count
2. **Corroboration multiplier** — skill score shrinks ×0.12 when career evidence is thin
3. **Consulting penalty** — ≥90% career at TCS/Infosys/Wipro etc. → career × 0.05

## Files

| File | Purpose |
|---|---|
| `rank.py` | Single-file ranker, stdlib only |
| `requirements.txt` | Empty — no pip install needed |
| `submission_metadata.yaml` | Hackathon submission metadata |
| `my_submission.csv` | Ranked output (100 rows, validated) |
| `analysis.md` | Full challenge analysis and JD breakdown |

## Constraints met

- Runtime: ~70 seconds for 100K candidates (limit: 5 minutes) ✓
- Memory: < 1 GB peak (streaming, min-heap of 100) (limit: 16 GB) ✓
- CPU only, no GPU, no network ✓
- Pure stdlib — `python rank.py` works on a fresh Python 3.8+ install ✓
