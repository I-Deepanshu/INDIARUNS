"""
app.py — HuggingFace Spaces (Gradio) demo for the Redrob hackathon ranker.

Deploy: push rank.py + app.py + requirements.txt to a HuggingFace Space
        with SDK = Gradio. No other dependencies — rank.py is stdlib only.
"""

import gradio as gr
import json
import importlib.util
import os

# Load rank.py as a module (same directory)
def _load_ranker():
    spec = importlib.util.spec_from_file_location(
        "rank",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "rank.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

rank = _load_ranker()

DEFAULT_JSONL = """\
{"candidate_id":"DEMO_001","profile":{"current_title":"Senior ML Engineer","years_of_experience":7,"location":"Bangalore","country":"India","headline":"ML Engineer specialising in vector search"},"career_history":[{"title":"ML Engineer","company":"Flipkart","duration_months":36,"description":"Built semantic search using FAISS and Pinecone. Implemented reranking with learning-to-rank models. A/B tested retrieval pipelines achieving 15% NDCG improvement. Deployed to production at scale."},{"title":"Data Scientist","company":"Zomato","duration_months":24,"description":"Recommendation system using collaborative filtering and matrix factorization. Deployed ranking model serving 2M queries per day."}],"skills":[{"name":"FAISS","proficiency":"expert","endorsements":42,"duration_months":30},{"name":"Python","proficiency":"expert","endorsements":85,"duration_months":60},{"name":"Elasticsearch","proficiency":"advanced","endorsements":18,"duration_months":24}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"B.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-10","recruiter_response_rate":0.82,"notice_period_days":30,"interview_completion_rate":0.75,"github_activity_score":65,"willing_to_relocate":true}}
{"candidate_id":"DEMO_002","profile":{"current_title":"HR Manager","years_of_experience":6,"location":"Delhi","country":"India","headline":"Human Resources professional"},"career_history":[{"title":"HR Manager","company":"Infosys","duration_months":60,"description":"Managed recruitment for 500+ engineers. Familiar with TensorFlow PyTorch FAISS semantic search NLP RAG pipeline machine learning deep learning reranking NDCG vector search."}],"skills":[{"name":"PyTorch","proficiency":"expert","endorsements":99,"duration_months":48},{"name":"FAISS","proficiency":"expert","endorsements":99,"duration_months":48},{"name":"NLP","proficiency":"expert","endorsements":99,"duration_months":48},{"name":"RAG","proficiency":"expert","endorsements":99,"duration_months":48}],"education":[{"tier":"tier_2","field_of_study":"Human Resources","degree":"MBA"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-15","recruiter_response_rate":0.95,"notice_period_days":0,"interview_completion_rate":0.90,"github_activity_score":5,"willing_to_relocate":false}}
{"candidate_id":"DEMO_003","profile":{"current_title":"Lead AI Engineer","years_of_experience":8,"location":"Pune","country":"India","headline":"Lead AI Engineer building retrieval and ranking systems"},"career_history":[{"title":"Lead AI Engineer","company":"Meesho","duration_months":24,"description":"Led vector search platform using Qdrant and FAISS. Designed hybrid retrieval with dense and sparse vectors. Built cross-encoder reranking pipeline. NDCG improved 22% via learning-to-rank."},{"title":"Senior ML Engineer","company":"Swiggy","duration_months":36,"description":"Built recommendation engine with two-tower model. Fine-tuned sentence-transformer for semantic search. Deployed RAG pipeline with retrieval augmented generation at scale."}],"skills":[{"name":"Qdrant","proficiency":"expert","endorsements":38,"duration_months":24},{"name":"Sentence Transformers","proficiency":"expert","endorsements":45,"duration_months":36},{"name":"RAG","proficiency":"expert","endorsements":30,"duration_months":18},{"name":"Python","proficiency":"expert","endorsements":90,"duration_months":60}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"M.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-15","recruiter_response_rate":0.85,"notice_period_days":30,"interview_completion_rate":0.88,"github_activity_score":80,"willing_to_relocate":false}}
{"candidate_id":"DEMO_004","profile":{"current_title":"NLP Engineer","years_of_experience":5,"location":"Noida","country":"India","headline":"NLP Engineer focused on retrieval and ranking"},"career_history":[{"title":"NLP Engineer","company":"Sarvam AI","duration_months":30,"description":"Developed RAG pipeline with Qdrant. Fine-tuned LLaMA with LoRA/QLoRA. Built sentence-transformer based semantic search. Offline evaluation using NDCG and MRR."},{"title":"ML Engineer","company":"Freshworks","duration_months":24,"description":"Built recommendation engine with collaborative filtering. Deployed to production serving 1M+ queries per day."}],"skills":[{"name":"Qdrant","proficiency":"expert","endorsements":30,"duration_months":24},{"name":"Sentence Transformers","proficiency":"advanced","endorsements":22,"duration_months":20},{"name":"LoRA","proficiency":"advanced","endorsements":15,"duration_months":18}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"M.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-14","recruiter_response_rate":0.78,"notice_period_days":0,"interview_completion_rate":0.80,"github_activity_score":72,"willing_to_relocate":true}}
{"candidate_id":"DEMO_005","profile":{"current_title":"Java Developer","years_of_experience":6,"location":"Mumbai","country":"India","headline":"Java backend developer with AI knowledge"},"career_history":[{"title":"Java Developer","company":"TCS","duration_months":72,"description":"Developed microservices in Spring Boot. Worked on REST APIs and database optimization. Familiar with machine learning concepts and PyTorch. Worked on storage systems and leverage frameworks."}],"skills":[{"name":"PyTorch","proficiency":"expert","endorsements":95,"duration_months":60},{"name":"FAISS","proficiency":"expert","endorsements":88,"duration_months":48},{"name":"Storage Systems","proficiency":"expert","endorsements":92,"duration_months":48}],"education":[{"tier":"tier_2","field_of_study":"Computer Science","degree":"B.E."}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-15","recruiter_response_rate":0.90,"notice_period_days":30,"interview_completion_rate":0.85,"github_activity_score":20,"willing_to_relocate":true}}
{"candidate_id":"DEMO_006","profile":{"current_title":"Applied Scientist","years_of_experience":6,"location":"Hyderabad","country":"India","headline":"Applied Scientist in ranking and recommendation"},"career_history":[{"title":"Applied Scientist","company":"Amazon","duration_months":48,"description":"Built learning-to-rank models for product search. Developed two-tower retrieval model. Offline evaluation with NDCG and MAP. Deployed reranking pipeline handling 50M queries per day. Feature engineering with XGBoost and LightGBM."},{"title":"Data Scientist","company":"Myntra","duration_months":24,"description":"Recommendation engine using collaborative filtering. A/B tested ranking algorithms improving NDCG@10 by 18%."}],"skills":[{"name":"Learning to Rank","proficiency":"expert","endorsements":55,"duration_months":48},{"name":"XGBoost","proficiency":"expert","endorsements":40,"duration_months":42},{"name":"FAISS","proficiency":"advanced","endorsements":28,"duration_months":36},{"name":"Python","proficiency":"expert","endorsements":100,"duration_months":60}],"education":[{"tier":"tier_1","field_of_study":"Statistics","degree":"M.Sc"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-15","recruiter_response_rate":0.80,"notice_period_days":60,"interview_completion_rate":0.82,"github_activity_score":58,"willing_to_relocate":false}}
{"candidate_id":"DEMO_007","profile":{"current_title":"Search Engineer","years_of_experience":7,"location":"Bangalore","country":"India","headline":"Search and relevance engineer with IR background"},"career_history":[{"title":"Search Engineer","company":"Myntra","duration_months":42,"description":"Built product search using Elasticsearch and BM25. Implemented semantic search with sentence-transformers. Designed hybrid retrieval combining dense and sparse vectors. Evaluated with NDCG MRR and precision. Led migration from keyword to neural retrieval."},{"title":"Backend Engineer","company":"Grofers","duration_months":30,"description":"Recommendation system for grocery search using collaborative filtering and item2vec. Deployed to production serving 800K daily active users."}],"skills":[{"name":"Elasticsearch","proficiency":"expert","endorsements":62,"duration_months":42},{"name":"Sentence Transformers","proficiency":"expert","endorsements":35,"duration_months":30},{"name":"BM25","proficiency":"expert","endorsements":28,"duration_months":36},{"name":"Python","proficiency":"expert","endorsements":88,"duration_months":66}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"B.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-12","recruiter_response_rate":0.75,"notice_period_days":30,"interview_completion_rate":0.78,"github_activity_score":75,"willing_to_relocate":true}}
{"candidate_id":"DEMO_008","profile":{"current_title":"Senior Data Scientist","years_of_experience":6,"location":"Pune","country":"India","headline":"Data Scientist building recommendation and search systems"},"career_history":[{"title":"Senior Data Scientist","company":"PhonePe","duration_months":36,"description":"Built semantic search using FAISS and Qdrant. Developed recommendation system with two-tower model. Fine-tuned sentence-transformer for payments search. Deployed to production handling 10M daily queries."},{"title":"Data Scientist","company":"Paytm","duration_months":30,"description":"Embedding-based retrieval for merchant search. NDCG improved 14% after neural ranker deployment."}],"skills":[{"name":"Qdrant","proficiency":"advanced","endorsements":22,"duration_months":24},{"name":"Sentence Transformers","proficiency":"advanced","endorsements":18,"duration_months":30},{"name":"FAISS","proficiency":"advanced","endorsements":30,"duration_months":36},{"name":"Python","proficiency":"expert","endorsements":75,"duration_months":60}],"education":[{"tier":"tier_2","field_of_study":"Computer Science","degree":"M.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-10","recruiter_response_rate":0.70,"notice_period_days":30,"interview_completion_rate":0.72,"github_activity_score":45,"willing_to_relocate":false}}
{"candidate_id":"DEMO_009","profile":{"current_title":"Research Engineer","years_of_experience":5,"location":"Chennai","country":"India","headline":"Research Engineer in information retrieval"},"career_history":[{"title":"Research Engineer","company":"Wadhwani AI","duration_months":36,"description":"Built dense retrieval using ColBERT and SPLADE. Evaluated on BEIR benchmark. Implemented late interaction models for passage retrieval. Published research on hybrid retrieval."},{"title":"Research Intern","company":"Microsoft Research India","duration_months":18,"description":"Developed neural retrieval models. Implemented cross-encoder reranking. Achieved strong NDCG on TREC Deep Learning track."}],"skills":[{"name":"ColBERT","proficiency":"expert","endorsements":15,"duration_months":24},{"name":"SPLADE","proficiency":"advanced","endorsements":10,"duration_months":18},{"name":"Sentence Transformers","proficiency":"expert","endorsements":20,"duration_months":36},{"name":"PyTorch","proficiency":"expert","endorsements":45,"duration_months":42}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"M.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-08","recruiter_response_rate":0.65,"notice_period_days":90,"interview_completion_rate":0.68,"github_activity_score":90,"willing_to_relocate":true}}
{"candidate_id":"DEMO_010","profile":{"current_title":"Software Engineer","years_of_experience":5,"location":"Gurgaon","country":"India","headline":"Software Engineer with NLP and search focus"},"career_history":[{"title":"Software Engineer","company":"Microsoft","duration_months":36,"description":"Built NLP pipeline for entity extraction and semantic search. Implemented text embedding service using sentence-transformers. Contributed to search ranking features using Elasticsearch and vector search."},{"title":"Junior Software Engineer","company":"Freshworks","duration_months":24,"description":"Chatbot with intent classification using BERT. Deployed NLP model serving 500K daily queries."}],"skills":[{"name":"NLP","proficiency":"advanced","endorsements":30,"duration_months":36},{"name":"Elasticsearch","proficiency":"advanced","endorsements":25,"duration_months":30},{"name":"Python","proficiency":"expert","endorsements":70,"duration_months":60}],"education":[{"tier":"tier_2","field_of_study":"Computer Science","degree":"B.E."}],"redrob_signals":{"open_to_work_flag":false,"last_active_date":"2026-05-20","recruiter_response_rate":0.55,"notice_period_days":60,"interview_completion_rate":0.60,"github_activity_score":35,"willing_to_relocate":true}}
{"candidate_id":"DEMO_011","profile":{"current_title":"Chartered Accountant","years_of_experience":8,"location":"Delhi","country":"India","headline":"Finance professional with AI interests"},"career_history":[{"title":"Senior Accountant","company":"Deloitte","duration_months":60,"description":"Financial reporting and audit management. Very familiar with PyTorch TensorFlow FAISS Qdrant vector search semantic search NDCG reranking RAG pipeline machine learning deep learning embeddings sentence-transformers."},{"title":"Accountant","company":"KPMG","duration_months":36,"description":"Tax compliance and financial analysis. Knowledge of AI machine learning NLP neural networks transformers BERT LLM RAG retrieval augmented generation."}],"skills":[{"name":"PyTorch","proficiency":"expert","endorsements":99,"duration_months":60},{"name":"FAISS","proficiency":"expert","endorsements":99,"duration_months":48},{"name":"RAG","proficiency":"expert","endorsements":99,"duration_months":36}],"education":[{"tier":"tier_1","field_of_study":"Chartered Accountancy","degree":"CA"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-16","recruiter_response_rate":0.95,"notice_period_days":0,"interview_completion_rate":0.95,"github_activity_score":2,"willing_to_relocate":true}}
{"candidate_id":"DEMO_012","profile":{"current_title":"Digital Marketing Manager","years_of_experience":7,"location":"Mumbai","country":"India","headline":"Digital marketing professional passionate about AI"},"career_history":[{"title":"Digital Marketing Manager","company":"Zomato","duration_months":48,"description":"Managed campaigns and grew user acquisition by 40%. Expert in SEO SEM content marketing. Also worked on AI machine learning NLP RAG pipeline FAISS vector search NDCG semantic search embeddings PyTorch TensorFlow reranking recommendation systems."},{"title":"SEO Specialist","company":"MakeMyTrip","duration_months":36,"description":"Optimised search rankings for organic growth. Knowledge of AI algorithms machine learning neural networks information retrieval."}],"skills":[{"name":"Machine Learning","proficiency":"expert","endorsements":85,"duration_months":48},{"name":"NLP","proficiency":"expert","endorsements":90,"duration_months":36},{"name":"FAISS","proficiency":"expert","endorsements":88,"duration_months":24}],"education":[{"tier":"tier_2","field_of_study":"Marketing","degree":"MBA"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-15","recruiter_response_rate":0.92,"notice_period_days":30,"interview_completion_rate":0.88,"github_activity_score":0,"willing_to_relocate":true}}
{"candidate_id":"DEMO_013","profile":{"current_title":"ML Engineer","years_of_experience":7,"location":"Bangalore","country":"India","headline":"ML Engineer at Wipro building AI solutions for clients"},"career_history":[{"title":"ML Engineer","company":"Wipro","duration_months":72,"description":"Deployed machine learning models for banking clients. Built NLP pipeline for document classification. Implemented text classification using BERT fine-tuning. Built recommendation API for retail client. Experience with Python scikit-learn and neural networks."}],"skills":[{"name":"Python","proficiency":"expert","endorsements":45,"duration_months":60},{"name":"Scikit-learn","proficiency":"expert","endorsements":30,"duration_months":48},{"name":"BERT","proficiency":"intermediate","endorsements":12,"duration_months":24}],"education":[{"tier":"tier_3","field_of_study":"Computer Science","degree":"B.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-14","recruiter_response_rate":0.72,"notice_period_days":60,"interview_completion_rate":0.65,"github_activity_score":18,"willing_to_relocate":false}}
{"candidate_id":"DEMO_014","profile":{"current_title":"Senior ML Engineer","years_of_experience":8,"location":"Singapore","country":"Singapore","headline":"Senior ML Engineer open to India roles"},"career_history":[{"title":"Senior ML Engineer","company":"Grab","duration_months":48,"description":"Built recommendation system for ride-hailing using two-tower model. Developed semantic search with FAISS and dense retrieval. Reranking with learning-to-rank. NDCG improved 20% after neural ranking deployment."},{"title":"ML Engineer","company":"Sea Group","duration_months":36,"description":"Product recommendation and semantic search using FAISS. Deployed vector search to production serving 5M daily users."}],"skills":[{"name":"FAISS","proficiency":"expert","endorsements":40,"duration_months":48},{"name":"Python","proficiency":"expert","endorsements":80,"duration_months":72},{"name":"Sentence Transformers","proficiency":"advanced","endorsements":22,"duration_months":30}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"M.S."}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-15","recruiter_response_rate":0.78,"notice_period_days":60,"interview_completion_rate":0.80,"github_activity_score":55,"willing_to_relocate":true}}
{"candidate_id":"DEMO_015","profile":{"current_title":"Principal AI Engineer","years_of_experience":9,"location":"Noida","country":"India","headline":"Principal AI Engineer leading vector search infrastructure"},"career_history":[{"title":"Principal AI Engineer","company":"Razorpay","duration_months":36,"description":"Led vector search infrastructure using Qdrant and FAISS. Designed hybrid retrieval with dense and sparse vectors. Built cross-encoder reranking service evaluated with NDCG MRR MAP. Manages team of 5 ML engineers."},{"title":"Senior ML Engineer","company":"Flipkart","duration_months":42,"description":"Built semantic product search and recommendation engine. Two-tower retrieval model. Deployed at scale serving 20M queries per day. Improved NDCG@10 by 25%."}],"skills":[{"name":"Qdrant","proficiency":"expert","endorsements":50,"duration_months":36},{"name":"FAISS","proficiency":"expert","endorsements":65,"duration_months":48},{"name":"Learning to Rank","proficiency":"expert","endorsements":40,"duration_months":42},{"name":"Python","proficiency":"expert","endorsements":120,"duration_months":84}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"M.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-16","recruiter_response_rate":0.88,"notice_period_days":30,"interview_completion_rate":0.90,"github_activity_score":70,"willing_to_relocate":false}}
{"candidate_id":"DEMO_016","profile":{"current_title":"Generative AI Engineer","years_of_experience":4,"location":"Bangalore","country":"India","headline":"Generative AI engineer building RAG and LLM systems"},"career_history":[{"title":"Generative AI Engineer","company":"Sarvam AI","duration_months":24,"description":"Built RAG pipeline with Qdrant and sentence-transformers for Indian language search. Implemented chunking strategy and hybrid retrieval with dense and sparse vectors. Fine-tuned LLaMA with LoRA. Offline evaluation using NDCG and faithfulness metrics."},{"title":"ML Engineer","company":"Unacademy","duration_months":24,"description":"Semantic search for educational content using FAISS. Question recommendation using embedding similarity."}],"skills":[{"name":"RAG","proficiency":"expert","endorsements":35,"duration_months":24},{"name":"Qdrant","proficiency":"expert","endorsements":28,"duration_months":24},{"name":"LLM","proficiency":"expert","endorsements":30,"duration_months":20},{"name":"Python","proficiency":"expert","endorsements":72,"duration_months":48}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"B.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-16","recruiter_response_rate":0.82,"notice_period_days":30,"interview_completion_rate":0.85,"github_activity_score":82,"willing_to_relocate":true}}
{"candidate_id":"DEMO_017","profile":{"current_title":"RAG Engineer","years_of_experience":3,"location":"Delhi","country":"India","headline":"RAG engineer and LLM specialist"},"career_history":[{"title":"RAG Engineer","company":"Krutrim","duration_months":18,"description":"Designed RAG architecture with hybrid retrieval using dense and sparse vectors. Implemented chunking strategy and cross-encoder reranking. Built agentic RAG pipeline. Evaluated retrieval quality using NDCG and MRR."},{"title":"ML Engineer","company":"Haptik","duration_months":18,"description":"NLP pipeline for intent detection. Deployed BERT-based classifier to production serving 300K daily queries."}],"skills":[{"name":"RAG","proficiency":"expert","endorsements":25,"duration_months":18},{"name":"Qdrant","proficiency":"advanced","endorsements":20,"duration_months":18},{"name":"LLM","proficiency":"advanced","endorsements":22,"duration_months":15},{"name":"Python","proficiency":"expert","endorsements":55,"duration_months":36}],"education":[{"tier":"tier_2","field_of_study":"Computer Science","degree":"B.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-15","recruiter_response_rate":0.78,"notice_period_days":30,"interview_completion_rate":0.75,"github_activity_score":70,"willing_to_relocate":true}}
{"candidate_id":"DEMO_018","profile":{"current_title":"Senior ML Engineer","years_of_experience":13,"location":"Hyderabad","country":"India","headline":"Senior ML Engineer with 13 years in AI and data"},"career_history":[{"title":"Senior ML Engineer","company":"Google India","duration_months":60,"description":"Machine learning for ads ranking and recommendation. Feature engineering and model deployment at scale. Some exposure to dense retrieval and semantic search using TensorFlow and FAISS."},{"title":"Data Scientist","company":"Uber India","duration_months":60,"description":"Demand forecasting and pricing models. Recommendation engine for driver matching. Statistical modelling and A/B testing at scale."},{"title":"Data Analyst","company":"Housing.com","duration_months":36,"description":"Data analysis A/B testing and basic NLP for property search."}],"skills":[{"name":"Python","proficiency":"expert","endorsements":90,"duration_months":120},{"name":"TensorFlow","proficiency":"expert","endorsements":60,"duration_months":84},{"name":"FAISS","proficiency":"intermediate","endorsements":15,"duration_months":24}],"education":[{"tier":"tier_1","field_of_study":"Statistics","degree":"M.Sc"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-05","recruiter_response_rate":0.65,"notice_period_days":90,"interview_completion_rate":0.60,"github_activity_score":22,"willing_to_relocate":false}}
{"candidate_id":"DEMO_019","profile":{"current_title":"ML Engineer","years_of_experience":2,"location":"Bangalore","country":"India","headline":"Junior ML Engineer eager to build AI systems"},"career_history":[{"title":"ML Engineer","company":"Sprinklr","duration_months":18,"description":"Built text classification models using BERT. Basic NLP pipeline development. Some exposure to semantic search and sentence-transformers for document retrieval."}],"skills":[{"name":"Python","proficiency":"intermediate","endorsements":15,"duration_months":18},{"name":"NLP","proficiency":"beginner","endorsements":5,"duration_months":12},{"name":"Sentence Transformers","proficiency":"beginner","endorsements":3,"duration_months":8}],"education":[{"tier":"tier_2","field_of_study":"Computer Science","degree":"B.Tech"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-15","recruiter_response_rate":0.80,"notice_period_days":30,"interview_completion_rate":0.70,"github_activity_score":30,"willing_to_relocate":true}}
{"candidate_id":"DEMO_020","profile":{"current_title":"Senior NLP Scientist","years_of_experience":8,"location":"Noida","country":"India","headline":"Senior NLP Scientist with IR and search expertise"},"career_history":[{"title":"Senior NLP Scientist","company":"Samsung Research India","duration_months":48,"description":"Built information retrieval systems using hybrid search with dense and sparse vectors. Developed dense retrieval with sentence-transformers. Cross-encoder reranking. Evaluated on NDCG MRR and MAP. Led team of 4 NLP engineers on the search platform."},{"title":"NLP Engineer","company":"Juspay","duration_months":36,"description":"NLP pipeline for payment intent classification. Semantic search using FAISS. Deployed to production handling 2M daily transactions."}],"skills":[{"name":"Sentence Transformers","proficiency":"expert","endorsements":45,"duration_months":42},{"name":"FAISS","proficiency":"expert","endorsements":38,"duration_months":36},{"name":"BM25","proficiency":"expert","endorsements":25,"duration_months":30},{"name":"Python","proficiency":"expert","endorsements":95,"duration_months":72}],"education":[{"tier":"tier_1","field_of_study":"Computer Science","degree":"Ph.D"}],"redrob_signals":{"open_to_work_flag":true,"last_active_date":"2026-06-16","recruiter_response_rate":0.85,"notice_period_days":30,"interview_completion_rate":0.88,"github_activity_score":65,"willing_to_relocate":false}}
"""


def rank_candidates(jsonl_text: str):
    """Parse JSONL input, score each candidate, return results table + summary."""
    lines = [l.strip() for l in (jsonl_text or "").strip().split("\n") if l.strip()]

    candidates, parse_errors = [], []
    for i, line in enumerate(lines, 1):
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as e:
            parse_errors.append(f"Line {i}: {e}")

    if parse_errors:
        return [], f"Parse errors:\n" + "\n".join(parse_errors)

    if not candidates:
        return [], "No candidates found. Paste at least one JSON line."

    scored = []
    for c in candidates:
        score, comps = rank.compute_score(c)
        scored.append((score, c, comps))

    scored.sort(key=lambda x: -x[0])

    rows = []
    for pos, (score, cand, comps) in enumerate(scored, 1):
        prof    = cand.get("profile") or {}
        signals = cand.get("redrob_signals") or {}
        reasoning = rank.generate_reasoning(cand, comps, pos)
        rows.append([
            pos,
            cand.get("candidate_id", "?"),
            prof.get("current_title", "?"),
            round(comps["years"], 1),
            round(comps["title_s"], 2),
            round(comps["career_s"], 2),
            round(comps["skill_s"], 2),
            round(comps["avail"], 2),
            round(score, 4),
            reasoning,
        ])

    summary_lines = []
    for pos, row in enumerate(rows, 1):
        title_s = row[4]
        flag = "HONEYPOT" if title_s <= 0.05 else ("low-title" if title_s < 0.30 else "OK")
        summary_lines.append(f"#{pos:2d}  {row[1]}  {row[2]:<35s}  score={row[8]:.4f}  title={title_s:.2f}  [{flag}]")

    summary = f"Ranked {len(rows)} candidates\n\n" + "\n".join(summary_lines)
    return rows, summary


HEADERS = ["Rank", "ID", "Title", "Years", "Title Score", "Career Score",
           "Skill Score", "Avail", "Final Score", "Reasoning"]

with gr.Blocks(title="Redrob AI Candidate Ranker", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# Redrob AI Candidate Ranker
**India Runs Data & AI Challenge 2026** — Submission by Deepanshu

Ranks candidates for a **Senior AI Engineer** role using a deterministic two-layer scorer
(stdlib only, no ML frameworks). Demonstrates the **anti-honeypot gate**: an HR Manager with
every AI skill listed at "expert" still scores <= 0.10 — capped by the title gate.

Twenty demo candidates: strong AI titles (Senior ML, Lead AI, Principal AI, NLP, Search, Applied Scientist, Generative AI, RAG Engineer, Senior NLP Scientist) | honeypots (HR Manager, Chartered Accountant, Marketing Manager) | consulting-penalised (Wipro 100%, TCS 100%) | adjacent title (Software Engineer) | abroad (Singapore, willing to relocate) | overqualified (13 yrs) | junior (2 yrs).
""")

    with gr.Row():
        with gr.Column(scale=2):
            inp = gr.Textbox(
                label="Paste candidates.jsonl (one JSON object per line)",
                value=DEFAULT_JSONL,
                lines=12,
                placeholder="Paste candidate JSON lines here...",
            )
            run_btn = gr.Button("Rank Candidates", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("""
### How scoring works

```
tc      = title x 0.65 + career x 0.35
relevance = tc x 0.45 + skill x 0.20
          + exp x 0.15 + loc x 0.12 + edu x 0.08
final   = relevance x (0.60 + 0.40 x avail)
```

**Relevance weights:**
| Signal | Weight |
|--------|--------|
| Title + Career (tc) | 45% |
| Skill depth | 20% |
| Experience yrs | 15% |
| Location | 12% |
| Education | 8% |

**4-level anti-honeypot defence:**
1. **Title gate** — 27+ negative titles -> 0.03, hard ceiling 0.10
2. **Corroboration** — skills x0.10 when career evidence thin
3. **Consulting penalty** — HCL/TCS/Infosys >=90% -> career x0.05
4. **Word-boundary guards** — "rag" won't match "storage"; "llm" won't match "fulfillment"
""")

    summary_out = gr.Textbox(label="Summary", lines=10, interactive=False)

    table_out = gr.Dataframe(
        headers=HEADERS,
        label="Ranked Candidates",
        wrap=True,
        interactive=False,
    )

    run_btn.click(
        fn=rank_candidates,
        inputs=[inp],
        outputs=[table_out, summary_out],
    )

    gr.Markdown("""
---
*rank.py uses Python stdlib only — json, heapq, math, csv, datetime, itertools.
Runtime for 100K candidates: ~70 seconds on CPU. Zero pip install required.
Top-15 on full dataset: all ts=1.00 — Senior/Lead/Staff ML/NLP/AI/Search Engineers.*
""")

if __name__ == "__main__":
    demo.launch()
