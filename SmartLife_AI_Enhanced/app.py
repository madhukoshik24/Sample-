import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------- CONFIGURATION --------------------
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CATALOG_DIR = DATA_DIR / "catalogs"
PROFILES_FILE = DATA_DIR / "profiles.json"
FEEDBACK_FILE = DATA_DIR / "feedback.csv"

st.set_page_config(
    page_title="SmartLife AI | Recommendation Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --sl-accent: #7557ff; }
.block-container {padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse at 80% 0%, rgba(119,87,255,.12), transparent 36%), linear-gradient(180deg, rgba(18,185,205,.035), transparent 440px);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #f4f2ff 0%, #f7fbff 100%); border-right:1px solid rgba(117,87,255,.16);}
.hero {padding: 2rem 2.1rem; border-radius: 26px; background: linear-gradient(115deg,#17152f 0%,#4831a8 48%,#7757ff 75%,#00a8b5 120%); color: white; margin-bottom: 1.3rem; box-shadow: 0 16px 40px rgba(76,54,180,.18); position:relative; overflow:hidden;}
.hero h1 {color:white; font-size:2.6rem; margin:0; letter-spacing:-.04em;}
.hero p {color:#ece9ff; margin:.55rem 0 0 0; font-size:1.04rem; max-width:850px;}
.hero .pill {display:inline-block; margin-top:1rem; padding:.35rem .7rem; border:1px solid rgba(255,255,255,.28); border-radius:999px; color:#fff; background:rgba(255,255,255,.10); font-size:.8rem;}
.section-card {padding:1.1rem 1.2rem; border:1px solid rgba(117,87,255,.18); border-radius:18px; background:rgba(255,255,255,.78); margin:.4rem 0 .8rem 0; box-shadow:0 5px 20px rgba(33,25,80,.035);}
div[data-testid="stMetric"] {background:linear-gradient(145deg,rgba(255,255,255,.94),rgba(239,236,255,.78)); padding:15px 16px; border-radius:17px; border:1px solid rgba(117,87,255,.16); box-shadow:0 6px 18px rgba(38,25,95,.04);}
div[data-testid="stMetric"] label {font-weight:600;}
div.stButton > button, div[data-testid="stFormSubmitButton"] button {border-radius:12px; font-weight:650; transition:all .18s ease; border:1px solid rgba(117,87,255,.25);}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {border-color:#7557ff; color:#5b3de5; box-shadow:0 5px 16px rgba(117,87,255,.15); transform:translateY(-1px);}
[data-testid="stTabs"] button {font-weight:650;}
[data-testid="stDataFrame"] {border:1px solid rgba(117,87,255,.15); border-radius:14px; overflow:hidden;}
.small-muted {color:#777; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

DATA_DIR.mkdir(exist_ok=True)
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

MODULES = {
    "Products": {
        "file": "products.csv",
        "title": "Smart Product Recommendations",
        "emoji": "🛍️",
        "query_help": "Describe what you need, e.g. 'affordable study tools for programming'",
    },
    "Movies": {
        "file": "movies.csv",
        "title": "Movie Discovery",
        "emoji": "🎬",
        "query_help": "Describe genres, themes, or films you enjoy",
    },
    "Career & Courses": {
        "file": "careers.csv",
        "title": "Career & Course Guidance",
        "emoji": "🎓",
        "query_help": "Describe your skills, subjects you enjoy, and career interests",
    },
    "Travel": {
        "file": "travel.csv",
        "title": "Travel Discovery",
        "emoji": "✈️",
        "query_help": "Describe activities, scenery, and the kind of trip you want",
    },
}
TEXT_COLUMNS = {
    "Products": ["name", "category", "tags", "description"],
    "Movies": ["name", "genre", "tags", "description"],
    "Career & Courses": ["name", "area", "skills", "interests", "courses", "tags", "description"],
    "Travel": ["name", "type", "tags", "description"],
}
DEFAULT_PROFILE = {"interests": "", "created_at": ""}

def load_catalog(module):
    path = CATALOG_DIR / MODULES[module]["file"]
    if not path.exists():
        st.error(f"Catalog file not found: {path}. Re-extract the complete project ZIP.")
        st.stop()
    df = pd.read_csv(path)
    for col in df.columns:
        if col in {"price", "daily_budget", "days", "rating", "popularity", "year"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.fillna({"tags": "", "description": ""})

def load_profiles():
    if PROFILES_FILE.exists():
        try:
            data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def save_profiles(profiles):
    PROFILES_FILE.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")

def load_feedback():
    columns = ["profile", "module", "item", "feedback", "timestamp"]
    if FEEDBACK_FILE.exists():
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            for c in columns:
                if c not in df.columns:
                    df[c] = ""
            return df[columns]
        except Exception:
            pass
    return pd.DataFrame(columns=columns)

def record_feedback(profile, module, item, value):
    df = load_feedback()
    # Keep the most recent judgement per profile/module/item.
    mask = (df["profile"].astype(str) == profile) & (df["module"].astype(str) == module) & (df["item"].astype(str) == item)
    df = df.loc[~mask].copy()
    row = pd.DataFrame([{
        "profile": profile,
        "module": module,
        "item": item,
        "feedback": int(value),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }])
    pd.concat([df, row], ignore_index=True).to_csv(FEEDBACK_FILE, index=False)

def text_for_row(row, module):
    parts = []
    for col in TEXT_COLUMNS[module]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    return " ".join(parts)

def minmax(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    if len(s) == 0:
        return np.array([])
    low, high = s.min(), s.max()
    if high == low:
        return np.full(len(s), 0.5)
    return ((s - low) / (high - low)).to_numpy()

def cosine_scores(descriptions, query):
    if not descriptions:
        return np.array([])
    corpus = [str(x) for x in descriptions] + [str(query or "")]
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
        return cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except ValueError:
        return np.zeros(len(descriptions))

def make_profile_text(profile, feedback, module, catalog):
    chunks = [str(profile.get("interests", "")).strip()]
    liked = feedback[
        (feedback["profile"].astype(str) == str(st.session_state.profile_name)) &
        (feedback["module"].astype(str) == module) &
        (pd.to_numeric(feedback["feedback"], errors="coerce") > 0)
    ]
    if not liked.empty:
        names = set(liked["item"].astype(str))
        # Use the full catalog so liked items still influence recommendations after filters change.
        full_catalog = load_catalog(module)
        liked_rows = full_catalog[full_catalog["name"].astype(str).isin(names)]
        chunks.extend(text_for_row(row, module) for _, row in liked_rows.iterrows())
    return " ".join(x for x in chunks if x)

def hybrid_recommend(catalog, module, query, profile, feedback, top_n=8, focus="Balanced"):
    """Hybrid rank = query content + profile/liked-item affinity + rating + popularity."""
    df = catalog.copy().reset_index(drop=True)
    descriptions = [text_for_row(row, module) for _, row in df.iterrows()]
    query_scores = cosine_scores(descriptions, query)
    profile_text = make_profile_text(profile, feedback, module, df)
    profile_scores = cosine_scores(descriptions, profile_text) if profile_text.strip() else np.zeros(len(df))

    rating_col = "rating" if "rating" in df.columns else None
    popularity_col = "popularity" if "popularity" in df.columns else None
    rating_scores = minmax(df[rating_col]) if rating_col else np.zeros(len(df))
    popularity_scores = minmax(df[popularity_col]) if popularity_col else np.zeros(len(df))

    has_profile = bool(profile_text.strip())
    if has_profile:
        weights = {"query": 0.50, "profile": 0.25, "rating": 0.15, "popularity": 0.10}
    else:
        weights = {"query": 0.70, "profile": 0.00, "rating": 0.20, "popularity": 0.10}

    # User-selectable ranking focus makes the recommender more useful for different goals.
    if focus == "Best match":
        weights = {"query": 0.75, "profile": 0.15 if has_profile else 0.0, "rating": 0.07, "popularity": 0.03}
    elif focus == "Top rated":
        weights = {"query": 0.25, "profile": 0.10 if has_profile else 0.0, "rating": 0.55, "popularity": 0.10}
    elif focus == "Popular picks":
        weights = {"query": 0.30, "profile": 0.10 if has_profile else 0.0, "rating": 0.15, "popularity": 0.45}
    elif focus == "Personalized":
        weights = {"query": 0.35, "profile": 0.45 if has_profile else 0.0, "rating": 0.12, "popularity": 0.08}
    elif focus == "Budget friendly":
        weights = {"query": 0.40, "profile": 0.10 if has_profile else 0.0, "rating": 0.15, "popularity": 0.05}
        if module in {"Products", "Travel"}:
            cost_col = "price" if module == "Products" else "daily_budget"
            costs = pd.to_numeric(df[cost_col], errors="coerce").fillna(0)
            cost_score = 1 - minmax(costs)
        else:
            cost_score = np.zeros(len(df))
        weights["cost"] = 0.30
    else:
        cost_score = np.zeros(len(df))
    df["Content score"] = np.round(query_scores * 100, 1)
    df["Profile affinity"] = np.round(profile_scores * 100, 1)
    weighted = (
        weights.get("query", 0) * query_scores +
        weights.get("profile", 0) * profile_scores +
        weights.get("rating", 0) * rating_scores +
        weights.get("popularity", 0) * popularity_scores +
        weights.get("cost", 0) * cost_score
    )
    # Normalize active weights so scores remain comparable on a 0–100 scale.
    active_weight = sum(weights.values()) or 1
    df["Hybrid score"] = np.round(100 * weighted / active_weight, 1)
    df["Reason"] = [
        make_reason(row, module, query, float(df.loc[i, "Content score"]), has_profile)
        for i, (_, row) in enumerate(df.iterrows())
    ]
    if focus == "Budget friendly" and module in {"Products", "Travel"}:
        cost_col = "price" if module == "Products" else "daily_budget"
        df = df.sort_values(["Hybrid score", cost_col], ascending=[False, True])
    else:
        df = df.sort_values(["Hybrid score", "rating", "popularity"], ascending=False)
    df = df.head(top_n)
    return df, weights

def make_reason(row, module, query, score, has_profile):
    name = str(row.get("name", "This item"))
    parts = []
    if score >= 35:
        parts.append("its description closely matches your request")
    elif score >= 10:
        parts.append("it shares some terms with your request")
    else:
        parts.append("it is included based on its overall quality and ranking")
    if has_profile:
        parts.append("your saved interests and positive feedback also influence its rank")
    if "rating" in row.index:
        parts.append(f"it has a catalog rating of {row['rating']}")
    return f"{name} is recommended because " + "; ".join(parts) + "."

def apply_filters(df, module, budget, category, min_rating, trip_days):
    out = df.copy()
    if module == "Products":
        if budget is not None:
            out = out[out["price"] <= budget]
        if category and category != "Any":
            out = out[out["category"] == category]
    elif module == "Movies":
        if category and category != "Any":
            out = out[out["genre"] == category]
    elif module == "Career & Courses":
        if category and category != "Any":
            out = out[out["area"] == category]
    elif module == "Travel":
        if budget is not None:
            out = out[out["daily_budget"] <= budget]
        if trip_days is not None:
            out = out[out["days"] <= trip_days]
        if category and category != "Any":
            out = out[out["type"] == category]
    if min_rating is not None and "rating" in out.columns:
        out = out[out["rating"] >= min_rating]
    return out.reset_index(drop=True)

def ai_explanation(module, query, profile, recs):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None, "AI explanations are currently in fallback mode. Set OPENAI_API_KEY and install the openai package to enable generative explanations."
    try:
        client = OpenAI(api_key=api_key)
        fields = [c for c in ["name", "category", "genre", "area", "type", "price", "daily_budget", "days", "rating", "Hybrid score", "Reason", "courses"] if c in recs.columns]
        records = recs[fields].to_dict(orient="records")
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            instructions=(
                "You are SmartLife AI, a careful recommendation assistant for a student capstone demo. "
                "Explain recommendations based only on the supplied catalog records and user preferences. "
                "Do not invent prices, availability, career guarantees, movie facts, or travel conditions. "
                "Clearly say scores are internal ranking scores, not probabilities. Be helpful and concise."
            ),
            input=f"Module: {module}\nUser request: {query}\nSaved profile interests: {profile.get('interests','')}\nRanked catalog records (JSON): {json.dumps(records, ensure_ascii=False, default=str)}\nExplain the top recommendations in plain language. Give one short paragraph and 3 concise bullets. Mention trade-offs or limitations where relevant."
        )
        return response.output_text, "Generated by an AI model using the ranked catalog records."
    except Exception as exc:
        return None, f"AI request failed; using local explanations instead. Check API key, model access, and internet. Details: {exc}"

def precision_recall_at_k(ranked_names, feedback_df, profile_name, module, k=5):
    judged = feedback_df[
        (feedback_df["profile"].astype(str) == profile_name) &
        (feedback_df["module"].astype(str) == module)
    ].copy()
    if judged.empty:
        return None
    judged["feedback"] = pd.to_numeric(judged["feedback"], errors="coerce").fillna(0)
    relevant = set(judged.loc[judged["feedback"] > 0, "item"].astype(str))
    judged_names = set(judged["item"].astype(str))
    top = list(ranked_names[:k])
    relevant_in_top = sum(1 for name in top if name in relevant)
    # Precision@K uses K as denominator; unjudged recommendations are not assumed relevant.
    precision = relevant_in_top / k if k else 0.0
    recall = relevant_in_top / len(relevant) if relevant else 0.0
    coverage = len(judged_names) / max(1, len(load_catalog(module)))
    return {"precision": precision, "recall": recall, "relevant": len(relevant), "judged": len(judged_names), "coverage": coverage}

# -------------------- SIDEBAR / PROFILE --------------------
profiles = load_profiles()
if not profiles:
    profiles["Student"] = {"interests": "", "created_at": datetime.now().isoformat(timespec="seconds")}
    save_profiles(profiles)

st.sidebar.markdown("## 👤 Your profile")
profile_names = sorted(profiles.keys())
chosen_profile = st.sidebar.selectbox("Active profile", profile_names)
st.session_state.profile_name = chosen_profile
profile = profiles.get(chosen_profile, DEFAULT_PROFILE.copy())
with st.sidebar.form("profile_form"):
    new_profile_name = st.text_input("Create or edit profile name", value=chosen_profile)
    profile_interests = st.text_area(
        "Saved interests and preferences",
        value=profile.get("interests", ""),
        placeholder="e.g. student, budget-conscious, programming, science fiction, nature travel",
        height=110
    )
    save_profile_clicked = st.form_submit_button("Save profile")
if save_profile_clicked:
    clean_name = new_profile_name.strip()
    if clean_name:
        profiles[clean_name] = {
            "interests": profile_interests.strip(),
            "created_at": profile.get("created_at") or datetime.now().isoformat(timespec="seconds")
        }
        save_profiles(profiles)
        st.session_state.profile_name = clean_name
        st.success("Profile saved.")
        st.rerun()
    else:
        st.sidebar.error("Profile name cannot be blank.")

st.sidebar.markdown("---")
st.sidebar.caption("Your profiles and feedback are stored locally in the project's data folder. Do not store sensitive personal information.")

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
  <h1>✨ SmartLife AI</h1>
  <p>Your personal discovery assistant for products, movies, careers, courses, and travel.</p><span class="pill">HYBRID AI · PERSONALIZED DISCOVERY · EXPLAINABLE RESULTS</span>
</div>
""", unsafe_allow_html=True)

feedback = load_feedback()
catalog_counts = {module: len(load_catalog(module)) for module in MODULES}
m1, m2, m3, m4 = st.columns(4)
m1.metric("Recommendation domains", "4")
m2.metric("Catalog items", f"{sum(catalog_counts.values()):,}")
m3.metric("Saved profiles", len(profiles))
m4.metric("Feedback records", len(feedback))

st.write("Choose a domain, describe what matters to you, and compare the ranked recommendations. Your feedback helps personalize future results.")

tabs = st.tabs([f"{MODULES[m]['emoji']} {m}" for m in MODULES] + ["📊 Evaluation", "🧠 Methodology"])
module_tabs = dict(zip(MODULES.keys(), tabs[:4]))

for module, tab in module_tabs.items():
    with tab:
        catalog = load_catalog(module)
        st.subheader(MODULES[module]["title"])
        st.caption(MODULES[module]["query_help"])
        with st.form(f"recommend_form_{module}"):
            query = st.text_area(
                "Your request",
                value=st.session_state.get(f"query_{module}", ""),
                placeholder=(
                    "e.g. recommend affordable options for a computer science student who likes learning and outdoor activities"
                ),
                key=f"query_input_{module}",
                height=90
            )
            f1, f2, f3 = st.columns(3)
            budget = None
            category = "Any"
            trip_days = None
            min_rating = None
            if module == "Products":
                budget = f1.number_input("Maximum budget (₹)", min_value=0, max_value=1000000, value=5000, step=500, key="budget_products")
                category = f2.selectbox("Category", ["Any"] + sorted(catalog["category"].dropna().astype(str).unique().tolist()), key="category_products")
                min_rating = f3.slider("Minimum rating", 1.0, 5.0, 3.5, 0.1, key="rating_products")
            elif module == "Movies":
                category = f1.selectbox("Genre", ["Any"] + sorted(catalog["genre"].dropna().astype(str).unique().tolist()), key="genre_movies")
                min_rating = f2.slider("Minimum rating", 1.0, 10.0, 6.0, 0.1, key="rating_movies")
                f3.caption("Ratings are catalog data, not streaming availability.")
            elif module == "Career & Courses":
                category = f1.selectbox("Career area", ["Any"] + sorted(catalog["area"].dropna().astype(str).unique().tolist()), key="area_careers")
                f2.caption("Career suggestions are exploratory, not guarantees of employment.")
                f3.caption("Review course requirements and local opportunities.")
            elif module == "Travel":
                budget = f1.number_input("Daily budget (₹)", min_value=500, max_value=100000, value=3500, step=500, key="budget_travel")
                trip_days = f2.slider("Maximum trip length (days)", 1, 14, 3, key="days_travel")
                category = f3.selectbox("Travel style", ["Any"] + sorted(catalog["type"].dropna().astype(str).unique().tolist()), key="type_travel")
            advanced_col1, advanced_col2 = st.columns([2, 1])
            with advanced_col1:
                focus = st.selectbox(
                    "Recommendation strategy",
                    ["Balanced", "Best match", "Top rated", "Popular picks", "Personalized", "Budget friendly"],
                    help="Changes how relevance, your profile, ratings, popularity, and (where available) price influence the ranking.",
                    key=f"focus_{module}"
                )
            with advanced_col2:
                top_n = st.slider("How many results?", 3, 25, 10, key=f"topn_{module}")
            submitted = st.form_submit_button("✨ Generate recommendations", use_container_width=True)

        if submitted:
            st.session_state[f"query_{module}"] = query
            candidates = apply_filters(catalog, module, budget, category, min_rating, trip_days)
            if candidates.empty:
                st.warning("No catalog items match those filters. Try widening your budget, rating, category, or duration.")
                st.session_state.pop(f"results_{module}", None)
            else:
                ranked, weights = hybrid_recommend(
                    candidates, module, query, profile, load_feedback(), top_n=top_n, focus=focus
                )
                st.session_state[f"results_{module}"] = {
                    "rows": ranked.to_dict(orient="records"),
                    "query": query,
                    "weights": weights,
                    "focus": focus,
                    "timestamp": datetime.now().isoformat(timespec="seconds")
                }

        saved_result = st.session_state.get(f"results_{module}")
        if saved_result:
            results = pd.DataFrame(saved_result["rows"])
            st.markdown("#### Your ranked results")
            st.caption(
                f"Strategy: {saved_result.get('focus', 'Balanced')}. Hybrid score combines request relevance, saved-profile affinity, rating, popularity, and optional cost preference. "
                "It is a ranking score, not a probability or objective measure of quality."
            )
            display_cols = [c for c in ["name", "category", "genre", "area", "type", "price", "daily_budget", "days", "rating", "year", "courses", "Content score", "Profile affinity", "Hybrid score"] if c in results.columns]
            st.dataframe(results[display_cols], use_container_width=True, hide_index=True)

            for index, row in results.reset_index(drop=True).iterrows():
                item_name = str(row.get("name", "Item"))
                with st.container(border=True):
                    left, right = st.columns([5, 1])
                    with left:
                        st.markdown(f"**#{index+1} · {item_name}**")
                        st.write(str(row.get("Reason", "")))
                        extra = []
                        for col, label in [("price", "Price"), ("daily_budget", "Est. daily budget"), ("days", "Suggested days"), ("rating", "Rating"), ("courses", "Learning path")]:
                            if col in row and pd.notna(row[col]):
                                val = row[col]
                                if col in ("price", "daily_budget"):
                                    extra.append(f"{label}: ₹{float(val):,.0f}")
                                else:
                                    extra.append(f"{label}: {val}")
                        if extra:
                            st.caption(" · ".join(extra))
                    with right:
                        st.metric("Hybrid score", f"{float(row.get('Hybrid score', 0)):.1f}/100")
                        like_col, dislike_col = st.columns(2)
                        safe_key = "".join(ch if ch.isalnum() else "_" for ch in f"{module}_{item_name}_{index}")
                        if like_col.button("👍", key=f"like_{safe_key}", help="I like this recommendation"):
                            record_feedback(st.session_state.profile_name, module, item_name, 1)
                            st.toast("Preference saved — this can influence future recommendations.")
                            st.rerun()
                        if dislike_col.button("👎", key=f"dislike_{safe_key}", help="Not for me"):
                            record_feedback(st.session_state.profile_name, module, item_name, -1)
                            st.toast("Feedback saved.")
                            st.rerun()

            with st.expander("🧠 Why these recommendations?"):
                st.write("**Scoring weights for this run**")
                st.json(saved_result["weights"])
                if st.button(f"Generate AI explanation for {module}", key=f"ai_explain_{module}"):
                    with st.spinner("Preparing explanation..."):
                        answer, status = ai_explanation(
                            module, saved_result["query"], profile, results.head(6)
                        )
                    st.session_state[f"ai_answer_{module}"] = answer
                    st.session_state[f"ai_status_{module}"] = status
                if st.session_state.get(f"ai_answer_{module}"):
                    st.markdown(st.session_state[f"ai_answer_{module}"])
                st.caption(st.session_state.get(f"ai_status_{module}", "AI explanation is optional. Configure an API key to enable it."))

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download these recommendations as CSV",
                data=csv_bytes,
                file_name=f"smartlife_{module.lower().replace(' & ', '_').replace(' ', '_')}_recommendations.csv",
                mime="text/csv",
                key=f"download_{module}"
            )
        else:
            st.info("Enter your preferences and select Generate personalized recommendations to begin.")

# -------------------- EVALUATION --------------------
with tabs[4]:
    st.subheader("📊 Recommendation Evaluation")
    feedback = load_feedback()
    current_profile = st.session_state.profile_name
    if feedback.empty:
        st.info("Rate recommendations with 👍 or 👎 first. The app will use this feedback to calculate simple evaluation metrics.")
    else:
        user_feedback = feedback[feedback["profile"].astype(str) == current_profile].copy()
        likes = int((pd.to_numeric(user_feedback["feedback"], errors="coerce") > 0).sum())
        dislikes = int((pd.to_numeric(user_feedback["feedback"], errors="coerce") < 0).sum())
        e1, e2, e3 = st.columns(3)
        e1.metric("Positive feedback", likes)
        e2.metric("Negative feedback", dislikes)
        e3.metric("Judged items", len(user_feedback))
        st.caption("Metrics are calculated against your explicit feedback. With very little feedback, results are preliminary.")
        module_choice = st.selectbox("Evaluate module", list(MODULES.keys()))
        k = st.slider("K for Precision@K and Recall@K", 3, 10, 5)
        cat = load_catalog(module_choice)
        query = st.session_state.get(f"query_{module_choice}", "")
        ranked, _ = hybrid_recommend(cat, module_choice, query, profile, feedback, top_n=len(cat))
        metrics = precision_recall_at_k(ranked["name"].astype(str).tolist(), feedback, current_profile, module_choice, k)
        if metrics is None:
            st.warning(f"No feedback recorded for {module_choice} under this profile yet.")
        else:
            a, b, c, d = st.columns(4)
            a.metric(f"Precision@{k}", f"{metrics['precision']:.3f}")
            b.metric(f"Recall@{k}", f"{metrics['recall']:.3f}")
            c.metric("Relevant items liked", metrics["relevant"])
            d.metric("Judged catalog coverage", f"{metrics['coverage']:.1%}")
            st.write(f"**Precision@{k}** = liked items among the top {k} divided by {k}.")
            st.write("**Recall@K** = liked items among the top K divided by all positively rated items in this module.")
            st.write("**Judged catalog coverage** = unique items this profile has rated divided by the catalog size.")
            st.warning("These are feedback-based offline metrics, not a controlled benchmark. Unrated items are not known to be irrelevant, and Precision@K can be conservative.")

# -------------------- METHODOLOGY --------------------
with tabs[5]:
    st.subheader("🧠 How SmartLife AI works")
    st.markdown("""
    **AI mapping:** Intelligent Recommendation Systems — hybrid content-based ranking with profile personalization.

    1. **Preference input:** The user writes a natural-language request and may save long-term interests in a profile.
    2. **Constraint filtering:** Hard constraints such as budget, category, minimum rating, or trip duration remove unsuitable candidates.
    3. **Content relevance:** TF-IDF vectorization and cosine similarity estimate how closely each item's text matches the current request.
    4. **Profile affinity:** Saved interests and previously liked items are compared with each candidate.
    5. **Hybrid scoring:** Content relevance, profile affinity, rating, and popularity are combined with configurable weights.
    6. **Feedback loop:** Likes and dislikes are stored locally and inform later profile-affinity calculations.
    7. **Evaluation:** Precision@K, Recall@K, and judged catalog coverage summarize behavior based on explicit feedback.
    8. **Generative explanations (optional):** An AI model explains ranked catalog results in natural language; it does not replace the deterministic ranking algorithm.

    **Limitations:** The included catalogs are curated educational demo data, not live inventories or authoritative external datasets. Product prices, travel budgets, ratings, and popularity values are illustrative. Career suggestions are not employment guarantees. Evaluation metrics are meaningful only to the extent that feedback is representative.
    """)
    st.markdown("**Hybrid scoring formula**")
    st.latex(r"S(i,u)=w_q C(i,q)+w_p P(i,u)+w_r R(i)+w_v V(i)")
    st.write("C = request-content similarity; P = profile/liked-item affinity; R = normalized catalog rating; V = normalized popularity. The weights are adjusted when a profile has no saved preferences.")
    st.markdown("**Project deliverables to document:** problem statement, objectives, data source and licensing, preprocessing, algorithm, architecture, screenshots, test cases, evaluation results, limitations, and future work.")

st.divider()
st.caption("SmartLife AI · Educational capstone prototype · Profiles and feedback are stored locally on this computer.")
