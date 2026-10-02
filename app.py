
import io
import hashlib
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

APP_VERSION = "1.2.0"

def normalize_colname(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")

def load_table(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported format. Please use CSV or Excel.")

def find_column(df, candidates):
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    for norm, original in norm_map.items():
        for cand in candidates:
            if cand in norm:
                return original
    return None

def safe_text_series(df, col):
    if col is None or col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[col].fillna("").astype(str)

def compose_text(df, title_col, abstract_col, keywords_col):
    title = safe_text_series(df, title_col)
    abstract = safe_text_series(df, abstract_col)
    keywords = safe_text_series(df, keywords_col)
    return (
        "TITLE: " + title
        + " ABSTRACT: " + abstract
        + " KEYWORDS: " + keywords
    ).str.replace(r"\s+", " ", regex=True).str.strip()

def stable_id(row, title_col, doi_col, idx):
    doi = ""
    title = ""
    if doi_col and doi_col in row.index:
        doi = str(row[doi_col]) if pd.notna(row[doi_col]) else ""
    if title_col and title_col in row.index:
        title = str(row[title_col]) if pd.notna(row[title_col]) else ""
    raw = (doi.strip().lower() or title.strip().lower() or str(idx)).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]

def infer_mapping(df):
    return {
        "title": find_column(df, ["title", "titre", "ti"]),
        "abstract": find_column(df, ["abstract", "resume", "résumé", "ab"]),
        "keywords": find_column(df, ["keywords", "keyword", "mots_cles", "de"]),
        "doi": find_column(df, ["doi", "di"]),
        "authors": find_column(df, ["authors", "auteurs", "au"]),
        "year": find_column(df, ["year", "annee", "année", "py"]),
    }


def normalize_doi_value(x):
    s = "" if pd.isna(x) else str(x).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s

def normalize_title_value(x):
    s = "" if pd.isna(x) else str(x).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def enrich_examples_from_main(example_df, main_df, main_map):
    """
    Match example records back to the main corpus using DOI first, then normalized title.
    This ensures the model is trained on the real title + abstract + keywords from the
    1,426-record corpus instead of short notes in the example files.
    """
    if example_df is None or example_df.empty:
        return pd.DataFrame(), 0

    emap = infer_mapping(example_df)
    main = main_df.copy()

    main["_doi_key"] = safe_text_series(main, main_map["doi"]).map(normalize_doi_value)
    main["_title_key"] = safe_text_series(main, main_map["title"]).map(normalize_title_value)

    doi_lookup = {}
    title_lookup = {}
    for idx, row in main.iterrows():
        d = row["_doi_key"]
        t = row["_title_key"]
        if d and d not in doi_lookup:
            doi_lookup[d] = idx
        if t and t not in title_lookup:
            title_lookup[t] = idx

    matched_rows = []
    for _, erow in example_df.iterrows():
        edoi = normalize_doi_value(erow[emap["doi"]]) if emap["doi"] else ""
        etitle = normalize_title_value(erow[emap["title"]]) if emap["title"] else ""
        midx = None

        if edoi and edoi in doi_lookup:
            midx = doi_lookup[edoi]
        elif etitle and etitle in title_lookup:
            midx = title_lookup[etitle]

        if midx is not None:
            matched_rows.append(main.loc[midx])

    if not matched_rows:
        return pd.DataFrame(), 0

    matched = pd.DataFrame(matched_rows).reset_index(drop=True)
    return matched, len(matched)

def make_training_from_examples(positive_df, negative_df):
    chunks = []
    main_df = st.session_state.main_df
    main_map = st.session_state.mapping

    def prep(df, label):
        if df is None or df.empty:
            return None, 0

        matched, matched_n = enrich_examples_from_main(
            df, main_df, main_map
        )

        # If some examples cannot be matched, fall back to their own text,
        # but matched corpus text is preferred.
        if matched_n > 0:
            text = compose_text(
                matched,
                main_map["title"],
                main_map["abstract"],
                main_map["keywords"]
            )
            out = pd.DataFrame({"text": text, "label": label})
        else:
            m = infer_mapping(df)
            text = compose_text(df, m["title"], m["abstract"], m["keywords"])
            out = pd.DataFrame({"text": text, "label": label})

        return out[out["text"].str.len() > 10], matched_n

    p, p_matched = prep(positive_df, 1)
    n, n_matched = prep(negative_df, 0)

    if p is not None:
        chunks.append(p)
    if n is not None:
        chunks.append(n)

    if "screening_df" in st.session_state:
        s = st.session_state.screening_df
        decided = s[s["human_decision"].isin(["INCLUDE", "EXCLUDE"])].copy()
        if not decided.empty:
            decided["label"] = (decided["human_decision"] == "INCLUDE").astype(int)
            chunks.append(
                decided[["text_for_model", "label"]]
                .rename(columns={"text_for_model": "text"})
            )

    st.session_state.example_match_stats = {
        "positive_matched": int(p_matched),
        "negative_matched": int(n_matched)
    }

    if not chunks:
        return pd.DataFrame(columns=["text", "label"])

    train = pd.concat(chunks, ignore_index=True)
    return train.drop_duplicates(subset=["text", "label"])

def keyword_evidence(text):
    text_l = str(text).lower()
    positive_terms = [
        "sectoral stock", "sector stock", "sector index", "sectoral index",
        "industry index", "industry indices", "industry portfolio", "sectoral portfolio",
        "equity sector", "sectoral equity", "stock market sector", "stock sectors",
        "volatility", "spillover", "connectedness", "contagion", "systemic risk",
        "tail risk", "value at risk", "expected shortfall", "cvar",
        "market risk", "resilience", "stability", "beta"
    ]
    negative_terms = [
        "public sector", "health sector", "agricultural sector", "education sector",
        "government sector", "firm-level", "firm level", "operational risk",
        "credit scoring", "fraud detection"
    ]
    pos = [t for t in positive_terms if t in text_l][:6]
    neg = [t for t in negative_terms if t in text_l][:6]
    return ", ".join(pos) if pos else "—", ", ".join(neg) if neg else "—"

def excel_bytes(df, audit_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="SCREENING_RESULTS")
        audit_df.to_excel(writer, index=False, sheet_name="AUDIT_LOG")
    output.seek(0)
    return output.getvalue()

st.set_page_config(
    page_title="BSLR Screening Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 BSLR Screening Assistant")
st.caption(
    "Academic screening assistant: transparent rules + TF-IDF + Logistic Regression + human validation"
)

with st.sidebar:
    st.header("Project")
    project_name = st.text_input("Project name", "Sectoral Market Risk Review")
    st.write(f"App version: **{APP_VERSION}**")
    st.info(
        "The model helps prioritize articles. "
        "The final INCLUDE / EXCLUDE decision remains with the researcher."
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Data & Criteria",
    "2️⃣ Train Model",
    "3️⃣ Results & Review",
    "4️⃣ Export & Audit"
])

with tab1:
    st.subheader("A. Upload the main dataset")
    main_file = st.file_uploader(
        "Main CSV/Excel dataset (e.g. 1,426 records)",
        type=["csv", "xlsx", "xls"],
        key="main"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        pos_file = st.file_uploader(
            "Relevant examples",
            type=["csv", "xlsx", "xls"],
            key="positive"
        )
    with col_b:
        neg_file = st.file_uploader(
            "Not relevant examples",
            type=["csv", "xlsx", "xls"],
            key="negative"
        )

    st.subheader("B. Eligibility criteria")
    research_question = st.text_area(
        "Research question / objective",
        "Market risk and stability of sectoral stock indices.",
        height=80
    )

    default_criteria = """INCLUDE:
- sectoral stock/equity indices
- industry indices or industry portfolios
- sector-level equity markets
- volatility, market risk, systemic risk, tail risk
- VaR / Expected Shortfall
- spillovers, connectedness, contagion
- stability, resilience, beta

EXCLUDE:
- firm-level studies without a sectoral index/portfolio
- public/health/agriculture/education sector studies unrelated to equity markets
- operational risk, fraud, or pure credit-risk studies without equity-market analysis
- non-financial sector studies outside stock/equity markets
"""
    eligibility_criteria = st.text_area(
        "Inclusion / exclusion criteria",
        default_criteria,
        height=260
    )

    if main_file is not None:
        try:
            main_df = load_table(main_file)
            mapping = infer_mapping(main_df)

            st.success(f"{len(main_df):,} records loaded.")
            st.write("Automatically detected columns:")
            st.json(mapping)

            options = ["—"] + list(main_df.columns)
            c1, c2, c3, c4 = st.columns(4)

            def idx_for(col):
                return options.index(col) if col in options else 0

            with c1:
                title_col = st.selectbox("Title", options, index=idx_for(mapping["title"]))
            with c2:
                abstract_col = st.selectbox("Abstract", options, index=idx_for(mapping["abstract"]))
            with c3:
                keywords_col = st.selectbox("Keywords", options, index=idx_for(mapping["keywords"]))
            with c4:
                doi_col = st.selectbox("DOI", options, index=idx_for(mapping["doi"]))

            title_col = None if title_col == "—" else title_col
            abstract_col = None if abstract_col == "—" else abstract_col
            keywords_col = None if keywords_col == "—" else keywords_col
            doi_col = None if doi_col == "—" else doi_col

            work = main_df.copy()
            work["record_id"] = [
                stable_id(row, title_col, doi_col, idx)
                for idx, row in work.iterrows()
            ]
            work["text_for_model"] = compose_text(work, title_col, abstract_col, keywords_col)
            work["model_probability"] = np.nan
            work["ai_category"] = ""
            work["human_decision"] = ""
            work["exclusion_reason"] = ""
            work["review_note"] = ""

            st.session_state.main_df = main_df
            st.session_state.screening_df = work
            st.session_state.mapping = {
                "title": title_col,
                "abstract": abstract_col,
                "keywords": keywords_col,
                "doi": doi_col,
                "authors": mapping["authors"],
                "year": mapping["year"],
            }
            st.session_state.research_question = research_question
            st.session_state.eligibility_criteria = eligibility_criteria

            st.dataframe(main_df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Could not read the file: {e}")

with tab2:
    st.subheader("Train the screening model")

    if "screening_df" not in st.session_state:
        st.warning("Please upload the main dataset in Tab 1 first.")
    else:
        positive_df = load_table(pos_file) if pos_file is not None else pd.DataFrame()
        negative_df = load_table(neg_file) if neg_file is not None else pd.DataFrame()

        train_df = make_training_from_examples(positive_df, negative_df)

        p_count = int((train_df["label"] == 1).sum()) if not train_df.empty else 0
        n_count = int((train_df["label"] == 0).sum()) if not train_df.empty else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Relevant examples", p_count)
        m2.metric("Not relevant examples", n_count)
        m3.metric("Total training records", len(train_df))

        match_stats = st.session_state.get("example_match_stats", {})
        if match_stats:
            st.info(
                f"Matched back to the main corpus: "
                f"{match_stats.get('positive_matched', 0)} relevant examples and "
                f"{match_stats.get('negative_matched', 0)} not relevant examples. "
                f"The model is trained on the real title + abstract + keywords from the main dataset."
            )

        st.caption(
            "Provide clear examples from both classes. "
            "Training cannot start if one class is missing."
        )

        max_features = st.slider("Maximum TF-IDF features", 3000, 30000, 12000, step=1000)
        ngram_max = st.selectbox("N-grams", [1, 2], index=1)

        if st.button("🚀 Train model and score dataset", type="primary"):
            if p_count < 5 or n_count < 5:
                st.error("Please provide at least 5 relevant and 5 not relevant examples.")
            else:
                X = train_df["text"].astype(str)
                y = train_df["label"].astype(int)

                pipe = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, ngram_max),
                        max_features=max_features,
                        min_df=1
                    )),
                    ("clf", LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=4.0,
                        random_state=42
                    ))
                ])

                metrics = {}
                if len(train_df) >= 30 and y.value_counts().min() >= 5:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=0.25,
                        random_state=42,
                        stratify=y
                    )
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_test)

                    pr, rc, f1, _ = precision_recall_fscore_support(
                        y_test, pred, average="binary", zero_division=0
                    )
                    metrics = {
                        "precision": float(pr),
                        "recall": float(rc),
                        "f1": float(f1),
                        "test_n": int(len(y_test))
                    }

                pipe.fit(X, y)

                scored = st.session_state.screening_df.copy()
                scored["model_probability"] = pipe.predict_proba(
                    scored["text_for_model"].astype(str)
                )[:, 1]

                st.session_state.model = pipe
                st.session_state.metrics = metrics
                st.session_state.screening_df = scored
                st.session_state.score_diagnostics = {
                    "min": float(scored["model_probability"].min()),
                    "q10": float(scored["model_probability"].quantile(0.10)),
                    "median": float(scored["model_probability"].median()),
                    "q90": float(scored["model_probability"].quantile(0.90)),
                    "max": float(scored["model_probability"].max()),
                }
                st.success("Model trained successfully and dataset scored.")

        if "metrics" in st.session_state and st.session_state.metrics:
            st.write("Indicative internal validation:")
            st.json(st.session_state.metrics)

with tab3:
    st.subheader("Results & human validation")

    if (
        "screening_df" not in st.session_state
        or st.session_state.screening_df["model_probability"].isna().all()
    ):
        st.warning("Train the model in Tab 2 first.")
    else:
        low_threshold = st.slider("LOW relevance threshold", 0.0, 0.5, 0.35, 0.01)
        high_threshold = st.slider("HIGH relevance threshold", 0.5, 1.0, 0.65, 0.01)

        diagnostics = st.session_state.get("score_diagnostics")
        if diagnostics:
            st.caption(
                "Score distribution — "
                f"min: {diagnostics['min']:.2f} | "
                f"10th pct: {diagnostics['q10']:.2f} | "
                f"median: {diagnostics['median']:.2f} | "
                f"90th pct: {diagnostics['q90']:.2f} | "
                f"max: {diagnostics['max']:.2f}"
            )
            st.warning(
                "Thresholds are for prioritization, not automatic academic exclusion. "
                "Do not adjust them merely to force a predetermined number of included studies."
            )

        if low_threshold >= high_threshold:
            st.error("The LOW threshold must be lower than the HIGH threshold.")
        else:
            df = st.session_state.screening_df.copy()

            def cat(p):
                if p >= high_threshold:
                    return "HIGH RELEVANCE"
                if p <= low_threshold:
                    return "LOW RELEVANCE"
                return "MANUAL REVIEW"

            df["ai_category"] = df["model_probability"].apply(cat)

            pos_ev, neg_ev = [], []
            for txt in df["text_for_model"]:
                p, n = keyword_evidence(txt)
                pos_ev.append(p)
                neg_ev.append(n)

            df["positive_evidence"] = pos_ev
            df["negative_evidence"] = neg_ev
            st.session_state.screening_df = df

            c1, c2, c3 = st.columns(3)
            c1.metric("HIGH", int((df["ai_category"] == "HIGH RELEVANCE").sum()))
            c2.metric("MANUAL REVIEW", int((df["ai_category"] == "MANUAL REVIEW").sum()))
            c3.metric("LOW", int((df["ai_category"] == "LOW RELEVANCE").sum()))

            st.divider()
            st.markdown("### Priority review")

            review_mode = st.selectbox(
                "Review category",
                ["MANUAL REVIEW", "HIGH RELEVANCE", "LOW RELEVANCE", "All undecided"]
            )

            pending = df[df["human_decision"] == ""].copy()
            if review_mode != "All undecided":
                pending = pending[pending["ai_category"] == review_mode]

            if review_mode == "MANUAL REVIEW":
                pending["uncertainty"] = (pending["model_probability"] - 0.5).abs()
                pending = pending.sort_values("uncertainty", ascending=True)
            else:
                pending = pending.sort_values("model_probability", ascending=False)

            if pending.empty:
                st.success("No remaining records in this category.")
            else:
                row = pending.iloc[0]
                idx = row.name
                mp = st.session_state.mapping

                title = str(row[mp["title"]]) if mp["title"] else "(Title unavailable)"
                abstract = str(row[mp["abstract"]]) if mp["abstract"] else ""
                keywords = str(row[mp["keywords"]]) if mp["keywords"] else ""
                doi = str(row[mp["doi"]]) if mp["doi"] else ""

                st.markdown(f"#### {title}")
                st.progress(float(row["model_probability"]))
                st.write(f"**Relevance score:** {row['model_probability']:.1%}")
                st.write(f"**Category:** {row['ai_category']}")

                if doi:
                    st.write(f"**DOI:** {doi}")

                st.write("**Abstract**")
                st.write(abstract if abstract and abstract != "nan" else "—")

                st.write("**Keywords**")
                st.write(keywords if keywords and keywords != "nan" else "—")

                st.write(f"**Positive evidence:** {row['positive_evidence']}")
                st.write(f"**Negative evidence:** {row['negative_evidence']}")

                reason = st.selectbox(
                    "Exclusion reason (only if EXCLUDE)",
                    [
                        "",
                        "Non-equity / non-stock-market",
                        "Firm-level only",
                        "Wrong sector meaning",
                        "Wrong outcome",
                        "Wrong document / scope",
                        "Other"
                    ]
                )
                note = st.text_input("Optional review note")

                b1, b2, b3 = st.columns(3)

                if b1.button("✅ INCLUDE", use_container_width=True):
                    st.session_state.screening_df.loc[idx, "human_decision"] = "INCLUDE"
                    st.session_state.screening_df.loc[idx, "review_note"] = note
                    st.rerun()

                if b2.button("❌ EXCLUDE", use_container_width=True):
                    st.session_state.screening_df.loc[idx, "human_decision"] = "EXCLUDE"
                    st.session_state.screening_df.loc[idx, "exclusion_reason"] = reason
                    st.session_state.screening_df.loc[idx, "review_note"] = note
                    st.rerun()

                if b3.button("❓ UNCERTAIN", use_container_width=True):
                    st.session_state.screening_df.loc[idx, "human_decision"] = "UNCERTAIN"
                    st.session_state.screening_df.loc[idx, "review_note"] = note
                    st.rerun()

            st.divider()
            st.markdown("### Scored dataset preview")

            mp = st.session_state.mapping
            display_cols = [
                "record_id",
                mp["title"] if mp["title"] else None,
                mp["doi"] if mp["doi"] else None,
                "model_probability",
                "ai_category",
                "human_decision",
                "exclusion_reason"
            ]
            display_cols = [c for c in display_cols if c and c in df.columns]

            st.dataframe(
                df[display_cols].sort_values("model_probability", ascending=False),
                use_container_width=True,
                height=420
            )

with tab4:
    st.subheader("Export & audit trail")

    if "screening_df" not in st.session_state:
        st.warning("No data available for export.")
    else:
        df = st.session_state.screening_df.copy()

        audit = pd.DataFrame([{
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "app_version": APP_VERSION,
            "project_name": project_name,
            "research_question": st.session_state.get("research_question", ""),
            "eligibility_criteria": st.session_state.get("eligibility_criteria", ""),
            "n_records": len(df),
            "n_include_human": int((df["human_decision"] == "INCLUDE").sum()),
            "n_exclude_human": int((df["human_decision"] == "EXCLUDE").sum()),
            "n_uncertain_human": int((df["human_decision"] == "UNCERTAIN").sum()),
            "random_state": 42,
            "model": "TF-IDF + LogisticRegression(class_weight='balanced', C=4.0)"
        }])

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download screening results (CSV)",
            data=csv_bytes,
            file_name="screening_results.csv",
            mime="text/csv"
        )

        xlsx_bytes = excel_bytes(df, audit)
        st.download_button(
            "⬇️ Download Excel + audit log",
            data=xlsx_bytes,
            file_name="screening_results_with_audit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("### Screening summary")
        st.write({
            "records": len(df),
            "human_include": int((df["human_decision"] == "INCLUDE").sum()),
            "human_exclude": int((df["human_decision"] == "EXCLUDE").sum()),
            "human_uncertain": int((df["human_decision"] == "UNCERTAIN").sum()),
            "undecided": int((df["human_decision"] == "").sum()),
        })
