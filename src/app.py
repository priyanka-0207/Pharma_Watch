"""
Drug Safety Intelligence Dashboard — Enhanced Edition
─────────────────────────────────────────────────────
Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drug Safety Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme-Safe CSS (works on both light & dark mode) ────────────────────────

st.markdown("""
<style>
    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: rgba(78, 205, 196, 0.08);
        border: 1px solid rgba(78, 205, 196, 0.25);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(78, 205, 196, 0.18);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0fa89e !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0fa89e, #0d8f87) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #12c4b8, #0fa89e) !important;
        box-shadow: 0 4px 15px rgba(15, 168, 158, 0.35) !important;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15,168,158,0.12), rgba(15,168,158,0.03));
        border: 1px solid rgba(15,168,158,0.25);
        border-radius: 16px;
        padding: 40px 35px;
        margin-bottom: 30px;
        text-align: center;
    }
    .hero-banner h1 {
        font-size: 2.2rem !important;
        color: #0d8f87 !important;
        margin-bottom: 8px !important;
    }
    .hero-banner p {
        font-size: 1.05rem;
        opacity: 0.75;
        margin: 0;
    }

    /* ── Info cards ── */
    .info-card {
        background: rgba(78, 205, 196, 0.06);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: transform 0.2s ease;
    }
    .info-card:hover {
        transform: translateX(4px);
    }
    .info-card h4 {
        color: #0fa89e !important;
        margin-bottom: 4px !important;
        font-size: 1.05rem !important;
    }
    .info-card p {
        margin: 0;
        font-size: 0.92rem;
    }

    /* ── Severity badges ── */
    .severity-high   { color: #e53e3e; font-weight: 700; }
    .severity-medium { color: #d69e2e; font-weight: 700; }
    .severity-low    { color: #38a169; font-weight: 700; }

    /* ── Section divider ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        padding-bottom: 6px;
        margin-bottom: 12px;
        border-bottom: 2px solid rgba(15,168,158,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ─── Plotly Theme Helper ─────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    font=dict(size=13),
)

TEAL_SCALE = ["#b2dfdb", "#4db6ac", "#00897b", "#00695c"]
RED_SCALE = ["#ffcdd2", "#ef9a9a", "#ef5350", "#c62828"]
ACCENT = "#0fa89e"


# ─── Data Loading ────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv(
        "C:/Users/priya/drug_safety_intelligence/result/processed_drug_safety.csv"
    )
    with open("C:/Users/priya/drug_safety_intelligence/data/model_results.json") as f:
        model_results = json.load(f)
    with open("C:/Users/priya/drug_safety_intelligence/data/nlp_analysis.json") as f:
        nlp_results = json.load(f)
    label_sim = pd.read_csv(
        "C:/Users/priya/drug_safety_intelligence/result/label_text_similarity.csv",
        index_col=0,
    )
    return df, model_results, nlp_results, label_sim


df, model_results, nlp_results, label_sim = load_data()


# ─── Helper ──────────────────────────────────────────────────────────────────

def to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💊 Drug Safety Intelligence")
    st.caption("Powered by ToolUniverse + FDA FAERS")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Drug Explorer", "🤖 ML Predictions", "📝 NLP Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")
    all_categories = sorted(df["drug_category"].unique())
    selected_categories = st.multiselect(
        "Drug Categories",
        all_categories,
        default=all_categories,
        help="Filter all pages by therapeutic category",
    )

    filtered_df = df[df["drug_category"].isin(selected_categories)]

    st.markdown("---")
    st.caption(f"📊 Showing **{len(filtered_df):,}** / {len(df):,} records")


# ═════════════════════════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":

    st.markdown("""
    <div class="hero-banner">
        <h1>Drug Safety Intelligence Platform</h1>
        <p>Comprehensive adverse event analysis across therapeutic categories using FDA FAERS data</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drugs Analyzed", filtered_df["drug"].nunique())
    c2.metric("Unique Reactions", filtered_df["reaction"].nunique())
    c3.metric("Total Records", f"{len(filtered_df):,}")
    c4.metric("Drug Categories", filtered_df["drug_category"].nunique())

    st.markdown("")
    col_left, col_right = st.columns(2)

    with col_left:
        reactions_by_cat = (
            filtered_df.groupby("drug_category")["reaction"]
            .nunique()
            .reset_index()
            .sort_values("reaction", ascending=True)
        )
        fig = px.bar(
            reactions_by_cat,
            y="drug_category",
            x="reaction",
            orientation="h",
            title="Unique Adverse Reactions by Category",
            color="reaction",
            color_continuous_scale=TEAL_SCALE,
        )
        fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="", xaxis_title="Unique Reactions")
        st.plotly_chart(fig, width="stretch")

    with col_right:
        records_by_cat = (
            filtered_df.groupby("drug_category")
            .size()
            .reset_index(name="records")
            .sort_values("records", ascending=False)
        )
        fig = px.pie(
            records_by_cat,
            values="records",
            names="drug_category",
            title="Record Distribution by Category",
            color_discrete_sequence=px.colors.sequential.Tealgrn_r,
            hole=0.45,
        )
        fig.update_layout(**CHART_LAYOUT)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-header">🔥 Top 20 Most Reported Adverse Reactions</div>', unsafe_allow_html=True)
    top_reactions = (
        filtered_df.groupby("reaction")["reaction_count"]
        .sum()
        .nlargest(20)
        .reset_index()
    )
    fig = px.bar(
        top_reactions,
        x="reaction_count",
        y="reaction",
        orientation="h",
        color="reaction_count",
        color_continuous_scale=RED_SCALE,
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed", title=""), xaxis_title="Total Reports", height=500)
    st.plotly_chart(fig, width="stretch")

    st.download_button("📥 Export Overview Data", to_csv_bytes(filtered_df), "drug_safety_overview.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# 🔍 DRUG EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Drug Explorer":
    st.markdown("# 🔍 Drug Explorer")

    drugs_in_filter = sorted(filtered_df["drug"].unique())
    search_term = st.text_input("Search for a drug", placeholder="Type to search...")
    if search_term:
        drugs_in_filter = [d for d in drugs_in_filter if search_term.lower() in d.lower()]

    if not drugs_in_filter:
        st.warning("No drugs match your search / filter. Adjust the sidebar filters.")
        st.stop()

    selected_drug = st.selectbox("Select a drug:", drugs_in_filter)
    drug_df = filtered_df[filtered_df["drug"] == selected_drug].sort_values("reaction_count", ascending=False)

    st.markdown("---")

    severity_data = drug_df.iloc[0] if len(drug_df) > 0 else {}
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Reactions", drug_df["reaction"].nunique())
    s2.metric("Serious Report Ratio", f"{severity_data.get('serious_ratio', 0):.1%}")
    s3.metric("Death Rate", f"{severity_data.get('death_rate', 0):.4%}")
    s4.metric("Hospitalization Rate", f"{severity_data.get('hospitalization_rate', 0):.2%}")

    st.markdown("")
    tab1, tab2, tab3 = st.tabs(["📊 Top Reactions", "📋 Full Data Table", "🔑 Key Warning Terms"])

    with tab1:
        n_reactions = st.slider("Number of reactions to show", 5, 50, 15, key="n_react")
        fig = px.bar(
            drug_df.head(n_reactions),
            x="reaction_count", y="reaction", orientation="h",
            title=f"Top {n_reactions} Reactions — {selected_drug.capitalize()}",
            color="reaction_count", color_continuous_scale=TEAL_SCALE,
        )
        fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed", title=""), xaxis_title="Report Count",
                          height=max(350, n_reactions * 28))
        st.plotly_chart(fig, width="stretch")

    with tab2:
        st.dataframe(drug_df, width="stretch", height=450)
        st.download_button(f"📥 Export {selected_drug} data", to_csv_bytes(drug_df),
                           f"{selected_drug}_reactions.csv", "text/csv")

    with tab3:
        keywords = nlp_results.get("keywords_per_drug", {}).get(selected_drug, [])
        if keywords:
            kw_df = pd.DataFrame(keywords)
            if "score" in kw_df.columns and "term" in kw_df.columns:
                fig = px.bar(
                    kw_df.sort_values("score", ascending=True).tail(20),
                    x="score", y="term", orientation="h",
                    title=f"TF-IDF Key Terms — {selected_drug.capitalize()}",
                    color="score", color_continuous_scale=["#fff9c4", "#f57f17"],
                )
                fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(fig, width="stretch")
            else:
                st.dataframe(kw_df, width="stretch")
        else:
            st.info(f"No keyword data available for {selected_drug}.")


# ═════════════════════════════════════════════════════════════════════════════
# 🤖 ML PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🤖 ML Predictions":
    st.markdown("# 🤖 ML Model Performance")

    results_df = pd.DataFrame(model_results).T
    results_df.index.name = "Model"

    if "roc_auc" in results_df.columns:
        best_model = results_df["roc_auc"].idxmax()
        best_auc = results_df["roc_auc"].max()
        st.success(f"🏆  Best model: **{best_model}** with ROC-AUC = **{best_auc:.4f}**")

    st.markdown("---")

    available_metrics = results_df.columns.tolist()
    selected_metrics = st.multiselect(
        "Select metrics to compare", available_metrics,
        default=[m for m in ["roc_auc", "pr_auc", "f1_high_sev"] if m in available_metrics],
    )

    if not selected_metrics:
        st.warning("Select at least one metric.")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Chart Comparison", "📋 Results Table"])

    with tab1:
        fig = go.Figure()
        colors = [ACCENT, "#e53e3e", "#d69e2e", "#805ad5", "#3182ce"]
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(go.Bar(
                name=metric, x=results_df.index, y=results_df[metric],
                marker_color=colors[i % len(colors)],
                text=results_df[metric].round(3), textposition="outside",
            ))
        fig.update_layout(**CHART_LAYOUT, barmode="group", title="Model Comparison",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          yaxis_title="Score", height=480)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        styled = results_df[selected_metrics].style.format("{:.4f}").background_gradient(cmap="YlGnBu", axis=0)
        st.dataframe(styled, width="stretch")
        st.download_button("📥 Export Model Results", to_csv_bytes(results_df.reset_index()),
                           "model_results.csv", "text/csv")

    # Radar chart
    st.markdown('<div class="section-header">🕸️ Radar Comparison</div>', unsafe_allow_html=True)
    radar_models = st.multiselect(
        "Select models for radar chart", results_df.index.tolist(),
        default=results_df.index.tolist()[:3],
    )
    if radar_models and len(selected_metrics) >= 3:
        fig = go.Figure()
        radar_colors = [ACCENT, "#e53e3e", "#d69e2e", "#805ad5", "#3182ce"]
        for i, model_name in enumerate(radar_models):
            vals = results_df.loc[model_name, selected_metrics].values.tolist()
            vals.append(vals[0])
            metrics_loop = selected_metrics + [selected_metrics[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=metrics_loop, name=model_name, fill="toself",
                line=dict(color=radar_colors[i % len(radar_colors)]),
            ))
        fig.update_layout(
            **CHART_LAYOUT,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=500,
        )
        st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# 📝 NLP INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📝 NLP Insights":
    st.markdown("# 📝 NLP Analysis of Drug Labels")

    tab1, tab2 = st.tabs(["🔗 Label Similarity", "⚠️ Risk Patterns"])

    with tab1:
        st.markdown('<div class="section-header">Drug Label Text Similarity</div>', unsafe_allow_html=True)
        st.caption("Cosine similarity between drug warning labels — brighter = more similar")

        fig = px.imshow(label_sim, title="", color_continuous_scale="Tealgrn", aspect="auto")
        fig.update_layout(**CHART_LAYOUT, height=600)
        st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="section-header">🔗 Most Similar Drug Pairs</div>', unsafe_allow_html=True)
        sim_pairs = []
        cols = label_sim.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                sim_pairs.append({"Drug A": cols[i], "Drug B": cols[j], "Similarity": label_sim.iloc[i, j]})
        pairs_df = pd.DataFrame(sim_pairs).sort_values("Similarity", ascending=False)
        st.dataframe(
            pairs_df.head(15).style.format({"Similarity": "{:.4f}"}).background_gradient(
                subset=["Similarity"], cmap="YlGnBu"
            ),
            width="stretch",
        )

    with tab2:
        st.markdown('<div class="section-header">⚠️ Risk Pattern Detection</div>', unsafe_allow_html=True)
        st.caption("Number of high-risk phrases found in drug labels")

        risk_scores = nlp_results.get("risk_scores", {})
        risk_df = (
            pd.DataFrame(list(risk_scores.items()), columns=["Drug", "Risk Patterns Found"])
            .sort_values("Risk Patterns Found", ascending=False)
        )

        fig = px.bar(
            risk_df, x="Drug", y="Risk Patterns Found",
            color="Risk Patterns Found", color_continuous_scale=RED_SCALE, title="",
        )
        fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig, width="stretch")

        if not risk_df.empty:
            st.markdown('<div class="section-header">🚨 Highest Risk Drugs</div>', unsafe_allow_html=True)
            q75 = risk_df["Risk Patterns Found"].quantile(0.75)
            q50 = risk_df["Risk Patterns Found"].quantile(0.5)
            for _, row in risk_df.head(5).iterrows():
                val = row["Risk Patterns Found"]
                level = "severity-high" if val > q75 else ("severity-medium" if val > q50 else "severity-low")
                st.markdown(
                    f'<div class="info-card">'
                    f'<h4>{row["Drug"].capitalize()}</h4>'
                    f'<p>Risk patterns found: <span class="{level}">{val}</span></p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.download_button("📥 Export Risk Data", to_csv_bytes(risk_df), "risk_patterns.csv", "text/csv")