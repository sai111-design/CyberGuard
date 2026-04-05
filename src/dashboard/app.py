"""CyberGuard — GRC Dashboard (Streamlit) — Enhanced with Model Insights."""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberGuard — GRC Dashboard",
    layout="wide",
    page_icon="🛡️",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_risks():
    return pd.read_csv(os.path.join(DATA_DIR, "risk_register.csv"))


@st.cache_data
def load_crosswalk():
    return pd.read_csv(os.path.join(DATA_DIR, "crosswalk.csv"))


@st.cache_data
def load_controls():
    return pd.read_csv(os.path.join(DATA_DIR, "org_controls.csv"))


@st.cache_data
def load_alerts():
    return pd.read_csv(os.path.join(DATA_DIR, "alerts.csv"))


@st.cache_data
def load_controls_clean():
    return pd.read_csv(os.path.join(DATA_DIR, "controls_clean.csv"))


risks_df = load_risks()
cross_df = load_crosswalk()
controls_df = load_controls()
alerts_df = load_alerts()
controls_clean_df = load_controls_clean()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

csf_filter = st.sidebar.multiselect(
    "CSF Function",
    options=sorted(risks_df["nist_csf_function"].unique()),
    default=sorted(risks_df["nist_csf_function"].unique()),
)
level_filter = st.sidebar.multiselect(
    "Risk Level",
    options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
)
status_filter = st.sidebar.multiselect(
    "Status",
    options=["implemented", "partial", "planned", "missing"],
    default=["implemented", "partial", "planned", "missing"],
)

# Apply filters
risks_f = risks_df[
    risks_df["nist_csf_function"].isin(csf_filter)
    & risks_df["risk_level"].isin(level_filter)
    & risks_df["status"].isin(status_filter)
]
controls_f = controls_df[
    controls_df["nist_csf_function"].isin(csf_filter)
    & controls_df["status"].isin(status_filter)
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛡️ CyberGuard — GRC Dashboard")
st.caption(
    "Local NIST 800-53 compliance intelligence · TF-IDF + IsolationForest · "
    "Trained on NIST cybersecurity dataset"
)

# ── KPI Row (5 columns) ───────────────────────────────────────────────────────
total = len(controls_df)
compliant = int((controls_df["status"] == "implemented").sum())
pct = round(compliant / total * 100, 1)
critical = int((risks_df["risk_level"] == "CRITICAL").sum())
high = int((risks_df["risk_level"] == "HIGH").sum())
anomalies = int(risks_df["anomaly_flag"].sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Compliance Score", f"{pct}%", delta=f"{compliant}/{total} controls")
col2.metric("Critical Risks", critical, delta_color="inverse")
col3.metric("High Risks", high, delta_color="inverse")
col4.metric("Anomalies Detected", anomalies, delta_color="inverse")
col5.metric("Frameworks Mapped", "NIST · ISO · SOC2")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Risk Heatmap",
    "🔗 Control Mapping",
    "📊 Gap Analysis",
    "⚠️ Anomaly Feed",
    "🧠 Model Insights",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Risk Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    level_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    pivot = (
        risks_f.groupby(["family_code", "risk_level"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=level_order, fill_value=0)
    )

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
        title="Risk Heatmap — Control Family vs Severity",
        labels=dict(x="Risk Level", y="Control Family", color="Count"),
        aspect="auto",
    )
    fig_heat.update_xaxes(side="top")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Risk Register Detail")
    display_cols = ["control_id", "control_family", "status", "anomaly_score", "risk_score", "risk_level"]
    st.dataframe(
        risks_f.sort_values("risk_score", ascending=False)[display_cols],
        column_config={
            "risk_level": st.column_config.TextColumn("Risk Level"),
        },
        hide_index=True,
        use_container_width=True,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Control Mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("Framework Crosswalk — NIST 800-53 ↔ ISO 27001 ↔ SOC 2")

    mapping_cols = [
        "nist_800_53_id", "nist_csf_id", "iso_27001_id", "iso_27001_name",
        "soc2_criteria", "soc2_name", "mapping_strength",
    ]
    st.dataframe(
        cross_df[mapping_cols],
        column_config={
            "mapping_strength": st.column_config.TextColumn(
                "Mapping Strength",
                help="DIRECT = hardcoded seed, INFERRED = TF-IDF similarity",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("DIRECT", int((cross_df["mapping_strength"] == "DIRECT").sum()))
    mc2.metric("INFERRED", int((cross_df["mapping_strength"] == "INFERRED").sum()))
    mc3.metric("TBD (ISO)", int((cross_df["iso_27001_id"] == "TBD").sum()))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Gap Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    # Top metrics
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Controls Missing", int((controls_f["status"] == "missing").sum()))
    col_b.metric("Controls Partial", int((controls_f["status"] == "partial").sum()))
    col_c.metric("Controls Planned", int((controls_f["status"] == "planned").sum()))

    gaps_f = controls_f[controls_f["status"] != "implemented"]

    if gaps_f.empty:
        st.info("No compliance gaps with current filter selection.")
    else:
        gap_data = gaps_f.groupby(["family_code", "status"]).size().reset_index(name="count")
        fig_gap = px.bar(
            gap_data,
            x="family_code",
            y="count",
            color="status",
            barmode="stack",
            color_discrete_map={
                "missing": "#E24B4A",
                "partial": "#EF9F27",
                "planned": "#378ADD",
            },
            title="Compliance Gaps by Control Family",
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    # Average risk score per family
    avg_risk = risks_f.groupby("family_code")["risk_score"].mean().reset_index()
    fig_avg = px.bar(
        avg_risk,
        x="family_code",
        y="risk_score",
        color="risk_score",
        color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
        title="Average Risk Score by Control Family",
        labels={"risk_score": "Avg Risk Score", "family_code": "Control Family"},
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # Gap details table
    st.subheader("Gap Details")
    st.dataframe(
        gaps_f.sort_values("family_code")[
            ["control_id", "control_family", "status", "owner", "last_reviewed"]
        ],
        hide_index=True,
        use_container_width=True,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Anomaly Feed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    # ── Active Alerts ──────────────────────────────────────────────────────────
    st.subheader("🚨 Active Alerts")
    alert_display_cols = [
        "alert_id", "control_id", "control_family", "risk_level",
        "risk_score", "status", "owner", "alert_message",
    ]
    alerts_show = alerts_df[alert_display_cols].copy()

    def highlight_critical(row):
        if row["risk_level"] == "CRITICAL":
            return ["background-color: #fde8e8"] * len(row)
        return [""] * len(row)

    styled_df = alerts_show.style.apply(highlight_critical, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Anomaly Scatter ────────────────────────────────────────────────────────
    st.subheader("📊 ML Anomaly Score vs Final Risk Score")
    anomalies_df = risks_f[risks_f["anomaly_flag"] == True]  # noqa: E712

    if anomalies_df.empty:
        st.info("No anomalies with current filter selection.")
    else:
        fig_scatter = px.scatter(
            anomalies_df,
            x="anomaly_score",
            y="risk_score",
            color="risk_level",
            hover_data=["control_id", "control_family", "status"],
            title="ML Anomaly Score vs Final Risk Score",
            color_discrete_map={
                "CRITICAL": "#E24B4A",
                "HIGH": "#EF9F27",
                "MEDIUM": "#378ADD",
                "LOW": "#1D9E75",
            },
        )
        fig_scatter.update_traces(marker=dict(size=10))
        # Vertical dashed threshold line at x=0.5
        fig_scatter.add_vline(
            x=0.5, line_dash="dash", line_color="grey",
            annotation_text="Anomaly threshold",
            annotation_position="top right",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Full Anomaly Table ─────────────────────────────────────────────────────
    st.subheader("🔍 All Flagged Controls")
    if anomalies_df.empty:
        st.info("No flagged controls with current filters.")
    else:
        st.dataframe(
            anomalies_df.sort_values("anomaly_score", ascending=False)[
                ["control_id", "control_family", "status", "anomaly_score", "risk_score", "risk_level"]
            ],
            use_container_width=True,
            hide_index=True,
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Model Insights (NEW)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    # ── Section A: What the Model Learned ──────────────────────────────────────
    st.subheader("📚 NIST Training Summary")
    ma1, ma2, ma3 = st.columns(3)
    ma1.metric("Controls Trained On", len(controls_clean_df))
    ma2.metric("Anomaly Threshold", "10% contamination rate")
    ma3.metric("Vocabulary Size", "3,000 TF-IDF features")

    st.info(
        "The IsolationForest model learned the statistical 'shape' of 984 unique NIST 800-53 "
        "control descriptions. Controls in your organisation's inventory that deviate from "
        "this baseline are flagged as anomalous — indicating poorly defined, out-of-scope, "
        "or missing security language."
    )

    # ── Section B: Risk Score Distribution ─────────────────────────────────────
    st.subheader("📈 Risk Score Distribution")
    fig_hist = px.histogram(
        risks_df,
        x="risk_score",
        color="risk_level",
        nbins=40,
        color_discrete_map={
            "CRITICAL": "#E24B4A",
            "HIGH": "#EF9F27",
            "MEDIUM": "#378ADD",
            "LOW": "#1D9E75",
        },
        title="Distribution of Risk Scores Across All Controls",
        labels={"risk_score": "Risk Score (0–10)", "count": "Number of Controls"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Section C: Top 10 Highest Risk Controls ───────────────────────────────
    st.subheader("🔴 Top 10 Highest Risk Controls")
    top10 = risks_df.nlargest(10, "risk_score")[
        ["control_id", "control_family", "nist_csf_function",
         "status", "anomaly_score", "risk_score", "risk_level"]
    ]
    st.dataframe(top10, use_container_width=True, hide_index=True)

    # ── Section D: Pattern Analysis ────────────────────────────────────────────
    st.subheader("🔍 Pattern Analysis — What Makes a Control Anomalous")
    col_left, col_right = st.columns(2)

    with col_left:
        group = risks_df.groupby("nist_csf_function")["anomaly_score"].mean().reset_index()
        fig_csf = px.bar(
            group,
            x="nist_csf_function",
            y="anomaly_score",
            color="anomaly_score",
            color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
            title="Avg Anomaly Score by CSF Function",
            labels={"anomaly_score": "Avg Anomaly Score", "nist_csf_function": "CSF Function"},
        )
        st.plotly_chart(fig_csf, use_container_width=True)

    with col_right:
        group2 = risks_df.groupby("status")["anomaly_score"].mean().reset_index()
        fig_status = px.bar(
            group2,
            x="status",
            y="anomaly_score",
            color="status",
            color_discrete_map={
                "missing": "#E24B4A",
                "partial": "#EF9F27",
                "planned": "#378ADD",
                "implemented": "#1D9E75",
            },
            title="Avg Anomaly Score by Implementation Status",
            labels={"anomaly_score": "Avg Anomaly Score", "status": "Status"},
        )
        st.plotly_chart(fig_status, use_container_width=True)

    # ── Section E: Framework Coverage Intelligence ─────────────────────────────
    st.subheader("🗺️ Framework Mapping Coverage")
    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        strength_counts = cross_df["mapping_strength"].value_counts().reset_index()
        strength_counts.columns = ["mapping_strength", "count"]
        fig_pie = px.pie(
            strength_counts,
            values="count",
            names="mapping_strength",
            title="Mapping Strength Distribution",
            color="mapping_strength",
            color_discrete_map={"DIRECT": "#1D9E75", "INFERRED": "#378ADD"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ec2:
        st.markdown("**Top 3 Riskiest Families**")
        top_families = (
            risks_df.groupby("family_code")["risk_score"]
            .mean()
            .nlargest(3)
            .reset_index()
        )
        for _, row in top_families.iterrows():
            st.metric(
                f"{row['family_code']} Family",
                f"{row['risk_score']:.1f} avg risk",
            )

    with ec3:
        immediate = int(
            ((risks_df["risk_level"] == "CRITICAL")
             | ((risks_df["risk_level"] == "HIGH") & (risks_df["status"] == "missing"))).sum()
        )
        st.metric("Immediate Attention", immediate, help="CRITICAL or HIGH+missing controls")

    tbd_count = int((cross_df["iso_27001_id"] == "TBD").sum())
    st.warning(
        f"{tbd_count} controls have no direct ISO 27001 mapping. "
        "These require manual review for full compliance coverage."
    )
