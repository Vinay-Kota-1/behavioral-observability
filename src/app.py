"""
SentinelRisk Dashboard - Streamlit Live Anomaly Detection

A live dashboard that shows:
- Anomaly detections in real-time
- Score distributions
- Explanations for detected anomalies
- Business impact summaries
- Feedback interface

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from sqlalchemy import create_engine, text

# Import pipeline components
from anomaly_model import AnomalyModel, ModelConfig
from explainer import AnomalyExplainer
from business_mapper import BusinessMapper
from feedback_collector import FeedbackCollector

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="SentinelRisk - Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# =============================================================================
# Initialize Components (cached)
# =============================================================================

@st.cache_resource
def get_engine():
    return create_engine("postgresql://vinaykota:12345678@localhost:5432/fintech_lab")

@st.cache_resource
def get_model():
    engine = get_engine()
    model = AnomalyModel(engine)
    model.load()
    return model

@st.cache_resource
def get_explainer():
    return AnomalyExplainer()

@st.cache_resource
def get_mapper():
    return BusinessMapper()

@st.cache_resource
def get_feedback():
    return FeedbackCollector(get_engine())

# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(ttl=60)
def load_recent_features(limit: int = 1000):
    """Load features sampled from all sources."""
    engine = get_engine()
    with engine.connect() as conn:
        # Sample evenly from all sources
        df = pd.read_sql(text(f"""
            WITH ranked AS (
                SELECT *, 
                       ROW_NUMBER() OVER (PARTITION BY source_dataset ORDER BY RANDOM()) as rn
                FROM sentinelrisk.features
            )
            SELECT * FROM ranked 
            WHERE rn <= {limit // 4 + 1}
            LIMIT {limit}
        """), conn)
    return df

@st.cache_data(ttl=60)
def load_data_summary():
    """Load summary statistics."""
    engine = get_engine()
    with engine.connect() as conn:
        summary = pd.read_sql(text("""
            SELECT 
                source_dataset,
                COUNT(*) as event_count,
                COUNT(DISTINCT entity_id) as entity_count,
                MIN(ts) as first_event,
                MAX(ts) as last_event
            FROM sentinelrisk.features
            GROUP BY source_dataset
            ORDER BY source_dataset
        """), conn)
    return summary

# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    st.title("🔍 SentinelRisk Anomaly Detection")
    st.markdown("*Real-time anomaly detection across multiple data signals*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        sample_size = st.slider("Sample Size", 100, 5000, 1000)
        threshold = st.slider("Anomaly Threshold", 0.1, 0.9, 0.5)
        auto_refresh = st.checkbox("Auto Refresh (60s)")
        
        st.markdown("---")
        st.header("📊 Data Sources")
        summary = load_data_summary()
        for _, row in summary.iterrows():
            st.metric(
                row["source_dataset"],
                f"{row['event_count']:,} events",
                f"{row['entity_count']:,} entities"
            )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Anomalies", 
        "📈 Distributions",
        "⚡ Business Impact",
        "📝 Feedback"
    ])
    
    # Load and score data
    with st.spinner("Loading and scoring data..."):
        df = load_recent_features(sample_size)
        model = get_model()
        scored = model.predict(df)
        anomalies = scored[scored["anomaly_score"] >= threshold].copy()
        anomalies = anomalies.sort_values("anomaly_score", ascending=False)
    
    # ==========================================================================
    # Tab 1: Anomalies
    # ==========================================================================
    with tab1:
        st.header(f"Detected Anomalies ({len(anomalies)})")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scored", f"{len(df):,}")
        col2.metric("Anomalies", f"{len(anomalies):,}")
        col3.metric("Anomaly Rate", f"{len(anomalies)/len(df):.1%}")
        
        if len(anomalies) > 0:
            explainer = get_explainer()
            
            for i, (_, row) in enumerate(anomalies.head(10).iterrows()):
                exp = explainer.explain_row(row)
                
                with st.expander(
                    f"⚠️ {exp.entity_id} ({exp.source_dataset}) - Score: {exp.anomaly_score:.1%}",
                    expanded=i < 3
                ):
                    st.markdown(f"**{exp.summary}**")
                    
                    # Contributing factors
                    st.markdown("#### Contributing Factors")
                    for c in exp.top_contributors[:5]:
                        direction = "🔺" if c.direction == "high" else "🔻"
                        st.markdown(
                            f"- {direction} **{c.feature_name}**: {c.feature_value:.2f} "
                            f"(z-score: {c.zscore:+.2f})"
                        )
        else:
            st.info("No anomalies detected above threshold")
    
    # ==========================================================================
    # Tab 2: Distributions
    # ==========================================================================
    with tab2:
        st.header("Score Distributions")
        
        # Show source breakdown
        source_counts = scored.groupby("source_dataset").size()
        st.write(f"**Sample breakdown:** {dict(source_counts)}")
        
        # Score histogram colored by source
        fig = px.histogram(
            scored, 
            x="anomaly_score",
            color="source_dataset",
            nbins=50,
            title="Anomaly Score Distribution by Source",
            barmode="overlay",
            opacity=0.7
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                      annotation_text=f"Threshold: {threshold}")
        st.plotly_chart(fig, use_container_width=True)
        
        # By source - box plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                scored,
                x="source_dataset",
                y="anomaly_score",
                title="Scores by Data Source",
                color="source_dataset"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "entity_type" in scored.columns:
                fig = px.box(
                    scored,
                    x="entity_type",
                    y="anomaly_score",
                    title="Scores by Entity Type",
                    color="entity_type"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # Tab 3: Business Impact
    # ==========================================================================
    with tab3:
        st.header("Business Impact Summary")
        
        mapper = get_mapper()
        impacts = mapper.get_all_impacts()
        
        if not impacts:
            st.warning("No business rules configured")
        else:
            # Risk summary
            risk_summary = mapper.get_risk_summary()
            cols = st.columns(4)
            for i, (level, count) in enumerate(risk_summary.items()):
                colors = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                cols[i].metric(f"{colors.get(level, '')} {level.upper()}", count)
            
            st.markdown("---")
            
            # Impact details
            for name, impact in impacts.items():
                with st.expander(f"📋 {name}"):
                    col1, col2 = st.columns(2)
                    
                    col1.markdown(f"**Category:** {impact.category}")
                    col1.markdown(f"**Risk Level:** {impact.risk_level.upper()}")
                    col1.markdown(f"**Estimated Cost:** {impact.format_cost()}")
                    
                    col2.markdown(f"**Escalation:** {impact.escalation_path}")
                    col2.markdown(f"**SLA:** {impact.escalation_sla_minutes} minutes")
                    col2.markdown(f"**Contact:** {impact.escalation_contact}")
                    
                    st.markdown(f"**Recommended Action:** {impact.recommended_action}")
                    
                    if impact.compliance_flags:
                        st.markdown(f"**Compliance:** {', '.join(impact.compliance_flags)}")
    
    # ==========================================================================
    # Tab 4: Feedback
    # ==========================================================================
    with tab4:
        st.header("Model Feedback")
        
        feedback = get_feedback()
        stats = feedback.get_accuracy_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", stats.get("total_predictions", 0))
        col2.metric("With Feedback", stats.get("with_feedback", 0))
        if stats.get("with_feedback", 0) > 0:
            col3.metric("Accuracy", f"{stats.get('accuracy', 0):.1%}")
            col4.metric("Precision", f"{stats.get('precision', 0):.1%}")
        
        st.markdown("---")
        st.subheader("Pending Feedback")
        
        pending = feedback.get_pending_feedback()
        if pending:
            for pred in pending[:5]:
                with st.expander(f"{pred.entity_id} - {pred.anomaly_score:.1%}"):
                    st.write(f"Prediction ID: `{pred.prediction_id}`")
                    st.write(f"Source: {pred.source_dataset}")
                    
                    col1, col2 = st.columns(2)
                    if col1.button("✅ Correct", key=f"correct_{pred.prediction_id}"):
                        feedback.record_feedback(pred.prediction_id, 1, "Confirmed via UI")
                        st.success("Feedback recorded!")
                        st.rerun()
                    if col2.button("❌ Wrong", key=f"wrong_{pred.prediction_id}"):
                        feedback.record_feedback(pred.prediction_id, 0, "Rejected via UI")
                        st.warning("Feedback recorded - will improve model")
                        st.rerun()
        else:
            st.info("No pending feedback items")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
