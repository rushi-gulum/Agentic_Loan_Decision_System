"""
Cloud-Ready Streamlit Dashboard for Agentic Loan Decision System
==============================================================

Production frontend optimized for Streamlit Community Cloud deployment.
Features: Real-time API integration, responsive design, error handling.
"""

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Agentic Loan Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Configuration Management for Cloud Deployment
# ============================================================================

class Config:
    """Configuration management for different deployment environments"""
    
    def __init__(self):
        # API endpoint: check Streamlit secrets first, then env var, then default
        try:
            # Streamlit Cloud: secrets set in dashboard
            self.API_BASE_URL = st.secrets["API_BASE_URL"]
            self.is_cloud = True
        except (KeyError, FileNotFoundError):
            # Local dev: use env var or hardcoded URL
            self.API_BASE_URL = os.getenv(
                "API_BASE_URL",
                "https://loan-decision-api-y23b.onrender.com"  # actual Render URL
            )
            self.is_cloud = False
        
        self.API_ENDPOINT   = f"{self.API_BASE_URL}/api/v1/evaluate"
        self.HEALTH_ENDPOINT = f"{self.API_BASE_URL}/health"
        self.HISTORY_ENDPOINT = f"{self.API_BASE_URL}/api/v1/history"
        self.STATS_ENDPOINT   = f"{self.API_BASE_URL}/api/v1/stats"

config = Config()

def apply_custom_css():
    """Apply custom CSS for better UI"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

def show_api_status():
    """Display API connection status in sidebar"""
    try:
        response = requests.get(config.HEALTH_ENDPOINT, timeout=15)
        if response.status_code == 200:
            health = response.json()
            overall = health.get("status", "unknown")
            if overall == "healthy":
                st.sidebar.success(f"✅ API Connected")
            else:
                st.sidebar.warning(f"⚠️ API Degraded")
            
            with st.sidebar.expander("Component Status"):
                for name, comp in health.get("components", {}).items():
                    icon = "✅" if comp.get("status") == "healthy" else "⚠️"
                    st.write(f"{icon} **{name}**: {comp.get('status','?')}")
        elif response.status_code == 503:
            st.sidebar.warning("⚠️ API Starting Up...")
        else:
            st.sidebar.error(f"❌ API Error: {response.status_code}")
    except requests.exceptions.Timeout:
        st.sidebar.warning("⏳ API Waking Up (free tier cold start ~30s)")
        st.sidebar.info("Try again in 30 seconds")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("❌ API Unreachable")
    except Exception as e:
        st.sidebar.error(f"❌ {str(e)[:60]}")

def create_loan_form() -> Optional[Dict[str, Any]]:
    """Create and return loan application form data"""
    
    st.subheader("📋 Loan Application Details")
    
    # Create two columns for form layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Personal Information**")
        age_years = st.number_input("Age (years)", min_value=18, max_value=80, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=10000000, value=80000, step=5000)
        existing_obligations = st.number_input("Existing Monthly Obligations (₹)", min_value=0, max_value=500000, value=15000, step=1000)
        
        st.markdown("**Credit Information**")
        bureau_score = st.slider("Credit Bureau Score", min_value=300, max_value=850, value=720)
        pep_flag = st.checkbox("Politically Exposed Person (PEP)")
        
    with col2:
        st.markdown("**Loan Details**")
        loan_type = st.selectbox("Loan Type", ["housing", "personal", "vehicle", "gold"])
        requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=50000, max_value=50000000, value=2500000, step=50000)
        tenure_months = st.slider("Loan Tenure (months)", min_value=12, max_value=360, value=240)
        
        if loan_type == "housing":
            st.markdown("**Property Details**")
            property_value = st.number_input("Property Value (₹)", min_value=100000, max_value=100000000, value=3500000, step=100000)
        else:
            property_value = 0
        
        # Calculate derived fields
        foir = (existing_obligations / monthly_income) * 100 if monthly_income > 0 else 0
        ltv_ratio = (requested_amount / property_value) if property_value > 0 else 0.8
    
    # Display calculated metrics
    st.markdown("**Calculated Metrics**")
    met_col1, met_col2, met_col3 = st.columns(3)
    
    with met_col1:
        st.metric("FOIR", f"{foir:.1f}%", 
                 delta="Good" if foir <= 50 else "High", 
                 delta_color="normal" if foir <= 50 else "inverse")
    
    with met_col2:
        if property_value > 0:
            st.metric("LTV Ratio", f"{ltv_ratio:.2f}", 
                     delta="Good" if ltv_ratio <= 0.8 else "High",
                     delta_color="normal" if ltv_ratio <= 0.8 else "inverse")
    
    with met_col3:
        st.metric("Credit Score", bureau_score,
                 delta="Excellent" if bureau_score >= 750 else "Good" if bureau_score >= 650 else "Fair")
    
    # Submit button
    if st.button("🚀 Evaluate Loan Application", type="primary", use_container_width=True):
        # Prepare application data
        application_data = {
            "age_years": age_years,
            "gender": gender,
            "monthly_income_inr": monthly_income,
            "existing_monthly_obligations_inr": existing_obligations,
            "foir_total_obligations_pct": foir,
            "bureau_score": bureau_score,
            "pep_flag": pep_flag,
            "loan_type": loan_type,
            "requested_amount_inr": requested_amount,
            "tenure_months": tenure_months,
            "property_value_inr": property_value,
            "ltv_ratio": ltv_ratio,
            # Additional fields for complete evaluation
            "sanctioned_amount_inr": requested_amount,
            "interest_rate_annual_pct": 8.5,
            "processing_fee_inr": requested_amount * 0.01,
            "other_charges_inr": 5000,
            "proposed_emi_inr": requested_amount / tenure_months * 1.2,
            "application_month": datetime.now().month,
            "pin_code": 400001,
            "interest_type_encoded": 1,
            "ovd_provided": 1,
            "time_to_sanction_days": 7,
            "kfs_provided": 1
        }
        
        return application_data
    
    return None

def display_results(result: Dict[str, Any]):
    """Display evaluation results with rich formatting"""
    
    decision    = result.get("decision", "Unknown")
    confidence  = round(float(result.get("confidence_score", 0)) * 100, 1)
    
    if str(decision).upper() == "APPROVED":
        st.success(f"🎉 **LOAN APPROVED** — Confidence: {confidence}%")
    elif str(decision).upper() == "REJECTED":
        st.error(f"❌ **LOAN REJECTED** — Confidence: {confidence}%")
    else:
        st.warning(f"🔍 **{decision}** — Confidence: {confidence}%")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🛡️ Compliance", "📈 Risk", "🔍 Explanation"])
    
    with tab1:
        display_executive_summary(result)
    with tab2:
        display_compliance_details(result)
    with tab3:
        display_risk_assessment(result)
    with tab4:
        display_explanations(result)

def display_executive_summary(result: Dict[str, Any]):
    """Display executive summary with key metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Decision", result.get("decision", "N/A"))
    with col2:
        conf = round(float(result.get("confidence_score", 0)) * 100, 1)
        st.metric("Confidence", f"{conf}%")
    with col3:
        ms = result.get("metadata", {}).get("processing_time_ms", 0)
        st.metric("Processing Time", f"{int(ms)}ms")
    with col4:
        st.metric("Model", result.get("selected_model", "N/A"))
    
    reason = result.get("decision_reason", "")
    if reason:
        st.markdown(f"**Reason:** {reason}")

def display_compliance_details(result: Dict[str, Any]):
    """Display detailed compliance information"""
    
    compliance = result.get("compliance_result", {})
    
    # Compliance overview
    is_compliant = compliance.get("is_compliant", False)
    compliance_score = compliance.get("compliance_score", 0)
    
    col1, col2 = st.columns(2)
    with col1:
        status_color = "🟢" if is_compliant else "🔴"
        st.metric("Compliance Status", f"{status_color} {'PASS' if is_compliant else 'FAIL'}")
    with col2:
        st.metric("Compliance Score", f"{compliance_score:.2f}")
    
    # Violations and warnings
    violations = compliance.get("violations", [])
    warnings = compliance.get("warnings", [])
    
    if violations:
        st.subheader("🚨 Policy Violations")
        for violation in violations:
            st.error(f"**{violation.get('rule_type', 'Unknown')}**: {violation.get('message', 'No details available')}")
    
    if warnings:
        st.subheader("⚠️ Warnings")
        for warning in warnings:
            st.warning(f"**{warning.get('rule_type', 'Unknown')}**: {warning.get('message', 'No details available')}")
    
    if not violations and not warnings:
        st.success("✅ No compliance issues detected")

def display_risk_assessment(result: Dict[str, Any]):
    """Display risk assessment with visualizations"""
    
    risk_data = result.get("risk_assessment", {})
    
    # Risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_score = risk_data.get("risk_score", 0)
        st.metric("Risk Score", f"{risk_score:.3f}")
    
    with col2:
        risk_grade = risk_data.get("risk_grade", "N/A")
        grade_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk_grade, "⚪")
        st.metric("Risk Grade", f"{grade_color} {risk_grade}")
    
    with col3:
        processing_time = risk_data.get("processing_time_ms", 0)
        st.metric("Risk Calc Time", f"{processing_time}ms")
    
    # Risk gauge chart
    if risk_score:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Risk Score"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_explanations(result: Dict[str, Any]):
    """Display AI-generated explanations"""
    
    # New schema: xai_report.customer_summary / regulator_summary
    xai = result.get("xai_report") or {}
    
    customer = xai.get("customer_summary") or result.get("decision_reason", "")
    technical = xai.get("regulator_summary", "")
    
    if customer:
        st.subheader("👤 Customer Explanation")
        st.info(customer)
    
    if technical:
        with st.expander("🔧 Regulatory Details"):
            st.write(technical)
    
    # SHAP summary if available
    shap_summary = xai.get("shap_summary", {})
    top_features = xai.get("top_features", [])
    if top_features:
        st.subheader("🔍 Key Decision Features")
        for f in top_features[:5]:
            st.write(f"• {f}")

def create_mock_result(application_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create mock result for demo purposes when API is unavailable"""
    
    bureau_score = application_data.get("bureau_score", 700)
    foir = application_data.get("foir_total_obligations_pct", 40)
    
    # Simple mock decision logic
    if bureau_score >= 750 and foir <= 50:
        decision = "APPROVED"
        confidence = 88
    elif bureau_score < 650 or foir > 70:
        decision = "REJECTED"  
        confidence = 92
    else:
        decision = "REVIEW"
        confidence = 65
    
    return {
        "decision": decision,
        "confidence": confidence,
        "processing_time_ms": 245,
        "model_used": "Mock_XGBoost",
        "key_factors": [
            f"Bureau score: {bureau_score} ({'positive' if bureau_score >= 700 else 'negative'})",
            f"FOIR: {foir:.1f}% ({'acceptable' if foir <= 50 else 'high'})",
            "Income verification: completed",
            "Property valuation: within norms"
        ],
        "compliance_result": {
            "is_compliant": foir <= 75 and not application_data.get("pep_flag", False),
            "compliance_score": 0.85,
            "violations": [] if foir <= 75 else [{"rule_type": "FOIR_MAX", "message": "FOIR exceeds regulatory limit"}],
            "warnings": [],
            "rules_applied": ["BUREAU_MIN", "FOIR_MAX", "LTV_CAP"]
        },
        "risk_assessment": {
            "risk_score": 0.35,
            "risk_grade": "MEDIUM",
            "processing_time_ms": 45
        },
        "explanations": {
            "customer_explanation": f"Your loan application has been {decision.lower()} based on your credit profile. With a bureau score of {bureau_score} and FOIR of {foir:.1f}%, you meet our lending criteria.",
            "technical_explanation": "Decision based on XGBoost model with SHAP feature importance analysis."
        }
    }

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application function"""
    
    # Apply custom styling
    apply_custom_css()
    
    # App header
    st.markdown('<h1 class="main-header">🏦 Agentic Loan Decision System</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered loan approval with RBI regulatory compliance and explainable decisions**")
    
    # Sidebar
    st.sidebar.title("🔧 System Status")
    show_api_status()
    st.sidebar.markdown("---")
    env_label = "☁️ Cloud" if config.is_cloud else "💻 Local"
    st.sidebar.caption(f"{env_label} | {config.API_BASE_URL}")
    
    # Tabs: Evaluate / History
    tab_eval, tab_history = st.tabs(["📋 Evaluate Application", "📊 Decision History"])
    
    with tab_eval:
        # Loan application form
        application_data = create_loan_form()
        
        if application_data:
            with st.spinner("🤖 Processing loan application through AI pipeline..."):
                try:
                    response = requests.post(
                        config.API_ENDPOINT,
                        json=application_data,
                        headers={"Content-Type": "application/json"},
                        timeout=120,   # allow time for Groq + ML pipeline
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Application processed successfully!")
                        display_results(result)
                    else:
                        try:
                            err = response.json()
                        except Exception:
                            err = {"detail": response.text[:300]}
                        st.error(f"❌ API Error {response.status_code}: {err.get('detail', err)}")
                        
                except requests.exceptions.Timeout:
                    st.warning("⏳ Request timed out — the API may be waking up (free tier cold start).")
                    st.info("Please wait 30 seconds and try again. The first request after inactivity takes longer.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach the API server.")
                    if st.button("🎭 Show Demo Result (Mock Data)"):
                        st.info("📝 This is a demo result using mock data")
                        display_results(create_mock_result(application_data))
    
    with tab_history:
        st.subheader("📊 Recent Loan Decisions")
        if st.button("🔄 Load Decision History"):
            try:
                r = requests.get(config.HISTORY_ENDPOINT, params={"limit": 20}, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    decisions = data.get("decisions", [])
                    if decisions:
                        import pandas as pd
                        df = pd.DataFrame(decisions)
                        cols = ["application_id", "decision", "risk_grade", "compliance_status",
                                "loan_type", "bureau_score", "confidence", "llm_provider"]
                        df = df[[c for c in cols if c in df.columns]]
                        st.dataframe(df, use_container_width=True)
                        st.caption(f"Showing {len(decisions)} most recent decisions")
                    else:
                        st.info("No decisions logged yet. Submit a loan application first.")
                else:
                    st.error(f"Could not load history: HTTP {r.status_code}")
            except requests.exceptions.Timeout:
                st.warning("⏳ Request timed out — API may be waking up.")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()