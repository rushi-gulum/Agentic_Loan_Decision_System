#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Agentic Loan Decision System
==============================================================

Professional loan officer interface for evaluating applications through the FastAPI backend.

Features:
- Interactive input forms based on Pydantic schemas
- Real-time API integration
- Multi-tab results visualization  
- Compliance checking and XAI explanations

Run: streamlit run frontend/streamlit_app.py
"""

import streamlit as st
import requests
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic Loan Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border: 1px solid #1f77b4;
    }
    
    .status-approved {
        color: #28a745;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .status-rejected {
        color: #dc3545;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .status-error {
        color: #fd7e14;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #fff3cd;
        border: 2px solid #fd7e14;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    
    .violation-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LoanEvaluationDashboard:
    """Main dashboard class for loan evaluation"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.api_base_url = "http://localhost:8000"
        
        # Initialize session state
        if 'evaluation_result' not in st.session_state:
            st.session_state.evaluation_result = None
        if 'last_request' not in st.session_state:
            st.session_state.last_request = None
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            🏦 Agentic Loan Decision System
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**AI-Powered Loan Evaluation with Regulatory Compliance & Explainable Decisions**")
        
        # API Status Check
        self.check_api_status()
    
    def check_api_status(self):
        """Check if FastAPI backend is running"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ FastAPI Backend is running")
            else:
                st.error("❌ FastAPI Backend returned error")
        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI Backend is not running. Please start with: `uvicorn api.app:app --reload --port 8000`")
        except Exception as e:
            st.warning(f"⚠️ Backend status unknown: {str(e)}")
    
    def render_input_form(self) -> Dict[str, Any]:
        """Render the loan application input form in sidebar"""
        st.sidebar.title("📝 Loan Application")
        st.sidebar.markdown("Fill in the applicant details below:")
        
        # Core Required Fields
        st.sidebar.subheader("👤 Applicant Information")
        
        age_years = st.sidebar.slider(
            "Age (years)", 
            min_value=18, max_value=100, value=35,
            help="Applicant age in years"
        )
        
        bureau_score = st.sidebar.slider(
            "Bureau Score", 
            min_value=300, max_value=900, value=750,
            help="Credit bureau score (300-900)"
        )
        
        monthly_income_inr = st.sidebar.number_input(
            "Monthly Income (₹)", 
            min_value=0.0, max_value=10_000_000.0, value=75000.0,
            help="Monthly income in INR"
        )
        
        # Loan Details
        st.sidebar.subheader("🏠 Loan Details")
        
        loan_type = st.sidebar.selectbox(
            "Loan Type",
            options=["housing", "personal", "vehicle", "gold", "microfinance", "msme", "business"],
            index=0,
            help="Type of loan being requested"
        )
        
        requested_amount_inr = st.sidebar.number_input(
            "Requested Amount (₹)", 
            min_value=0.0, max_value=100_000_000.0, value=500000.0,
            help="Requested loan amount in INR"
        )
        
        tenure_months = st.sidebar.slider(
            "Tenure (months)", 
            min_value=6, max_value=360, value=60,
            help="Loan tenure in months"
        )
        
        interest_rate_annual_pct = st.sidebar.slider(
            "Interest Rate (%)", 
            min_value=0.0, max_value=50.0, value=8.5, step=0.1,
            help="Annual interest rate percentage"
        )
        
        # Financial Ratios
        st.sidebar.subheader("📊 Financial Ratios")
        
        foir_total_obligations_pct = st.sidebar.slider(
            "FOIR (%)", 
            min_value=0.0, max_value=150.0, value=30.0, step=0.5,
            help="Fixed Obligation to Income Ratio percentage"
        )
        
        # Personal Details
        st.sidebar.subheader("🆔 Personal Details")
        
        gender = st.sidebar.selectbox(
            "Gender",
            options=["Male", "Female", "Other"],
            index=0
        )
        
        interest_type = st.sidebar.selectbox(
            "Interest Type",
            options=["Fixed", "Floating"],
            index=0
        )
        
        # Optional Fields
        st.sidebar.subheader("⚙️ Optional Details")
        
        pin_code = st.sidebar.number_input(
            "PIN Code", 
            min_value=100000, max_value=999999, value=110001,
            help="6-digit PIN code"
        )
        
        pep_flag = st.sidebar.checkbox(
            "Politically Exposed Person (PEP)",
            value=False,
            help="Check if applicant is a PEP"
        )
        
        kfs_provided = st.sidebar.checkbox(
            "Key Fact Statement Provided",
            value=True,
            help="Key Fact Statement provided to applicant"
        )
        
        existing_monthly_obligations_inr = st.sidebar.number_input(
            "Existing Obligations (₹)", 
            min_value=0.0, value=0.0,
            help="Existing monthly obligations in INR"
        )
        
        processing_fee_inr = st.sidebar.number_input(
            "Processing Fee (₹)", 
            min_value=0.0, value=0.0,
            help="Processing fee in INR"
        )
        
        # Build request payload
        request_data = {
            "age_years": age_years,
            "bureau_score": bureau_score,
            "monthly_income_inr": monthly_income_inr,
            "requested_amount_inr": requested_amount_inr,
            "tenure_months": tenure_months,
            "interest_rate_annual_pct": interest_rate_annual_pct,
            "foir_total_obligations_pct": foir_total_obligations_pct,
            "loan_type": loan_type,
            "gender": gender,
            "interest_type": interest_type,
            "pin_code": pin_code,
            "pep_flag": pep_flag,
            "kfs_provided": kfs_provided,
            "existing_monthly_obligations_inr": existing_monthly_obligations_inr,
            "processing_fee_inr": processing_fee_inr
        }
        
        return request_data
    
    def call_evaluation_api(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call the FastAPI evaluation endpoint"""
        try:
            with st.spinner("🔄 Evaluating loan application..."):
                response = requests.post(
                    f"{self.api_base_url}/api/v1/evaluate",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    return None
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to FastAPI backend. Please ensure it's running on port 8000.")
            return None
        except requests.exceptions.Timeout:
            st.error("⏰ Request timed out. The evaluation is taking too long.")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            return None
    
    def render_executive_summary(self, result: Dict[str, Any]):
        """Render executive summary tab"""
        st.header("📋 Executive Summary")
        
        # Decision Status
        decision = result.get("decision", "UNKNOWN")
        confidence = result.get("confidence_score", 0.0)
        decision_reason = result.get("decision_reason", "No reason provided")
        
        if decision == "APPROVED":
            st.markdown(f'<div class="status-approved">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
        elif decision == "REJECTED":
            st.markdown(f'<div class="status-rejected">❌ LOAN REJECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-error">⚠️ {decision}</div>', unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.1%}",
                help="Overall confidence in the decision"
            )
        
        with col2:
            risk_score = result.get("risk_assessment", {}).get("risk_score_10", 0.0)
            risk_grade = result.get("risk_assessment", {}).get("grade", "UNKNOWN")
            st.metric(
                label="Risk Score", 
                value=f"{risk_score:.1f}/10",
                delta=f"Grade: {risk_grade}",
                help="Risk assessment score (higher = riskier)"
            )
        
        with col3:
            processing_time = result.get("processing_time_ms", 0)
            st.metric(
                label="Processing Time",
                value=f"{processing_time}ms",
                help="Time taken to process the application"
            )
        
        # Decision Reason
        st.subheader("📄 Primary Decision Reason")
        st.info(decision_reason)
        
        # Risk Visualization
        if "risk_assessment" in result:
            self.render_risk_gauge(result["risk_assessment"])
        
        # Next Steps
        if result.get("summary", {}).get("next_steps"):
            st.subheader("📋 Recommended Next Steps")
            next_steps = result["summary"]["next_steps"]
            for i, step in enumerate(next_steps, 1):
                st.write(f"{i}. {step}")
    
    def render_risk_gauge(self, risk_data: Dict[str, Any]):
        """Render risk score gauge chart"""
        risk_score = risk_data.get("risk_score_10", 0.0)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Assessment"},
            delta = {'reference': 5.0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "lightgreen"},
                    {'range': [3, 7], 'color': "yellow"},
                    {'range': [7, 10], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 7.0
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_compliance_tab(self, result: Dict[str, Any]):
        """Render compliance and policy tab"""
        st.header("⚖️ Compliance & Policy Analysis")
        
        compliance_result = result.get("compliance_result", {})
        
        # Compliance Status
        is_compliant = compliance_result.get("is_compliant", False)
        compliance_score = compliance_result.get("compliance_score", 0.0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            status_text = "✅ COMPLIANT" if is_compliant else "❌ NON-COMPLIANT"
            status_color = "green" if is_compliant else "red"
            st.markdown(f"**Overall Status:** <span style='color: {status_color}; font-weight: bold;'>{status_text}</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("Compliance Score", f"{compliance_score:.1%}")
        
        # Hard Violations
        hard_violations = compliance_result.get("hard_violations", [])
        if hard_violations:
            st.subheader("🚨 Hard Rule Violations")
            for violation in hard_violations:
                rule_name = violation.get("rule", "Unknown Rule")
                description = violation.get("description", "No description")
                st.markdown(f"""
                <div class="violation-box">
                    <strong>Rule:</strong> {rule_name}<br>
                    <strong>Issue:</strong> {description}
                </div>
                """, unsafe_allow_html=True)
        
        # Hard Warnings
        hard_warnings = compliance_result.get("hard_warnings", [])
        if hard_warnings:
            st.subheader("⚠️ Warnings")
            for warning in hard_warnings:
                rule_name = warning.get("rule", "Unknown Rule")
                description = warning.get("description", "No description")
                st.markdown(f"""
                <div class="warning-box">
                    <strong>Rule:</strong> {rule_name}<br>
                    <strong>Warning:</strong> {description}
                </div>
                """, unsafe_allow_html=True)
        
        # Soft Compliance Scores
        soft_scores = compliance_result.get("soft_scores", {})
        if soft_scores:
            st.subheader("📊 Soft Compliance Evaluation")
            
            # Create DataFrame for better visualization
            scores_df = pd.DataFrame(list(soft_scores.items()), columns=['Category', 'Score'])
            
            fig = px.bar(scores_df, x='Category', y='Score', 
                        title='LLM Compliance Assessment by Category',
                        color='Score', color_continuous_scale='RdYlGn')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # RAG Guidelines (if available)
        rag_guidelines = compliance_result.get("rag_guidelines_used", [])
        if rag_guidelines:
            st.subheader("📚 RBI Guidelines Referenced")
            with st.expander("View Retrieved Guidelines"):
                for i, guideline in enumerate(rag_guidelines, 1):
                    st.write(f"**{i}.** {guideline}")
        
        # Compliance Explanation
        explanation = compliance_result.get("explanation", "")
        if explanation:
            st.subheader("💭 Compliance Analysis")
            st.info(explanation)
    
    def render_xai_tab(self, result: Dict[str, Any]):
        """Render XAI and explainability tab"""
        st.header("🤖 Explainable AI Analysis")
        
        xai_report = result.get("xai_report")
        if not xai_report:
            st.warning("XAI report not available for this decision")
            return
        
        # Model Information
        model_used = xai_report.get("model_used", "Unknown")
        prediction_prob = xai_report.get("prediction_probability", 0.0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Used", model_used)
        with col2:
            st.metric("Prediction Probability", f"{prediction_prob:.1%}")
        
        # Feature Importance
        feature_importance = xai_report.get("feature_importance", {})
        if feature_importance:
            st.subheader("📈 Feature Importance Analysis")
            
            # Create DataFrame and sort by importance
            importance_df = pd.DataFrame(list(feature_importance.items()), 
                                       columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title='Feature Importance Scores',
                        color='Importance', color_continuous_scale='viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Explanations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Customer Explanation")
            user_explanation = xai_report.get("user_explanation", "No explanation available")
            st.info(user_explanation)
            
            customer_summary = xai_report.get("customer_summary", "")
            if customer_summary:
                with st.expander("Detailed Customer Summary"):
                    st.write(customer_summary)
        
        with col2:
            st.subheader("🏛️ Regulator Explanation")
            regulator_explanation = xai_report.get("regulator_explanation", "No explanation available")
            st.info(regulator_explanation)
            
            regulator_summary = xai_report.get("regulator_summary", "")
            if regulator_summary:
                with st.expander("Detailed Regulator Summary"):
                    st.write(regulator_summary)
        
        # Key Factors
        key_factors = xai_report.get("key_factors", [])
        if key_factors:
            st.subheader("🔑 Key Decision Factors")
            for i, factor in enumerate(key_factors, 1):
                st.write(f"{i}. {factor}")
        
        # SHAP and LIME Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            shap_summary = xai_report.get("shap_summary", "")
            if shap_summary:
                st.subheader("📊 SHAP Analysis")
                with st.expander("View SHAP Summary"):
                    st.code(shap_summary)
        
        with col2:
            lime_explanation = xai_report.get("lime_explanation", "")
            if lime_explanation:
                st.subheader("🍋 LIME Analysis")
                with st.expander("View LIME Explanation"):
                    st.code(lime_explanation)
        
        # Top Features
        top_features = xai_report.get("top_features", [])
        if top_features:
            st.subheader("⭐ Top Contributing Features")
            for feature in top_features:
                st.write(f"• {feature}")
    
    def render_results(self, result: Dict[str, Any]):
        """Render the evaluation results in tabs"""
        if not result:
            return
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "📋 Executive Summary", 
            "⚖️ Compliance & Policy", 
            "🤖 XAI & Explainability"
        ])
        
        with tab1:
            self.render_executive_summary(result)
        
        with tab2:
            self.render_compliance_tab(result)
        
        with tab3:
            self.render_xai_tab(result)
        
        # Raw JSON viewer
        with st.expander("🔍 View Raw API Response"):
            st.json(result)
    
    def run(self):
        """Main dashboard execution"""
        self.render_header()
        
        # Sidebar form
        request_data = self.render_input_form()
        
        # Evaluation button
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Evaluate Loan Application", type="primary", use_container_width=True):
            st.session_state.last_request = request_data
            result = self.call_evaluation_api(request_data)
            if result:
                st.session_state.evaluation_result = result
                st.rerun()
        
        # Clear results button
        if st.sidebar.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.evaluation_result = None
            st.session_state.last_request = None
            st.rerun()
        
        # Display results if available
        if st.session_state.evaluation_result:
            st.markdown("---")
            self.render_results(st.session_state.evaluation_result)
        else:
            # Welcome message
            st.markdown("""
            ## Welcome to the Agentic Loan Decision System! 👋
            
            This dashboard provides AI-powered loan evaluation with:
            
            - **🔍 Intelligent Risk Assessment** - Multi-factor risk scoring with ML models
            - **⚖️ Regulatory Compliance** - Automated RBI guideline checking  
            - **🤖 Explainable AI** - SHAP and LIME analysis for transparency
            - **📊 Interactive Visualization** - Professional charts and metrics
            
            ### Getting Started
            1. Fill in the loan application details in the sidebar
            2. Click **"Evaluate Loan Application"** to process
            3. Review results in the tabbed interface
            
            ### System Requirements
            - FastAPI backend running on `localhost:8000`
            - RAG vector database initialized with RBI guidelines
            - All ML models and explainers loaded
            """)
            
            # Sample application button
            if st.button("📝 Load Sample Application"):
                st.info("Sample application loaded! Click 'Evaluate Loan Application' in the sidebar.")

def main():
    """Main entry point"""
    try:
        dashboard = LoanEvaluationDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()