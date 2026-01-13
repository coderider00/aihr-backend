import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="AIHR Analytics", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_artifacts():
    try:
        return joblib.load('aihr_models.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please run 'train_model.py' first.")
        return None

artifacts = load_artifacts()

if artifacts:
    clf = artifacts['classifier']
    reg = artifacts['regressor']
    encoders = artifacts['encoders']
    features = artifacts['feature_names']

    # --- HEADER ---
    st.title("🤖 AIHR: Employee Performance Analytics")
    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🔮 Predictive", "🧠 Prescriptive"])

    # --- TAB 1: DESCRIPTIVE (KPIs) ---
    with tab1:
        st.subheader("Workforce Overview")
        col1, col2, col3 = st.columns(3)
        # Dummy metrics for display - replace with real data calculations if you load full dataset here
        col1.metric("Avg Performance Rating", "3.2", "1.2%")
        col2.metric("Attrition Rate", "14%", "-2%")
        col3.metric("Avg Salary Hike", "15.4%", "0.5%")
        
        # Mock Data for plotting
        chart_data = pd.DataFrame({
            'Department': ['Sales', 'HR', 'R&D', 'Sales', 'HR', 'R&D'],
            'Performance': [3, 4, 3, 2, 5, 4],
            'Count': [10, 5, 15, 8, 2, 12]
        })
        
        fig = px.bar(chart_data, x='Department', y='Count', color='Performance', barmode='group', title="Performance Distribution by Dept")
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: DIAGNOSTIC (Correlations) ---
    with tab2:
        st.subheader("Why are things happening?")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Correlation Heatmap (Mock Data)")
            # Generating random correlation matrix for demo
            corr = pd.DataFrame(np.random.rand(5,5), columns=['Age', 'Salary', 'Rating', 'Satisfaction', 'Distance'])
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col2:
            st.info("Salary vs Performance")
            scatter_data = pd.DataFrame({
                'SalaryHike': np.random.randint(10, 25, 50),
                'Performance': np.random.randint(1, 5, 50),
                'Department': np.random.choice(['Sales', 'HR'], 50)
            })
            fig_scat = px.scatter(scatter_data, x='SalaryHike', y='Performance', color='Department', size='SalaryHike')
            st.plotly_chart(fig_scat, use_container_width=True)

    # --- TAB 3: PREDICTIVE (ML Models) ---
    with tab3:
        st.subheader("Future Forecast")
        
        with st.form("prediction_form"):
            st.write("Enter Employee Details:")
            inputs = {}
            cols = st.columns(3)
            
            # Dynamically generate inputs based on training features
            for i, col_name in enumerate(features):
                with cols[i % 3]:
                    if col_name in encoders:
                        # Categorical: Show Dropdown
                        options = list(encoders[col_name].classes_)
                        inputs[col_name] = st.selectbox(col_name, options)
                    else:
                        # Numerical: Show Number Input
                        inputs[col_name] = st.number_input(col_name, value=0)
            
            submitted = st.form_submit_button("Run Prediction Model")

        if submitted:
            # Process Input
            input_df = pd.DataFrame([inputs])
            
            # Encode Categorical Input
            for col, le in encoders.items():
                input_df[col] = le.transform(input_df[col])
            
            # Make Predictions
            pred_perf = clf.predict(input_df)[0]
            pred_hike = reg.predict(input_df)[0]
            
            st.success(f"Predicted Performance Rating: **{pred_perf}**")
            st.info(f"Predicted Salary Hike: **{pred_hike:.2f}%**")
            
            # Save logic for Tab 4
            st.session_state['last_pred_perf'] = pred_perf
            st.session_state['last_pred_hike'] = pred_hike

    # --- TAB 4: PRESCRIPTIVE (Recommendations) ---
    with tab4:
        st.subheader("AI Recommendations")
        
        if 'last_pred_perf' in st.session_state:
            perf = st.session_state['last_pred_perf']
            hike = st.session_state['last_pred_hike']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🎯 Performance Strategy")
                if perf < 3:
                    st.error("⚠️ Risk Alert: Low Performance")
                    st.markdown("- **Action:** Enroll in 'Skill Up' Training Program.")
                    st.markdown("- **Action:** Schedule weekly 1-on-1 mentorship.")
                else:
                    st.success("✅ High Potential")
                    st.markdown("- **Action:** Consider for Leadership Track.")
            
            with col2:
                st.markdown(f"### 💰 Compensation Strategy")
                if hike < 12 and perf >= 3:
                    st.warning("⚠️ Retention Risk: Salary Hike below market standard.")
                    st.markdown("- **Action:** Review budget for correction.")
                else:
                    st.info("Market Competitive.")
        else:
            st.write("👈 Go to the 'Predictive' tab and run a prediction to see recommendations.")

else:
    st.warning("Artifacts not loaded.")
