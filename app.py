import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
from config.use_case_config import USE_CASES
from src.data_loader import load_data, load_metadata, save_example_data
from src.quality_checks import run_all_checks
from src.rating import get_ratings, get_overall_rating

st.set_page_config(
    page_title="Standard Data Quality Framework",
    page_icon="📊",
    layout="wide"
)

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
    st.session_state.metadata = None
    st.session_state.selected_use_case = None
    st.session_state.ratings = None
    st.session_state.overall_rating = None

def create_assets_dir():
    if not os.path.exists("assets"):
        os.makedirs("assets")

def display_metrics(ratings):
    metric_names = {
        "population_representativity": "Population Representativity",
        "metadata_granularity": "Metadata Granularity",
        "accuracy": "Accuracy",
        "coherence": "Coherence",
        "semantic_coherence_option1": "Semantic Coherence (Option 1)",
        "semantic_coherence_option2": "Semantic Coherence (Option 2)",
        "completeness": "Completeness",
        "relational_consistency": "Relational Consistency"
    }
    
    calculation_methods = {
        "population_representativity": "samples in minority class / samples in majority class",
        "metadata_granularity": "patients with metadata / total patients",
        "accuracy": "1 - (erroneous samples / total samples)",
        "coherence": "1 - (inconsistent features / total features)",
        "semantic_coherence_option1": "Number of detected columns / total columns",
        "semantic_coherence_option2": "Number of detected columns / total columns",
        "completeness": "1 - (missing values / total values)",
        "relational_consistency": "Number of duplicate rows / total rows"
    }
    
    rating_thresholds = {
        "population_representativity": """
        - Value 0.0-0.2: Rating 1/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.8-1.0: Rating 5/5
        """,
        "metadata_granularity": """
        - Value 0.0-0.2: Rating 1/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.8-1.0: Rating 5/5
        """,
        "semantic_coherence_option1": """
        - Value 0.0-0.2: Rating 1/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.8-1.0: Rating 5/5
        """,
        "semantic_coherence_option2": """
        - Value 0.0-0.2: Rating 1/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.8-1.0: Rating 5/5
        """,
        "relational_consistency": """
        - Value 0.0-0.2: Rating 1/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.8-1.0: Rating 5/5
        """,
        "accuracy": """
        - Value 0.8-1.0: Rating 5/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.0-0.2: Rating 1/5
        """,
        "coherence": """
        - Value 0.8-1.0: Rating 5/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.0-0.2: Rating 1/5
        """,
        "completeness": """
        - Value 0.8-1.0: Rating 5/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.0-0.2: Rating 1/5
        """
    }
    
    for metric, (rating, value, explanation) in ratings.items():
        display_name = metric_names.get(metric, metric)
        
        with st.expander(f"{display_name}: {rating}/5", expanded=True):
            cols = st.columns([2, 1])
            
            with cols[0]:
                st.markdown("**Explanation:**")
                st.write(explanation)
                
                st.markdown("**Calculation Method:**")
                st.write(calculation_methods.get(metric, "No calculation method available"))
                
                st.markdown("**Rating Thresholds:**")
                st.markdown(rating_thresholds.get(metric, "No threshold information available"))
            
            with cols[1]:
                # Display the rating as a gauge
                fig = px.pie(values=[rating, 5-rating], names=["Score", "Remaining"], 
                            hole=0.7, color_discrete_sequence=["#1f77b4", "#e0e0e0"])
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    annotations=[dict(text=f"{rating}/5", x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**Raw Value:** {value:.3f}")
                
                # Determine if higher is better for this metric
                positive_metrics = ["population_representativity", "metadata_granularity", 
                                   "semantic_coherence_option1", "semantic_coherence_option2", 
                                   "relational_consistency"]
                
                if metric in positive_metrics:
                    st.markdown("_Higher values are better for this metric_")
                else:
                    st.markdown("_Lower values are better for this metric_")

def display_radar_chart(ratings):
    metrics = []
    values = []
    
    for metric, (rating, _, _) in ratings.items():
        metrics.append(metric)
        values.append(rating)
    
    fig = px.line_polar(
        r=values,
        theta=metrics,
        line_close=True,
        range_r=[0, 5],
        title="Quality Ratings Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    create_assets_dir()
    
    if not os.path.exists("assets/example_data.csv"):
        save_example_data()
    
    st.title("Standard Data Quality Framework")
    
    st.sidebar.title("Use Case Selection")
    selected_use_case = st.sidebar.selectbox(
        "Select a Use Case",
        list(USE_CASES.keys()),
        index=3,  # Default to Use Case 4
        key="use_case_selector"
    )
    
    use_case_info = USE_CASES[selected_use_case]
    
    st.sidebar.markdown(f"### {use_case_info['name']}")
    st.sidebar.markdown(f"*{use_case_info['description']}*")
    
    if not use_case_info['implemented']:
        st.warning(f"⚠️ {selected_use_case} is not yet implemented in this POC. Only Use Case 4 (ASCOPD) is currently functional.")
    
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")
    uploaded_metadata = st.sidebar.file_uploader("Upload metadata (optional)", type="csv")
    
    if st.sidebar.button("Load Example Data"):
        with open("assets/example_data.csv", "rb") as f:
            st.session_state.processed_data = load_data(f)
        st.session_state.metadata = None
        st.session_state.selected_use_case = selected_use_case
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Preview")
        
        if uploaded_file is not None:
            try:
                st.session_state.processed_data = load_data(uploaded_file)
                st.session_state.selected_use_case = selected_use_case
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        if uploaded_metadata is not None:
            try:
                st.session_state.metadata = load_metadata(uploaded_metadata)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        if st.session_state.processed_data is not None:
            st.dataframe(st.session_state.processed_data, use_container_width=True)
            
            if st.button("Run Quality Checks"):
                use_case_config = USE_CASES[st.session_state.selected_use_case]
                
                if not use_case_config['implemented']:
                    st.error(f"⚠️ {st.session_state.selected_use_case} is not yet implemented. Please select Use Case 4.")
                else:
                    with st.spinner("Running quality checks..."):
                        check_results = run_all_checks(
                            st.session_state.processed_data, 
                            use_case_config, 
                            st.session_state.metadata
                        )
                        st.session_state.ratings = get_ratings(check_results)
                        st.session_state.overall_rating = get_overall_rating(st.session_state.ratings)
                    st.success("Quality checks completed!")
        else:
            st.info("Please upload a dataset or load the example data to begin.")
    
    with col2:
        st.subheader("Quality Assessment")
        
        if st.session_state.ratings is not None:
            st.markdown(f"### Overall Rating: {st.session_state.overall_rating:.1f}/5")
            
            overall_color = "green" if st.session_state.overall_rating >= 4 else "orange" if st.session_state.overall_rating >= 2.5 else "red"
            
            st.markdown(
                f"""
                <div style="
                    background-color: {overall_color};
                    padding: 10px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 20px;
                ">
                    {st.session_state.overall_rating:.1f}/5
                </div>
                """,
                unsafe_allow_html=True
            )
            
            display_radar_chart(st.session_state.ratings)
    
    # Display detailed metrics if ratings exist
    if st.session_state.ratings is not None:
        st.markdown("---")
        st.subheader("Detailed Quality Scores")
        display_metrics(st.session_state.ratings)

if __name__ == "__main__":
    main()