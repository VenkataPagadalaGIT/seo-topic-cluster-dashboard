import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="SEO Topic Cluster Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add sidebar without image
with st.sidebar:
    st.markdown("""
    ### SEO Analytics Dashboard
    Created by an SEO professional with expertise in:
    - SEO Analysis
    - Data Visualization
    - Topic Clustering
    """)
    st.divider()

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric:hover {
        background-color: #2d3748;
    }
    </style>
""", unsafe_allow_html=True)

def process_numeric_columns(df):
    # Convert Volume to numeric, removing any non-numeric characters
    df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Convert Keyword Difficulty to numeric
    df['Keyword Difficulty'] = pd.to_numeric(df['Keyword Difficulty'], errors='coerce')
    
    return df

def calculate_keyword_value(row):
    """Calculate estimated monthly value based on volume and CPC"""
    try:
        volume = float(row['Volume'])
        difficulty = float(row['Keyword Difficulty'])
        # Estimated CTR based on difficulty (simplified model)
        estimated_ctr = max(0.1, 1 - (difficulty / 100))
        return volume * estimated_ctr
    except:
        return 0

def optimize_cluster_number(df, min_clusters=5, max_clusters=8):
    n_keywords = len(df)
    suggested_clusters = n_keywords // 1000
    return min(max_clusters, max(min_clusters, suggested_clusters))

def create_meaningful_cluster_names(df, clusters, vectorizer):
    cluster_keywords = {}
    cluster_intents = {}
    meaningful_names = {}
    
    for cluster_id in range(len(set(clusters))):
        cluster_mask = clusters == cluster_id
        cluster_keywords[cluster_id] = df[cluster_mask]['Keyword'].tolist()
        if 'Intent' in df.columns:
            cluster_intents[cluster_id] = df[cluster_mask]['Intent'].mode().iloc[0]
    
    for cluster_id, keywords in cluster_keywords.items():
        text = ' '.join(keywords).lower()
        
        if any(term in text for term in ['near me', 'local', 'nearby']):
            base_name = "Local Services"
        elif any(term in text for term in ['price', 'cost', 'cheap', 'affordable']):
            base_name = "Pricing & Deals"
        elif any(term in text for term in ['roll off', 'rolloff']):
            base_name = "Roll-Off Services"
        elif any(term in text for term in ['commercial', 'business', 'industrial']):
            base_name = "Commercial Solutions"
        elif any(term in text for term in ['residential', 'home', 'house']):
            base_name = "Residential Services"
        elif any(term in text for term in ['size', 'yard', 'cubic']):
            base_name = "Size & Capacity"
        elif any(term in text for term in ['waste', 'garbage', 'trash']):
            base_name = "Waste Management"
        elif any(term in text for term in ['rental', 'rent', 'hire']):
            base_name = "Rental Services"
        else:
            base_name = "General Services"
        
        # Create a simplified name for visualizations
        if cluster_id in cluster_intents:
            intent = cluster_intents[cluster_id]
            meaningful_names[cluster_id] = f"{base_name} - {intent}"
        else:
            meaningful_names[cluster_id] = base_name
        
        # Add top keywords without special characters
        top_keywords = Counter(' '.join(keywords).split()).most_common(3)
        top_terms = [word for word, _ in top_keywords if word not in ['dumpster', 'rental', 'rentals']][:2]
        if top_terms:
            meaningful_names[cluster_id] += f" [{', '.join(top_terms)}]"
    
    return meaningful_names

def create_topic_clusters(df):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2
    )
    X = vectorizer.fit_transform(df['Keyword'])
    
    n_clusters = optimize_cluster_number(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    cluster_names = create_meaningful_cluster_names(df, clusters, vectorizer)
    return clusters, cluster_names, vectorizer

def create_hierarchical_visualization(df):
    """Create hierarchical visualization of topics"""
    hierarchy = {
        'Commercial': ['Roll-Off Services', 'Waste Management', 'Business Solutions'],
        'Residential': ['Local Services', 'Size & Capacity', 'Home Services'],
        'Service Type': ['Rental Services', 'Pricing & Deals', 'General Services']
    }
    return hierarchy

def main():
    st.title("🎯 SEO Topic Cluster Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload your keyword research CSV file",
        type=['csv'],
        help="Upload a CSV file containing your keyword data"
    )

    if uploaded_file is not None:
        try:
            # Read and process data
            df = pd.read_csv(uploaded_file)
            
            # Ensure required columns exist
            required_columns = ['Keyword', 'Intent', 'Volume', 'Keyword Difficulty']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Convert Volume to numeric, handling various formats
            df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
            
            # Convert Keyword Difficulty to numeric, handling various formats
            df['Keyword Difficulty'] = pd.to_numeric(df['Keyword Difficulty'].astype(str).str.replace('%', ''), errors='coerce')
            
            # Fill NaN values with appropriate defaults
            df['Volume'] = df['Volume'].fillna(0)
            df['Keyword Difficulty'] = df['Keyword Difficulty'].fillna(0)
            
            # Ensure Intent is string type
            df['Intent'] = df['Intent'].astype(str)
            
            # Create topic clusters
            clusters, cluster_names, vectorizer = create_topic_clusters(df)
            df['Cluster'] = clusters
            df['Cluster_Name'] = df['Cluster'].map(cluster_names)
            
            # Calculate metrics with proper type handling
            df['Market Opportunity'] = df['Volume'] / (df['Keyword Difficulty'] + 1)
            df['Competition Score'] = df['Keyword Difficulty'] * np.log1p(df['Volume'])
            
            # Main metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Keywords", f"{len(df):,}")
            with col2:
                st.metric("Number of Topics", len(df['Cluster_Name'].unique()))
            with col3:
                st.metric("Total Search Volume", f"{df['Volume'].sum():,.0f}")
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Topic Cluster Analysis",
                "🔍 Keyword Distribution",
                "📈 Topic Insights",
                "🎯 Interactive Analysis",
                "📊 Advanced Analytics",
                "🎯 Keyword Opportunities",
                "🍎 Low Hanging Fruits"
            ])
            
            with tab1:
                st.subheader("Topic Cluster Analysis")
                
                # Enhanced cluster summary
                cluster_summary = df.groupby('Cluster_Name').agg({
                    'Keyword': 'count',
                    'Volume': ['sum', 'mean'],
                    'Keyword Difficulty': ['mean', 'min', 'max'],
                    'Market Opportunity': 'sum',
                    'Competition Score': 'mean'
                }).round(2)
                
                # Flatten column names
                cluster_summary.columns = [
                    'Keywords Count', 'Total Volume', 'Avg Volume',
                    'Avg Difficulty', 'Min Difficulty', 'Max Difficulty',
                    'Market Opportunity', 'Competition Score'
                ]
                
                cluster_summary = cluster_summary.sort_values('Market Opportunity', ascending=False)
                st.dataframe(cluster_summary)
                
                # Simplified treemap visualization
                df_treemap = df.copy()
                df_treemap['Category'] = df_treemap['Cluster_Name'].str.split(' - ').str[0]
                df_treemap = df_treemap.fillna({'Category': 'Other', 'Intent': 'Other'})
                
                # Aggregate data for treemap
                treemap_data = df_treemap.groupby(['Category', 'Cluster_Name']).agg({
                    'Volume': 'sum',
                    'Market Opportunity': 'mean',
                    'Keyword Difficulty': 'mean'
                }).reset_index()
                
                fig_treemap = px.treemap(
                    treemap_data,
                    path=['Category', 'Cluster_Name'],
                    values='Volume',
                    color='Market Opportunity',
                    title="Topic Clusters by Search Volume and Opportunity",
                    color_continuous_scale='RdYlBu'
                )
                fig_treemap.update_traces(
                    textinfo="label+value",
                    hovertemplate='<b>%{label}</b><br>Volume: %{value:,.0f}<br>Opportunity: %{color:.1f}'
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
                
                # Simplified sunburst visualization
                sunburst_data = df_treemap.groupby(['Intent', 'Category']).agg({
                    'Volume': 'sum',
                    'Keyword Difficulty': 'mean'
                }).reset_index()
                
                fig_intent = px.sunburst(
                    sunburst_data,
                    path=['Intent', 'Category'],
                    values='Volume',
                    color='Keyword Difficulty',
                    title="Intent Distribution within Topics",
                    color_continuous_scale='RdYlBu_r'
                )
                fig_intent.update_traces(
                    textinfo="label+value",
                    hovertemplate='<b>%{label}</b><br>Volume: %{value:,.0f}<br>Difficulty: %{color:.1f}'
                )
                st.plotly_chart(fig_intent, use_container_width=True)
            
            with tab2:
                st.subheader("Keyword Distribution")
                fig = px.scatter(
                    df,
                    x='Keyword Difficulty',
                    y='Volume',
                    color='Cluster_Name',
                    hover_data=['Keyword', 'Intent'],
                    title="Keyword Distribution by Topic Cluster"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Topic Insights")
                fig1 = px.bar(
                    df.groupby('Cluster_Name')['Volume'].sum().reset_index(),
                    x='Cluster_Name',
                    y='Volume',
                    title="Search Volume Distribution by Topic",
                    color='Cluster_Name'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab4:
                st.subheader("Interactive Analysis")
                
                # Metric selectors
                metrics = {
                    "Volume": "Search Volume",
                    "Keyword Difficulty": "Keyword Difficulty",
                    "Market Opportunity": "Market Opportunity",
                    "Competition Score": "Competition Score"
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    x_metric = st.selectbox("X-Axis Metric", options=list(metrics.keys()), index=0)
                with col2:
                    y_metric = st.selectbox("Y-Axis Metric", options=list(metrics.keys()), index=1)
                
                # Filters
                col3, col4, col5 = st.columns(3)
                with col3:
                    selected_intents = st.multiselect(
                        "Filter by Intent",
                        options=sorted(df['Intent'].unique()),
                        default=sorted(df['Intent'].unique())
                    )
                with col4:
                    selected_clusters = st.multiselect(
                        "Filter by Topic Cluster",
                        options=sorted(df['Cluster_Name'].unique()),
                        default=sorted(df['Cluster_Name'].unique())
                    )
                with col5:
                    volume_range = st.slider(
                        "Volume Range",
                        min_value=int(df['Volume'].min()),
                        max_value=int(df['Volume'].max()),
                        value=(int(df['Volume'].min()), int(df['Volume'].max()))
                    )
                
                # Filter data
                filtered_df = df[
                    (df['Intent'].isin(selected_intents)) &
                    (df['Cluster_Name'].isin(selected_clusters)) &
                    (df['Volume'].between(*volume_range))
                ]
                
                # Interactive scatter plot
                fig_scatter = px.scatter(
                    filtered_df,
                    x=x_metric,
                    y=y_metric,
                    color='Cluster_Name',
                    size='Volume',
                    hover_data=['Keyword', 'Intent', 'Volume', 'Keyword Difficulty'],
                    title=f"Interactive Analysis: {metrics[x_metric]} vs {metrics[y_metric]}",
                    labels=metrics
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Correlation analysis
                if len(filtered_df) > 0:
                    correlation = filtered_df[x_metric].corr(filtered_df[y_metric])
                    st.metric(
                        "Correlation",
                        f"{correlation:.2f}",
                        f"between {metrics[x_metric]} and {metrics[y_metric]}"
                    )
            
            with tab5:
                st.subheader("Advanced Analytics")
                
                # Advanced Metrics with Interactive Filters
                st.subheader("Interactive Metrics Dashboard")
                col1, col2 = st.columns(2)
                with col1:
                    difficulty_threshold = st.slider(
                        "Keyword Difficulty Threshold",
                        min_value=0,
                        max_value=100,
                        value=30,
                        help="Set threshold for low difficulty keywords"
                    )
                with col2:
                    volume_threshold = st.slider(
                        "Volume Threshold",
                        min_value=0,
                        max_value=int(df['Volume'].max()),
                        value=int(df['Volume'].mean()),
                        help="Set threshold for high volume keywords"
                    )
                
                # Dynamic Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_difficulty = df['Keyword Difficulty'].mean()
                    st.metric("Average Keyword Difficulty", f"{avg_difficulty:.1f}")
                with col2:
                    high_volume_keywords = len(df[df['Volume'] > volume_threshold])
                    st.metric("High Volume Keywords", f"{high_volume_keywords:,}")
                with col3:
                    low_difficulty_keywords = len(df[df['Keyword Difficulty'] < difficulty_threshold])
                    st.metric("Low Difficulty Keywords", f"{low_difficulty_keywords:,}")
                with col4:
                    opportunity_score = df['Market Opportunity'].mean()
                    st.metric("Average Opportunity Score", f"{opportunity_score:.1f}")

                # Interactive Keyword Difficulty Distribution
                st.subheader("Interactive Keyword Difficulty Distribution")
                difficulty_bins = st.slider(
                    "Number of Bins",
                    min_value=5,
                    max_value=50,
                    value=30,
                    help="Adjust the number of bins in the histogram"
                )
                
                fig_difficulty = px.histogram(
                    df,
                    x='Keyword Difficulty',
                    nbins=difficulty_bins,
                    title="Distribution of Keyword Difficulty",
                    color_discrete_sequence=['#1f77b4']
                )
                fig_difficulty.update_layout(
                    xaxis_title="Keyword Difficulty",
                    yaxis_title="Number of Keywords",
                    showlegend=False
                )
                st.plotly_chart(fig_difficulty, use_container_width=True)

                # Fixed Volume vs Difficulty Heatmap
                st.subheader("Volume vs Difficulty Heatmap")
                try:
                    # Create custom bins with equal frequency
                    n_bins = 5
                    volume_percentiles = np.percentile(df['Volume'], np.linspace(0, 100, n_bins + 1))
                    difficulty_percentiles = np.percentile(df['Keyword Difficulty'], np.linspace(0, 100, n_bins + 1))
                    
                    # Ensure unique bin edges
                    volume_percentiles = np.unique(volume_percentiles)
                    difficulty_percentiles = np.unique(difficulty_percentiles)
                    
                    # Create labels
                    volume_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'][:len(volume_percentiles)-1]
                    difficulty_labels = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'][:len(difficulty_percentiles)-1]
                    
                    # Create bins
                    df['Volume_Bin'] = pd.cut(df['Volume'], 
                                            bins=volume_percentiles, 
                                            labels=volume_labels, 
                                            include_lowest=True)
                    df['Difficulty_Bin'] = pd.cut(df['Keyword Difficulty'], 
                                                bins=difficulty_percentiles, 
                                                labels=difficulty_labels, 
                                                include_lowest=True)
                    
                    # Create heatmap
                    heatmap_data = pd.crosstab(df['Volume_Bin'], df['Difficulty_Bin'])
                    fig_heatmap = px.imshow(
                        heatmap_data,
                        title="Keyword Distribution Heatmap",
                        labels=dict(x="Keyword Difficulty", y="Search Volume", color="Number of Keywords"),
                        aspect="auto"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                except Exception as e:
                    st.warning("Could not create heatmap due to data distribution. Please check your data.")

                # Interactive Top Performing Clusters
                st.subheader("Interactive Cluster Performance Analysis")
                performance_metric = st.selectbox(
                    "Select Performance Metric",
                    options=['Efficiency Score', 'Volume', 'Market Opportunity', 'Competition Score'],
                    help="Choose the metric to rank clusters by"
                )
                
                cluster_performance = df.groupby('Cluster_Name').agg({
                    'Volume': 'sum',
                    'Keyword Difficulty': 'mean',
                    'Market Opportunity': 'mean',
                    'Competition Score': 'mean'
                }).round(2)
                
                cluster_performance['Efficiency Score'] = (
                    cluster_performance['Volume'] * 
                    (100 - cluster_performance['Keyword Difficulty']) / 100
                )
                
                cluster_performance = cluster_performance.sort_values(performance_metric, ascending=False)
                st.dataframe(cluster_performance)

            with tab6:
                st.subheader("Keyword Opportunities")
                
                # Opportunity Analysis
                df['Opportunity_Score'] = (
                    df['Volume'] * 
                    (100 - df['Keyword Difficulty']) / 100
                )
                
                # Top Opportunities
                st.subheader("Top Keyword Opportunities")
                top_opportunities = df.nlargest(10, 'Opportunity_Score')[['Keyword', 'Volume', 'Keyword Difficulty', 'Intent', 'Opportunity_Score']]
                top_opportunities['Opportunity_Score'] = top_opportunities['Opportunity_Score'].round(2)
                st.dataframe(top_opportunities)
                
                # Opportunity Distribution by Intent
                st.subheader("Opportunity Distribution by Intent")
                intent_opportunities = df.groupby('Intent')['Opportunity_Score'].sum().reset_index()
                fig_intent_opp = px.pie(
                    intent_opportunities,
                    values='Opportunity_Score',
                    names='Intent',
                    title="Opportunity Distribution by Intent"
                )
                st.plotly_chart(fig_intent_opp, use_container_width=True)
                
                # Opportunity vs Volume Scatter
                st.subheader("Opportunity vs Volume Analysis")
                fig_opp_vol = px.scatter(
                    df,
                    x='Volume',
                    y='Opportunity_Score',
                    color='Keyword Difficulty',
                    size='Volume',
                    hover_data=['Keyword', 'Intent'],
                    title="Keyword Opportunities by Volume and Difficulty"
                )
                st.plotly_chart(fig_opp_vol, use_container_width=True)
                
                # Competitive Analysis
                st.subheader("Competitive Analysis")
                df['Competitive_Score'] = df['Keyword Difficulty'] * np.log1p(df['Volume'])
                competitive_analysis = df.groupby('Cluster_Name').agg({
                    'Competitive_Score': 'mean',
                    'Keyword Difficulty': 'mean',
                    'Volume': 'mean'
                }).round(2)
                
                competitive_analysis = competitive_analysis.sort_values('Competitive_Score', ascending=False)
                st.dataframe(competitive_analysis)

            with tab7:
                st.subheader("🍎 Low Hanging Fruits Analysis")
                
                # Interactive Filters for Low Hanging Fruits
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_difficulty = st.slider(
                        "Maximum Keyword Difficulty",
                        min_value=0,
                        max_value=100,
                        value=30,
                        help="Set maximum difficulty for low hanging fruits"
                    )
                with col2:
                    min_volume = st.slider(
                        "Minimum Search Volume",
                        min_value=0,
                        max_value=int(df['Volume'].max()),
                        value=int(df['Volume'].mean() * 0.5),
                        help="Set minimum volume for low hanging fruits"
                    )
                with col3:
                    min_opportunity = st.slider(
                        "Minimum Opportunity Score",
                        min_value=0,
                        max_value=int(df['Market Opportunity'].max()),
                        value=int(df['Market Opportunity'].mean()),
                        help="Set minimum opportunity score"
                    )
                
                # Filter for Low Hanging Fruits
                low_hanging_fruits = df[
                    (df['Keyword Difficulty'] <= max_difficulty) &
                    (df['Volume'] >= min_volume) &
                    (df['Market Opportunity'] >= min_opportunity)
                ].copy()
                
                # Sort by Opportunity Score
                low_hanging_fruits = low_hanging_fruits.sort_values('Market Opportunity', ascending=False)
                
                # Display Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Low Hanging Fruits", f"{len(low_hanging_fruits):,}")
                with col2:
                    st.metric("Average Volume", f"{low_hanging_fruits['Volume'].mean():,.0f}")
                with col3:
                    st.metric("Average Difficulty", f"{low_hanging_fruits['Keyword Difficulty'].mean():.1f}")
                
                # Top Low Hanging Fruits
                st.subheader("Top Low Hanging Fruits")
                top_opportunities = low_hanging_fruits.nlargest(10, 'Market Opportunity')[
                    ['Keyword', 'Volume', 'Keyword Difficulty', 'Intent', 'Market Opportunity', 'Cluster_Name']
                ]
                top_opportunities['Market Opportunity'] = top_opportunities['Market Opportunity'].round(2)
                st.dataframe(top_opportunities)
                
                # Distribution by Intent
                st.subheader("Low Hanging Fruits by Intent")
                intent_distribution = low_hanging_fruits.groupby('Intent').size().reset_index(name='count')
                fig_intent_dist = px.pie(
                    intent_distribution,
                    values='count',
                    names='Intent',
                    title="Distribution of Low Hanging Fruits by Intent"
                )
                st.plotly_chart(fig_intent_dist, use_container_width=True)
                
                # Cluster Analysis
                st.subheader("Low Hanging Fruits by Cluster")
                cluster_analysis = low_hanging_fruits.groupby('Cluster_Name').agg({
                    'Keyword': 'count',
                    'Volume': 'mean',
                    'Keyword Difficulty': 'mean',
                    'Market Opportunity': 'mean'
                }).round(2)
                
                cluster_analysis.columns = ['Count', 'Avg Volume', 'Avg Difficulty', 'Avg Opportunity']
                cluster_analysis = cluster_analysis.sort_values('Count', ascending=False)
                st.dataframe(cluster_analysis)
                
                # Interactive Scatter Plot
                st.subheader("Low Hanging Fruits Analysis")
                fig_lhf = px.scatter(
                    low_hanging_fruits,
                    x='Volume',
                    y='Market Opportunity',
                    color='Keyword Difficulty',
                    size='Volume',
                    hover_data=['Keyword', 'Intent', 'Cluster_Name'],
                    title="Low Hanging Fruits: Volume vs Opportunity"
                )
                st.plotly_chart(fig_lhf, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains these columns: Keyword, Intent, Volume, Keyword Difficulty")

if __name__ == "__main__":
    main()
