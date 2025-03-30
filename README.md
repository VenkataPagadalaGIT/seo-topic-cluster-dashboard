# SEO Topic Cluster Analysis Dashboard

An advanced Streamlit dashboard for analyzing and visualizing SEO keyword data, with a focus on topic clustering and opportunity identification.

## Features

- **Topic Cluster Analysis**: Automatically clusters keywords into meaningful topics using TF-IDF and K-means clustering
- **Interactive Visualizations**: 
  - Treemap and Sunburst charts for topic hierarchy
  - Scatter plots for keyword distribution
  - Heatmaps for volume vs difficulty analysis
- **Advanced Analytics**:
  - Market opportunity scoring
  - Competition analysis
  - Efficiency metrics
- **Low Hanging Fruits Analysis**: Identify high-value, low-competition keywords
- **Interactive Filters**: Customize analysis based on various metrics and thresholds

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- NumPy

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required packages:
```bash
pip install streamlit pandas plotly scikit-learn numpy
```

## Usage

1. Prepare your CSV file with the following columns:
   - Keyword
   - Intent
   - Volume
   - Keyword Difficulty

2. Run the dashboard:
```bash
streamlit run advanced_dashboard.py
```

3. Upload your CSV file through the interface

## Dashboard Sections

1. **Topic Cluster Analysis**: Overview of keyword clusters and their performance
2. **Keyword Distribution**: Visual analysis of keyword spread
3. **Topic Insights**: Detailed metrics for each topic cluster
4. **Interactive Analysis**: Customizable visualizations and filters
5. **Advanced Analytics**: Detailed performance metrics and trends
6. **Keyword Opportunities**: Identification of high-potential keywords
7. **Low Hanging Fruits**: Easy-to-target, high-value keywords

## Contributing

Feel free to submit issues and enhancement requests! 