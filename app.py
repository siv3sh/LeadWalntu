import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import requests
from bs4 import BeautifulSoup
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from urllib.parse import urlparse
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SEO Content Quality Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">SEO Content Quality & Duplicate Detector</h1>', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Dataset Analysis", "Live Analysis", "Model Insights", "About"]
)

# Helper Functions
@st.cache_data
def load_data():
    """Load processed data"""
    try:
        extracted = pd.read_csv('data/extracted_content.csv')
        features = pd.read_csv('data/features.csv')
        duplicates = pd.read_csv('data/duplicates.csv')
        return extracted, features, duplicates
    except FileNotFoundError:
        return None, None, None

def fetch_url_content(url, timeout=10):
    """Fetch HTML content from URL"""
    try:
        # Add headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return None, str(e)

def process_raw_data():
    """Process raw data.csv and generate all outputs"""
    if not os.path.exists('data/data.csv'):
        return False, "data/data.csv not found"
    
    try:
        # Load raw data
        df = pd.read_csv('data/data.csv')
        
        # Parse HTML
        parsed_data = []
        for idx, row in df.iterrows():
            result = parse_html(row['html_content'])
            if result:
                result['url'] = row['url']
                parsed_data.append(result)
        
        extracted_df = pd.DataFrame(parsed_data)
        extracted_df = extracted_df[extracted_df['word_count'] > 0]
        
        # Extract features
        features_list = []
        for idx, row in extracted_df.iterrows():
            text = row['clean_text']
            sentences = re.split(r'[.!?]+', text)
            sentence_count = len([s for s in sentences if s.strip()])
            
            try:
                readability = textstat.flesch_reading_ease(text)
            except:
                readability = 50
            
            features_list.append({
                'url': row['url'],
                'word_count': row['word_count'],
                'sentence_count': sentence_count,
                'readability': readability,
                'clean_text': text
            })
        
        features_df = pd.DataFrame(features_list)
        
        # Assign quality labels
        def assign_quality(row):
            if row['word_count'] > 1500 and 50 <= row['readability'] <= 70:
                return 'High'
            elif row['word_count'] < 500 or row['readability'] < 30:
                return 'Low'
            else:
                return 'Medium'
        
        features_df['quality_label'] = features_df.apply(assign_quality, axis=1)
        
        # Generate embeddings and find duplicates
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = features_df['clean_text'].head(min(1000, len(features_df))).tolist()
        embeddings = model.encode(texts, show_progress_bar=False)
        
        similarity_matrix = cosine_similarity(embeddings)
        duplicates_list = []
        n = len(similarity_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > 0.8:
                    duplicates_list.append({
                        'url_1': features_df.iloc[i]['url'],
                        'url_2': features_df.iloc[j]['url'],
                        'similarity': similarity_matrix[i][j]
                    })
        
        duplicates_df = pd.DataFrame(duplicates_list)
        if len(duplicates_df) == 0:
            duplicates_df = pd.DataFrame(columns=['url_1', 'url_2', 'similarity'])
        
        # Train model
        X = features_df[['word_count', 'sentence_count', 'readability']]
        y = features_df['quality_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        ml_model.fit(X_train, y_train)
        
        # Save all files
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        extracted_df.to_csv('data/extracted_content.csv', index=False)
        features_df.to_csv('data/features.csv', index=False)
        duplicates_df.to_csv('data/duplicates.csv', index=False)
        
        with open('models/quality_model.pkl', 'wb') as f:
            pickle.dump(ml_model, f)
        
        return True, f"Successfully processed {len(extracted_df)} pages"
    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        with open('models/quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

@st.cache_resource
def load_embedder():
    """Load sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def parse_html(html_content):
    """Parse HTML and extract clean text"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""
        
        # Extract text
        content_tags = soup.find_all(['p', 'article', 'main', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        texts = [tag.get_text().strip() for tag in content_tags if tag.get_text().strip() and len(tag.get_text().strip()) > 10]
        
        clean_text = ' '.join(texts)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        word_count = len(clean_text.split())
        
        return {
            'title': title,
            'clean_text': clean_text,
            'word_count': word_count
        }
    except Exception as e:
        return None

def extract_features(text):
    """Extract SEO features from text"""
    # Basic counts
    word_count = len(text.split())
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Readability
    readability = textstat.flesch_reading_ease(text)
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'readability': readability
    }

def predict_quality(word_count, sentence_count, readability):
    """Predict content quality"""
    if word_count > 1500 and 50 <= readability <= 70:
        return "High"
    elif word_count < 500 or readability < 30:
        return "Low"
    else:
        return "Medium"

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value/3], 'color': "lightgray"},
                {'range': [max_value/3, 2*max_value/3], 'color': "gray"},
                {'range': [2*max_value/3, max_value], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

# PAGE 1: DASHBOARD
if page == "Dashboard":
    st.header("Overview Dashboard")
    
    extracted, features, duplicates = load_data()
    
    if extracted is None:
        st.warning("No processed data found.")
        
        # Show two options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Option 1: Jupyter Notebook")
            st.markdown("""
            **Recommended for large datasets**
            
            1. Open terminal
            2. Run: `jupyter notebook notebooks/seo_pipeline.ipynb`
            3. Execute all cells
            4. Come back here to view results
            
            • Best for: 1000+ pages
            • More detailed output
            • Can pause/resume
            """)
            
            if st.button("How to Run Jupyter"):
                st.code("jupyter notebook notebooks/seo_pipeline.ipynb", language="bash")
                st.info("Copy the command above and run it in your terminal")
        
        with col2:
            st.subheader("Option 2: Process Here")
            st.markdown("""
            **Quick processing in Streamlit**
            
            One-click processing directly in this dashboard.
            
            • Best for: <1000 pages
            • Faster to start
            • No terminal needed
            """)
            
            if os.path.exists('data/data.csv'):
                st.success("Found data/data.csv")
                
                if st.button("Process Data Now", type="primary"):
                    with st.spinner("Processing data... This may take a few minutes."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Loading data...")
                        progress_bar.progress(10)
                        
                        success, message = process_raw_data()
                        progress_bar.progress(100)
                        
                        if success:
                            st.success(f"Success: {message}")
                            st.info("Refreshing page...")
                            st.rerun()
                        else:
                            st.error(f"Failed: {message}")
            else:
                st.warning("data/data.csv not found")
                st.info("Place your data.csv in the data/ folder first")
    else:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Pages Analyzed",
                value=len(extracted)
            )
        
        with col2:
            avg_word_count = extracted['word_count'].mean()
            st.metric(
                label="Avg Word Count",
                value=f"{avg_word_count:.0f}"
            )
        
        with col3:
            if duplicates is not None:
                dup_count = len(duplicates)
                st.metric(
                    label="Duplicate Pairs",
                    value=dup_count
                )
            else:
                st.metric(label="Duplicate Pairs", value="N/A")
        
        with col4:
            thin_content = len(extracted[extracted['word_count'] < 500])
            st.metric(
                label="Thin Content",
                value=thin_content
            )
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Word Count Distribution")
            fig = px.histogram(
                extracted,
                x='word_count',
                nbins=50,
                title="Distribution of Word Counts",
                labels={'word_count': 'Word Count', 'count': 'Frequency'}
            )
            fig.add_vline(
                x=extracted['word_count'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {extracted['word_count'].mean():.0f}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if features is not None and 'readability' in features.columns:
                st.subheader("Readability Score Distribution")
                fig = px.histogram(
                    features,
                    x='readability',
                    nbins=40,
                    title="Flesch Reading Ease Scores",
                    labels={'readability': 'Readability Score', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2
        if features is not None and 'quality_label' in features.columns:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Quality Distribution")
                quality_counts = features['quality_label'].value_counts()
                fig = px.pie(
                    values=quality_counts.values,
                    names=quality_counts.index,
                    title="Content Quality Breakdown",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Quality by Word Count")
                fig = px.box(
                    features,
                    x='quality_label',
                    y='word_count',
                    title="Word Count Distribution by Quality",
                    color='quality_label',
                    labels={'quality_label': 'Quality', 'word_count': 'Word Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Data Preview
        st.markdown("---")
        st.subheader("Sample Data")
        st.dataframe(extracted.head(10), use_container_width=True)

# PAGE 2: DATASET ANALYSIS
elif page == "Dataset Analysis":
    st.header("Detailed Dataset Analysis")
    
    extracted, features, duplicates = load_data()
    
    if extracted is None:
        st.warning("No data available. Run the notebook first.")
    else:
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Content Stats", "Duplicates", "Correlations"])
        
        with tab1:
            st.subheader("Content Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Word Count Statistics:**")
                stats_df = extracted['word_count'].describe().to_frame()
                st.dataframe(stats_df, use_container_width=True)
                
                # Thin content analysis
                thin = len(extracted[extracted['word_count'] < 500])
                medium = len(extracted[(extracted['word_count'] >= 500) & (extracted['word_count'] < 1500)])
                rich = len(extracted[extracted['word_count'] >= 1500])
                
                st.write("**Content Length Categories:**")
                categories = pd.DataFrame({
                    'Category': ['Thin (<500)', 'Medium (500-1500)', 'Rich (>1500)'],
                    'Count': [thin, medium, rich]
                })
                st.dataframe(categories, use_container_width=True)
            
            with col2:
                if features is not None and 'readability' in features.columns:
                    st.write("**Readability Statistics:**")
                    read_stats = features['readability'].describe().to_frame()
                    st.dataframe(read_stats, use_container_width=True)
                    
                    # Readability categories
                    easy = len(features[features['readability'] >= 60])
                    moderate = len(features[(features['readability'] >= 30) & (features['readability'] < 60)])
                    difficult = len(features[features['readability'] < 30])
                    
                    st.write("**Readability Categories:**")
                    read_cat = pd.DataFrame({
                        'Category': ['Easy (≥60)', 'Moderate (30-60)', 'Difficult (<30)'],
                        'Count': [easy, moderate, difficult]
                    })
                    st.dataframe(read_cat, use_container_width=True)
            
            # Scatter plot
            if features is not None and 'readability' in features.columns:
                st.markdown("---")
                st.subheader("Word Count vs Readability")
                fig = px.scatter(
                    features,
                    x='word_count',
                    y='readability',
                    color='quality_label' if 'quality_label' in features.columns else None,
                    title="Relationship between Content Length and Readability",
                    labels={'word_count': 'Word Count', 'readability': 'Readability Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Duplicate Content Analysis")
            
            if duplicates is not None and len(duplicates) > 0:
                st.write(f"**Found {len(duplicates)} duplicate/near-duplicate pairs**")
                
                # Show duplicates table
                st.dataframe(duplicates, use_container_width=True)
                
                # Similarity distribution
                if 'similarity' in duplicates.columns:
                    fig = px.histogram(
                        duplicates,
                        x='similarity',
                        nbins=30,
                        title="Similarity Score Distribution",
                        labels={'similarity': 'Similarity Score', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No significant duplicates found (similarity > 0.8)")
        
        with tab3:
            st.subheader("Feature Correlations")
            
            if features is not None:
                numeric_features = features.select_dtypes(include=[np.number])
                if len(numeric_features.columns) > 1:
                    corr_matrix = numeric_features.corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Feature Correlation Heatmap",
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)

# PAGE 3: LIVE ANALYSIS
elif page == "Live Analysis":
    st.header("Live SEO Content Analysis")
    
    st.write("Analyze your content in real-time!")
    
    # Load models
    if st.session_state.embedder is None:
        with st.spinner("Loading AI models..."):
            st.session_state.embedder = load_embedder()
    
    # Input methods
    input_method = st.radio("Choose input method:", ["Paste HTML", "Enter URL", "Upload File"])
    
    html_content = None
    
    if input_method == "Paste HTML":
        html_content = st.text_area("Paste your HTML content here:", height=200)
    
    elif input_method == "Enter URL":
        url = st.text_input("Enter URL:", placeholder="https://example.com")
        if url and st.button("Fetch from URL", type="secondary"):
            with st.spinner(f"Fetching content from {url}..."):
                result = fetch_url_content(url)
                if isinstance(result, tuple):  # Error occurred
                    st.error(f"Failed to fetch URL: {result[1]}")
                    st.info("Common issues: Invalid URL, website blocking bots, or timeout. Try pasting HTML instead.")
                else:
                    html_content = result
                    st.success(f"Successfully fetched content from {url}")
                    st.info(f"Fetched {len(html_content)} characters")
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload HTML file", type=['html', 'htm'])
        if uploaded_file:
            html_content = uploaded_file.read().decode('utf-8')
    
    if st.button("Analyze Content", type="primary") and html_content:
        with st.spinner("Analyzing content..."):
            # Parse HTML
            parsed = parse_html(html_content)
            
            if parsed is None:
                st.error("Failed to parse HTML content. Please check your input.")
            else:
                # Extract features
                features = extract_features(parsed['clean_text'])
                
                # Predict quality
                quality = predict_quality(
                    features['word_count'],
                    features['sentence_count'],
                    features['readability']
                )
                
                # Display results
                st.success("Analysis Complete")
                
                st.markdown("---")
                st.subheader("Analysis Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Word Count", features['word_count'])
                
                with col2:
                    st.metric("Sentences", features['sentence_count'])
                
                with col3:
                    st.metric("Readability", f"{features['readability']:.1f}")
                
                with col4:
                    st.metric("Quality", quality)
                
                # Gauge charts
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig = create_gauge_chart(features['word_count'], "Word Count", 3000)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = create_gauge_chart(features['readability'], "Readability", 100)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3:
                    quality_score = {"High": 90, "Medium": 60, "Low": 30}.get(quality, 50)
                    fig = create_gauge_chart(quality_score, "Quality Score", 100)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.subheader("Recommendations")
                
                recommendations = []
                
                if features['word_count'] < 500:
                    recommendations.append("**Content is too thin.** Aim for at least 500 words for better SEO.")
                elif features['word_count'] < 1500:
                    recommendations.append("**Consider expanding content.** 1500+ words perform better in search rankings.")
                else:
                    recommendations.append("**Good content length.** Your content has sufficient depth.")
                
                if features['readability'] < 30:
                    recommendations.append("**Content is too difficult to read.** Simplify sentence structure.")
                elif features['readability'] > 70:
                    recommendations.append("**Content might be too simple.** Consider adding more depth.")
                else:
                    recommendations.append("**Good readability.** Your content is accessible to most readers.")
                
                if features['sentence_count'] > 0:
                    avg_words_per_sentence = features['word_count'] / features['sentence_count']
                    if avg_words_per_sentence > 25:
                        recommendations.append("**Sentences are too long.** Break them down for better readability.")
                    elif avg_words_per_sentence < 10:
                        recommendations.append("**Sentences are too short.** Consider combining some for better flow.")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # Preview
                st.markdown("---")
                st.subheader("Content Preview")
                st.write(f"**Title:** {parsed['title']}")
                st.text_area("Extracted Text (first 500 chars):", parsed['clean_text'][:500] + "...", height=150)

# PAGE 4: MODEL INSIGHTS
elif page == "Model Insights":
    st.header("Model Performance Insights")
    
    extracted, features, duplicates = load_data()
    model = load_model()
    
    if model is None or features is None:
        st.warning("Model or data not available. Run the notebook first.")
    else:
        # Model info
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Type:** Random Forest Classifier")
            st.write("**Features Used:**")
            st.write("- Word Count")
            st.write("- Sentence Count")
            st.write("- Readability Score")
        
        with col2:
            if hasattr(model, 'feature_importances_'):
                st.write("**Feature Importance:**")
                importance_df = pd.DataFrame({
                    'Feature': ['word_count', 'sentence_count', 'readability'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Quality distribution
        if 'quality_label' in features.columns:
            st.markdown("---")
            st.subheader("Quality Label Distribution")
            
            quality_dist = features['quality_label'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=quality_dist.index,
                    y=quality_dist.values,
                    title="Count by Quality Label",
                    labels={'x': 'Quality', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Quality Statistics:**")
                st.dataframe(quality_dist.to_frame('Count'), use_container_width=True)

# PAGE 5: ABOUT
elif page == "About":
    st.header("About This Application")
    
    st.markdown("""
    ## SEO Content Quality & Duplicate Detector
    
    This application provides comprehensive analysis of web content for SEO optimization.
    
    ### Features:
    
    1. **Content Analysis**
       - Word count analysis
       - Readability scoring (Flesch Reading Ease)
       - Sentence structure analysis
    
    2. **Quality Classification**
       - Automatic quality labeling (High/Medium/Low)
       - Machine learning-based predictions
       - Actionable recommendations
    
    3. **Duplicate Detection**
       - Semantic similarity analysis
       - Near-duplicate identification
       - Cosine similarity scoring
    
    4. **Live Analysis**
       - Real-time content analysis
       - Multiple input methods
       - Instant recommendations
    
    ### Technology Stack:
    
    - **Frontend:** Streamlit
    - **ML Models:** Scikit-learn, Sentence Transformers
    - **NLP:** BeautifulSoup, TextStat
    - **Visualization:** Plotly, Matplotlib
    
    ### How to Use:
    
    1. **Run Jupyter Notebook:** First, execute `notebooks/seo_pipeline.ipynb` to process your data
    2. **Launch Dashboard:** Run `streamlit run app.py`
    3. **Explore Data:** Navigate through different pages using the sidebar
    4. **Live Analysis:** Use the Live Analysis page to analyze new content
    
    ### Data Flow:
    
    ```
    Raw HTML → Text Extraction → Feature Engineering → ML Model → Quality Prediction
    ```
    
    ### Quality Criteria:
    
    - **High Quality:** >1500 words, readability 50-70
    - **Medium Quality:** 500-1500 words, readability 30-70
    - **Low Quality:** <500 words or readability <30
    
    ### Support:
    
    For questions or issues, please refer to the project documentation.
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** 2025
    """)
    
    # Statistics if data available
    extracted, features, duplicates = load_data()
    if extracted is not None:
        st.markdown("---")
        st.subheader("Current Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pages", len(extracted))
        with col2:
            st.metric("Avg Words", f"{extracted['word_count'].mean():.0f}")
        with col3:
            if duplicates is not None:
                st.metric("Duplicates", len(duplicates))
        with col4:
            thin = len(extracted[extracted['word_count'] < 500])
            st.metric("Thin Content", thin)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>SEO Content Quality & Duplicate Detector | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
