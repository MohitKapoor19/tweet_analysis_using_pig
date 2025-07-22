# Twitter Sentiment Analysis Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
import io
from collections import Counter
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="Twitter Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DA1F2;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #657786;
        margin-bottom: 20px; /* Add some space */
    }
    .metric-container {
        background-color: #f5f8fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center; /* Center metric content */
        margin-bottom: 10px; /* Space between metrics */
    }
    .small-text {
        font-size: 0.8rem;
        color: #657786;
    }
    .insight-box {
        background-color: #e8f5fd;
        border-left: 5px solid #1DA1F2;
        padding: 10px;
        margin: 15px 0;
        border-radius: 5px;
    }
    /* Ensure columns in metric row have some gap */
    .st-emotion-cache-z5fcl4 {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📊 Twitter Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Comprehensive Twitter Data Analysis using Apache Pig & Streamlit</p>", unsafe_allow_html=True)

# --- File Paths ---
# Assume files are in the same directory as the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Pig Results
TU_FILE = os.path.join(BASE_DIR, "analysis_top_users.csv")
LT_FILE = os.path.join(BASE_DIR, "analysis_longest_tweets.csv")
SD_FILE = os.path.join(BASE_DIR, "analysis_sentiment_dist.csv")
AL_FILE = os.path.join(BASE_DIR, "analysis_avg_length.csv")
TW_FILE = os.path.join(BASE_DIR, "analysis_top_words.csv")
# Raw Data File
RAW_FILE = os.path.join(BASE_DIR, "sampled_tweets.csv")

# --- Load Data Function ---
@st.cache_data
def load_data(file_path, column_names, separator=',', parse_dates=None, date_format=None, timezone='UTC'):
    """Loads data, attempts numeric conversion, handles dates."""
    if not os.path.exists(file_path):
        st.error(f"Required file not found: {os.path.basename(file_path)}")
        return None
    if os.path.getsize(file_path) == 0:
         st.warning(f"Analysis file '{os.path.basename(file_path)}' is empty.")
         return pd.DataFrame(columns=column_names) # Return empty DF

    try:
        df = pd.read_csv(file_path, header=None, names=column_names, sep=separator, on_bad_lines='warn', encoding='utf-8')

        # Convert potential numeric columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['count', 'length', 'freq', 'score', 'avg', 'id', 'target']):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Parse dates if requested
        if parse_dates and date_format:
            for date_col in parse_dates:
                if date_col in df.columns:
                    try:
                        # Attempt parsing with specified format
                        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce', utc=True)
                        # If parsing failed, try inferring (less reliable but fallback)
                        if df[date_col].isnull().all():
                             df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='coerce', utc=True)

                        # Convert timezone if needed (and if parsing succeeded)
                        if not df[date_col].isnull().all() and df[date_col].dt.tz is not None:
                            try:
                                df[date_col] = df[date_col].dt.tz_convert(timezone)
                            except Exception as tz_e:
                                st.warning(f"Could not convert timezone for column '{date_col}': {tz_e}")
                        elif not df[date_col].isnull().all(): # If parsed but tz naive
                             df[date_col] = df[date_col].dt.tz_localize(timezone)

                    except Exception as date_e:
                        st.warning(f"Could not parse date column '{date_col}' effectively: {date_e}. Setting to NaT.")
                        df[date_col] = pd.NaT # Set to NaT if parsing fails completely
        return df

    except Exception as e:
        st.error(f"Error loading {os.path.basename(file_path)}: {e}")
        return pd.DataFrame(columns=column_names) # Return empty DF on error

# --- Load Pig Analysis Results ---
df_top_users = load_data(TU_FILE, ["user_name", "tweet_count"])
df_longest = load_data(LT_FILE, ["tweet_id", "text", "length"])
df_sentiment = load_data(SD_FILE, ["sentiment_label", "count"])
df_avg_len = load_data(AL_FILE, ["avg_len"])
df_top_words = load_data(TW_FILE, ["word", "freq"])

# --- Load Raw Data for Direct Analysis ---
raw_column_names = ['target', 'tweet_id', 'timestamp_str', 'flag', 'user_name', 'text']
# Standard Twitter API date format (if applicable, adjust if different)
# Example: Tue Apr 29 23:27:28 +0000 2025 -> '%a %b %d %H:%M:%S %z %Y'
# The example file used '%a %b %d %H:%M:%S PDT %Y' which is timezone specific and harder to parse directly.
# Using utc=True and letting pandas handle PDT offset might work, or specify format if consistent.
# Let's try a more general approach first, then the specific one if needed.
raw_date_format = '%a %b %d %H:%M:%S PDT %Y' # Keep original format for now
df_raw = load_data(RAW_FILE, raw_column_names, parse_dates=['timestamp_str'], date_format=raw_date_format, timezone='UTC') # Specify UTC target

# --- Basic data processing for analysis (No NLP) ---
@st.cache_data
def process_raw_data_light(df):
    """Process raw tweet data for basic analytics (no NLP)."""
    if df is None or df.empty:
        return None, pd.DataFrame(columns=['hashtag', 'count']), pd.DataFrame(columns=['mention', 'count'])

    processed_df = df.copy()

    # Ensure text column is string
    processed_df['text'] = processed_df['text'].astype(str)

    # Add derived time columns if timestamp is valid datetime
    if 'timestamp_str' in processed_df.columns and pd.api.types.is_datetime64_any_dtype(processed_df['timestamp_str']):
        valid_dates = processed_df['timestamp_str'].notna()
        processed_df.loc[valid_dates, 'date'] = processed_df.loc[valid_dates, 'timestamp_str'].dt.date
        processed_df.loc[valid_dates, 'hour'] = processed_df.loc[valid_dates, 'timestamp_str'].dt.hour
        processed_df.loc[valid_dates, 'day_of_week'] = processed_df.loc[valid_dates, 'timestamp_str'].dt.day_name()
        processed_df.loc[valid_dates, 'day_num'] = processed_df.loc[valid_dates, 'timestamp_str'].dt.dayofweek # For sorting
        processed_df.loc[valid_dates, 'is_weekend'] = processed_df['day_num'] >= 5
    else:
        st.warning("Timestamp column 'timestamp_str' not found or not in expected datetime format. Time-based analysis disabled.")
        # Add empty columns to prevent errors later
        processed_df['date'] = pd.NaT
        processed_df['hour'] = pd.NA
        processed_df['day_of_week'] = pd.NA
        processed_df['day_num'] = pd.NA
        processed_df['is_weekend'] = pd.NA


    # Calculate text metrics
    processed_df['text_length'] = processed_df['text'].str.len()
    processed_df['word_count'] = processed_df['text'].str.split().str.len()

    # Extract entities using Regex (more efficient way)
    hashtags = processed_df['text'].str.findall(r"#(\w+)").explode().str.lower()
    mentions = processed_df['text'].str.findall(r"@(\w+)").explode().str.lower()

    df_hashtags = pd.DataFrame(Counter(hashtags.dropna()).most_common(50), columns=['hashtag', 'count'])
    df_mentions = pd.DataFrame(Counter(mentions.dropna()).most_common(50), columns=['mention', 'count'])

    # Calculate BASIC sentiment using the 'target' column (if available)
    if 'target' in processed_df.columns:
        # Ensure target is numeric-like before mapping
        processed_df['target'] = pd.to_numeric(processed_df['target'], errors='coerce')
        sentiment_map = {0: 'Negative', 4: 'Positive'}
        processed_df['sentiment'] = processed_df['target'].map(sentiment_map) # No fillna('Neutral') unless target==2 exists
        # Handle potential NaN targets if conversion failed or value not in map
        processed_df['sentiment'] = processed_df['sentiment'].fillna('Unknown')
    else:
        processed_df['sentiment'] = 'Not Available'


    return processed_df, df_hashtags, df_mentions

# Process raw data
processed_raw, hashtags_df, mentions_df = process_raw_data_light(df_raw)

# --- Calculate Key Performance Indicators ---
@st.cache_data
def calculate_kpis(proc_df, sent_pig_df, avg_len_pig_df):
    """Calculate KPIs for dashboard using available data."""
    kpis = {}
    total_tweets_raw = len(proc_df) if proc_df is not None else 0

    # Pig Results (Primary source if available)
    if sent_pig_df is not None and not sent_pig_df.empty and 'count' in sent_pig_df.columns:
        total_tweets_pig = sent_pig_df['count'].sum()
        kpis['total_analyzed_pig'] = int(total_tweets_pig)
        sentiment_counts = dict(zip(sent_pig_df['sentiment_label'], sent_pig_df['count']))
        pos_count = sentiment_counts.get(4, 0) # Assuming 4 is positive
        neg_count = sentiment_counts.get(0, 0) # Assuming 0 is negative
        if total_tweets_pig > 0:
            kpis['positive_pct_pig'] = (pos_count / total_tweets_pig) * 100
            kpis['negative_pct_pig'] = (neg_count / total_tweets_pig) * 100
            kpis['sentiment_ratio_pig'] = pos_count / max(neg_count, 1) # Avoid division by zero
        else:
             kpis['positive_pct_pig'], kpis['negative_pct_pig'], kpis['sentiment_ratio_pig'] = 0, 0, 0
    else: # Use Raw data sentiment if pig unavailable and raw available
        if proc_df is not None and 'sentiment' in proc_df.columns and proc_df['sentiment'].nunique() > 1:
             sentiment_counts_raw = proc_df['sentiment'].value_counts()
             pos_count = sentiment_counts_raw.get('Positive', 0)
             neg_count = sentiment_counts_raw.get('Negative', 0)
             if total_tweets_raw > 0:
                 kpis['positive_pct_raw'] = (pos_count / total_tweets_raw) * 100
                 kpis['negative_pct_raw'] = (neg_count / total_tweets_raw) * 100
                 kpis['sentiment_ratio_raw'] = pos_count / max(neg_count, 1)

    if avg_len_pig_df is not None and not avg_len_pig_df.empty and 'avg_len' in avg_len_pig_df.columns:
         kpis['avg_len_pig'] = avg_len_pig_df['avg_len'].iloc[0]

    # From Processed Raw Data
    if proc_df is not None and not proc_df.empty:
        kpis['total_raw_tweets'] = total_tweets_raw
        kpis['unique_users'] = proc_df['user_name'].nunique()
        kpis['avg_text_length_raw'] = proc_df['text_length'].mean() if 'text_length' in proc_df else 0
        kpis['avg_word_count_raw'] = proc_df['word_count'].mean() if 'word_count' in proc_df else 0
        # Calculate rates based on non-empty lists from findall
        kpis['hashtag_rate'] = proc_df['text'].str.contains(r"#\w+").sum() / total_tweets_raw * 100 if total_tweets_raw > 0 else 0
        kpis['mention_rate'] = proc_df['text'].str.contains(r"@\w+").sum() / total_tweets_raw * 100 if total_tweets_raw > 0 else 0

        # Fix for the date issue: check for valid dates before calculating min/max
        if 'date' in proc_df.columns:
            # Filter out NaN values before calculating min/max
            valid_dates = proc_df['date'].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                
                # Only process if both dates are valid
                if pd.notna(min_date) and pd.notna(max_date):
                    try:
                        date_range_days = (max_date - min_date).days + 1  # Add 1 to include start/end days
                        kpis['date_range_days'] = max(date_range_days, 1)
                        kpis['tweets_per_day'] = total_tweets_raw / kpis['date_range_days'] if kpis['date_range_days'] > 0 else 0
                        kpis['start_date'] = min_date
                        kpis['end_date'] = max_date
                    except Exception as e:
                        st.warning(f"Error calculating date range: {e}")

    return kpis

# Calculate KPIs
# Map sentiment labels for Pig data display if available
if df_sentiment is not None and not df_sentiment.empty:
    sentiment_map = {0: 'Negative', 4: 'Positive'} # Assuming 0=Neg, 4=Pos
    df_sentiment['sentiment'] = df_sentiment['sentiment_label'].map(sentiment_map).fillna('Unknown')

kpis = calculate_kpis(processed_raw, df_sentiment, df_avg_len)

# --- Sidebar ---
st.sidebar.header("Dashboard Controls")

# Status indicators
st.sidebar.subheader("Data Sources Status")
pig_status = {
    "Top Users": df_top_users is not None and not df_top_users.empty,
    "Longest Tweets": df_longest is not None and not df_longest.empty,
    "Sentiment Dist": df_sentiment is not None and not df_sentiment.empty,
    "Avg Length": df_avg_len is not None and not df_avg_len.empty,
    "Top Words": df_top_words is not None and not df_top_words.empty,
}
raw_status = df_raw is not None and not df_raw.empty

st.sidebar.markdown("**Pig Analysis Files:**")
for name, loaded in pig_status.items():
    st.sidebar.markdown(f"- {name}: {'✅ Loaded' if loaded else '❌ Missing/Empty'}")

st.sidebar.markdown("**Raw Data File:**")
st.sidebar.markdown(f"- Sampled Tweets: {'✅ Loaded (' + str(kpis.get('total_raw_tweets', 0)) + ' rows)' if raw_status else '❌ Missing/Empty'}")
if raw_status and 'start_date' in kpis:
    st.sidebar.markdown(f"- Date Range: {kpis.get('start_date', 'N/A')} to {kpis.get('end_date', 'N/A')}")
elif raw_status:
     st.sidebar.markdown(f"- Date Range: (Timestamp unreadable)")


# Display average length (prefer Pig, fallback to raw)
avg_len_display = kpis.get('avg_len_pig', kpis.get('avg_text_length_raw'))
if avg_len_display is not None and avg_len_display > 0:
    source = "(Pig)" if 'avg_len_pig' in kpis else "(Raw)"
    st.sidebar.metric(f"Avg Tweet Length {source}", f"{avg_len_display:.1f} chars")
else:
    st.sidebar.text("Avg Tweet Length: N/A")


# Theme selector (Simplified CSS injection)
theme = st.sidebar.selectbox(
    "Dashboard Theme",
    ["Default", "Light Blue"] # Add more themes later if needed
)

# Simple theme adjustments (add more complex CSS if needed)
if theme == "Light Blue":
    st.markdown("""
    <style>
        .metric-container { background-color: #e8f5fd; }
    </style>
    """, unsafe_allow_html=True)


# --- Main Dashboard Tabs ---
tab_titles = [
    "📊 Overview",
    "🕒 Time Analysis",
    "💬 Content Analysis",
    "👤 User Analysis",
    "📋 Data Explorer"
]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

# --- Tab 1: Overview ---
with tab1:
    st.header("Dashboard Overview")

    # KPI Metrics Row
    st.subheader("Key Metrics")
    kpi_cols = st.columns(4)

    # Define metrics with fallbacks
    metrics_to_show = [
        {"label": "Total Tweets", "value": kpis.get('total_analyzed_pig', kpis.get('total_raw_tweets')), "format": "{:,.0f}", "source": "(Pig)" if 'total_analyzed_pig' in kpis else "(Raw)"},
        {"label": "Unique Users", "value": kpis.get('unique_users'), "format": "{:,.0f}", "source": "(Raw)"},
        {"label": "Positive Sentiment", "value": kpis.get('positive_pct_pig', kpis.get('positive_pct_raw')), "format": "{:.1f}%", "source": "(Pig)" if 'positive_pct_pig' in kpis else "(Raw)"},
        {"label": "Avg Tweet Length", "value": kpis.get('avg_len_pig', kpis.get('avg_text_length_raw')), "format": "{:.1f} chars", "source": "(Pig)" if 'avg_len_pig' in kpis else "(Raw)"}
    ]

    for i, metric_info in enumerate(metrics_to_show):
        with kpi_cols[i % len(kpi_cols)]:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            value = metric_info['value']
            if value is not None:
                 display_val = metric_info['format'].format(value)
                 st.metric(label=f"{metric_info['label']} {metric_info['source']}", value=display_val)
            else:
                 st.metric(label=metric_info['label'], value="N/A")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---") # Separator
    st.subheader("Key Visualizations")

    # Two column layout for charts
    col1, col2 = st.columns(2)

    with col1:
        # Sentiment Distribution (Using Pig data preferably)
        st.markdown("#### Sentiment Distribution")
        sentiment_data_source = None
        if df_sentiment is not None and not df_sentiment.empty and 'sentiment' in df_sentiment.columns:
             sentiment_data_source = df_sentiment
             sentiment_label_col = 'sentiment'
             sentiment_count_col = 'count'
             sentiment_title = "Tweet Sentiment Distribution (Pig)"
             source_text = "Pig analysis"
             positive_pct = kpis.get('positive_pct_pig', 0)
             sentiment_ratio = kpis.get('sentiment_ratio_pig', 0)
        elif processed_raw is not None and 'sentiment' in processed_raw.columns and processed_raw['sentiment'].nunique() > 1 and processed_raw['sentiment'].value_counts().get('Positive',0) > 0:
            st.info("Using sentiment from raw data 'target' column as Pig result is unavailable.")
            raw_sentiment_counts = processed_raw['sentiment'].value_counts().reset_index()
            raw_sentiment_counts.columns = ['sentiment', 'count']
            sentiment_data_source = raw_sentiment_counts[raw_sentiment_counts['sentiment'].isin(['Positive', 'Negative'])] # Filter out Unknown/Neutral if needed
            sentiment_label_col = 'sentiment'
            sentiment_count_col = 'count'
            sentiment_title = "Tweet Sentiment Distribution (Raw Data Target)"
            source_text = "raw data analysis"
            positive_pct = kpis.get('positive_pct_raw', 0)
            sentiment_ratio = kpis.get('sentiment_ratio_raw', 0)


        if sentiment_data_source is not None and not sentiment_data_source.empty:
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_data_source[sentiment_label_col],
                values=sentiment_data_source[sentiment_count_col],
                hole=.4,
                marker_colors=['#E0245E', '#1DA1F2'] # Assuming Negative, Positive order or map colors explicitly
            )])
            fig.update_layout(
                title=sentiment_title,
                annotations=[dict(text='SENTIMENT', x=0.5, y=0.5, font_size=16, showarrow=False)],
                height=350, margin=dict(t=50, b=0, l=0, r=0),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Add insight box
            st.markdown(f"""
            <div class='insight-box'>
                <strong>Insight ({source_text}):</strong> {positive_pct:.1f}% positive sentiment.
                Sentiment ratio (Pos:Neg) is {sentiment_ratio:.2f}.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Sentiment distribution data not available or calculable.")

    with col2:
        # Top Users Visualization (Using Pig data)
        st.markdown("#### Most Active Users")
        if df_top_users is not None and not df_top_users.empty:
            # Ensure numeric type for color scaling
            df_top_users['tweet_count'] = pd.to_numeric(df_top_users['tweet_count'], errors='coerce')
            df_top_users.dropna(subset=['tweet_count'], inplace=True)

            fig = px.bar(
                df_top_users.head(10), # Show top 10
                x="user_name",
                y="tweet_count",
                text="tweet_count",
                color="tweet_count",
                color_continuous_scale="Blues",
                title="Most Active Twitter Users (Pig)"
            )
            fig.update_layout(
                xaxis_title="Username",
                yaxis_title="Number of Tweets",
                xaxis_tickangle=-45,
                height=350, margin=dict(t=50, b=0, l=0, r=0),
                coloraxis_showscale=False
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # Add insight
            most_active = df_top_users.iloc[0]['user_name']
            most_tweets = df_top_users.iloc[0]['tweet_count']
            st.markdown(f"""
            <div class='insight-box'>
                <strong>Insight (Pig):</strong> User <code>{most_active}</code> is the most active with {int(most_tweets)} tweets.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Top users data (Pig analysis) not available.")

    st.markdown("---")
    # Top Words Bar Chart (No Word Cloud)
    st.markdown("#### Top Words Analysis (Pig)")
    if df_top_words is not None and not df_top_words.empty:
        top_n_words = 15
        df_top_words['freq'] = pd.to_numeric(df_top_words['freq'], errors='coerce')
        df_top_words.dropna(subset=['freq'], inplace=True)

        fig = px.bar(
            df_top_words.head(top_n_words).sort_values('freq', ascending=True), # Horizontal looks better sorted asc
            y="word",
            x="freq",
            orientation='h',
            color="freq",
            color_continuous_scale="Viridis",
            text="freq",
            title=f"Top {top_n_words} Words in Tweets (Pig)"
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Frequency",
            height=400,
            coloraxis_showscale=False,
            margin=dict(t=50, b=20, l=10, r=10) # Adjust margins
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Word frequency data (Pig analysis) not available.")

    # Longest Tweets Section
    st.markdown("#### Notable Tweets (Longest - Pig)")
    if df_longest is not None and not df_longest.empty:
        st.markdown("<div class='metric-container' style='padding: 5px;'>", unsafe_allow_html=True) # Use container for styling
        df_display = df_longest.copy()
        # Basic cleaning for display
        df_display['text'] = df_display['text'].astype(str).str.replace(r'&quot;', '"', regex=False).str.replace(r'&amp;', '&', regex=False)
        df_display = df_display.rename(columns={"tweet_id": "ID", "text": "Content", "length": "Length"})
        st.dataframe(df_display[['ID', 'Length', 'Content']].head(), use_container_width=True, height=180) # Show limited columns/rows
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Longest tweets data (Pig analysis) not available.")

# --- Tab 2: Time Analysis ---
with tab2:
    st.header("Time Series Analysis (Raw Data)")

    if processed_raw is not None and 'timestamp_str' in processed_raw.columns and pd.api.types.is_datetime64_any_dtype(processed_raw['timestamp_str']) and processed_raw['timestamp_str'].notna().any():
        df_time_analysis = processed_raw.dropna(subset=['timestamp_str']).copy()

        # Ensure date objects for comparison
        df_time_analysis['date_only'] = df_time_analysis['timestamp_str'].dt.date

        min_date = df_time_analysis['date_only'].min()
        max_date = df_time_analysis['date_only'].max()

        # Date Range Selector
        st.info("Select a date range to analyze temporal patterns in the raw data.")
        date_range = st.date_input(
             "Select Date Range:",
             value=(min_date, max_date) if min_date != max_date else [min_date], # Handle single day case
             min_value=min_date,
             max_value=max_date,
             key="time_date_range" # Unique key
        )

        # Handle single date selection or range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        elif isinstance(date_range, list) and len(date_range) == 1:
             start_date = end_date = date_range[0]
        elif isinstance(date_range, datetime.date): # Handle case where only one date might be returned initially
             start_date = end_date = date_range
        else:
            st.warning("Please select a valid date range.")
            start_date, end_date = None, None # Avoid processing if range is invalid

        if start_date and end_date:
            filtered_df = df_time_analysis[(df_time_analysis['date_only'] >= start_date) &
                                           (df_time_analysis['date_only'] <= end_date)]

            if filtered_df.empty:
                st.warning("No data available for the selected date range.")
            else:
                # Daily Tweet Volume over Time
                st.subheader("Daily Tweet Volume")
                daily_counts = filtered_df.groupby('date_only').size().reset_index(name='count')

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_counts['date_only'], y=daily_counts['count'], name="Tweet Volume",
                    mode='lines+markers', line=dict(width=1, color='rgb(131, 90, 241)'),
                    fill='tozeroy', fillcolor='rgba(131, 90, 241, 0.2)'
                ))
                # Add Moving Average
                if len(daily_counts) >= 3:
                     daily_counts['moving_avg'] = daily_counts['count'].rolling(window=3, min_periods=1, center=True).mean()
                     fig.add_trace(go.Scatter(
                         x=daily_counts['date_only'], y=daily_counts['moving_avg'], name="3-day MA",
                         mode='lines', line=dict(width=2, color='rgb(255, 102, 0)')
                     ))

                fig.update_layout(
                    title="Daily Tweet Volume Over Selected Range",
                    xaxis_title="Date", yaxis_title="Number of Tweets",
                    hovermode="x unified", height=350, margin=dict(t=50, b=20, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Hourly and Daily Patterns
                st.subheader("Activity Patterns")
                col1, col2 = st.columns(2)
                with col1:
                    # Hourly analysis
                    hourly_counts = filtered_df.groupby('hour').size().reset_index(name='count')
                    if not hourly_counts.empty:
                        fig = px.bar(hourly_counts, x='hour', y='count', title="Tweets by Hour of Day",
                                     labels={"hour": "Hour (0-23)", "count": "Tweet Count"},
                                     color='count', color_continuous_scale="Viridis")
                        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2), coloraxis_showscale=False, height=300, margin=dict(t=50, b=20, l=10, r=10))
                        st.plotly_chart(fig, use_container_width=True)

                        peak_hour = hourly_counts.loc[hourly_counts['count'].idxmax()]['hour']
                        st.markdown(f"<div class='insight-box'><strong>Peak Hour:</strong> Most activity occurs around {int(peak_hour)}:00.</div>", unsafe_allow_html=True)
                    else:
                         st.info("No hourly data to display.")

                with col2:
                    # Day of week analysis
                    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    daily_pattern = filtered_df.groupby(['day_num', 'day_of_week']).size().reset_index(name='count').sort_values('day_num')
                    if not daily_pattern.empty:
                        fig = px.bar(daily_pattern, x='day_of_week', y='count', title="Tweets by Day of Week",
                                     labels={"day_of_week": "Day", "count": "Tweet Count"},
                                     color='count', color_continuous_scale="Viridis",
                                     category_orders={"day_of_week": days_order})
                        fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False, height=300, margin=dict(t=50, b=20, l=10, r=10))
                        st.plotly_chart(fig, use_container_width=True)

                        weekend_count = filtered_df[filtered_df['is_weekend'] == True].shape[0]
                        total_count = len(filtered_df)
                        weekend_ratio = weekend_count / total_count if total_count > 0 else 0
                        st.markdown(f"<div class='insight-box'><strong>Weekend Activity:</strong> {weekend_ratio:.1%} of tweets occur on weekends.</div>", unsafe_allow_html=True)
                    else:
                        st.info("No day-of-week data to display.")

                # Heatmap
                st.subheader("Activity Heatmap: Hour vs. Day")
                heatmap_data = filtered_df.groupby(['day_num', 'hour']).size().unstack(fill_value=0)
                # Reindex to ensure all days/hours are present and in order
                heatmap_data = heatmap_data.reindex(index=range(7), columns=range(24), fill_value=0)
                if not heatmap_data.empty:
                     fig = px.imshow(
                         heatmap_data,
                         labels=dict(x="Hour of Day", y="Day of Week", color="Tweet Count"),
                         y=days_order, x=[f"{h}" for h in range(24)],
                         color_continuous_scale="Viridis"
                     )
                     fig.update_layout(xaxis=dict(side="top", tickmode='linear', tick0=0, dtick=2), height=350, margin=dict(t=50, b=0, l=0, r=0))
                     st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("No data for activity heatmap.")
        else:
              st.warning("Select a valid date range to view time analysis.") # Handles the initial state or invalid range

    else:
        st.error("Time series analysis requires valid 'timestamp_str' data in the raw tweets dataset (sampled_tweets.csv).")

# --- Tab 3: Content Analysis ---
with tab3:
    st.header("Content Analysis (Raw Data)")

    if processed_raw is not None and 'text' in processed_raw.columns:
        content_tabs = st.tabs(["Hashtags", "Mentions", "Text Patterns"])

        with content_tabs[0]: # Hashtags
            st.subheader("Hashtag Analysis")
            if hashtags_df is not None and not hashtags_df.empty:
                col1, col2 = st.columns([3, 1]) # Adjust column ratio
                with col1:
                    fig = px.bar(hashtags_df.head(15).sort_values('count'), y="hashtag", x="count", orientation='h',
                                 color="count", color_continuous_scale="Blues", title="Top 15 Hashtags")
                    fig.update_layout(yaxis_title="", xaxis_title="Frequency", height=400, coloraxis_showscale=False, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    total_hashtags = hashtags_df['count'].sum()
                    unique_hashtags = len(hashtags_df)
                    st.markdown("<div class='metric-container' style='height: 400px;'>", unsafe_allow_html=True) # Match height roughly
                    st.markdown(f"<h6>Hashtag Stats</h6><hr>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Total Uses:</strong> {total_hashtags}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Unique Tags:</strong> {unique_hashtags}</p>", unsafe_allow_html=True)
                    if kpis.get('total_raw_tweets', 0) > 0:
                         st.markdown(f"<p><strong>Avg per Tweet:</strong> {total_hashtags / kpis['total_raw_tweets']:.2f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Most Popular:</strong><br>#{hashtags_df.iloc[0]['hashtag']} ({hashtags_df.iloc[0]['count']} uses)</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No hashtags found in the raw dataset.")

        with content_tabs[1]: # Mentions
            st.subheader("Mentions Analysis")
            if mentions_df is not None and not mentions_df.empty:
                col1, col2 = st.columns([3, 1]) # Adjust column ratio
                with col1:
                    fig = px.bar(mentions_df.head(15).sort_values('count'), y="mention", x="count", orientation='h',
                                 color="count", color_continuous_scale="Greens", title="Top 15 Mentioned Users")
                    fig.update_layout(yaxis_title="", xaxis_title="Frequency", height=400, coloraxis_showscale=False, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    total_mentions = mentions_df['count'].sum()
                    unique_mentions = len(mentions_df)
                    st.markdown("<div class='metric-container' style='height: 400px;'>", unsafe_allow_html=True) # Match height roughly
                    st.markdown(f"<h6>Mention Stats</h6><hr>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Total Uses:</strong> {total_mentions}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Unique Mentions:</strong> {unique_mentions}</p>", unsafe_allow_html=True)
                    if kpis.get('total_raw_tweets', 0) > 0:
                         st.markdown(f"<p><strong>Avg per Tweet:</strong> {total_mentions / kpis['total_raw_tweets']:.2f}</p>", unsafe_allow_html=True)
                # Tweet length distribution
                if 'text_length' in processed_raw.columns:
                    fig = px.histogram(processed_raw, x="text_length", nbins=20, title="Distribution of Tweet Lengths",
                                       labels={"text_length": "Character Count", "count": "Frequency"}, color_discrete_sequence=["#1DA1F2"])
                    fig.update_layout(height=300, margin=dict(t=50, b=20, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Text length data unavailable.")
            with col2:
                # Word count distribution
                if 'word_count' in processed_raw.columns:
                    fig = px.histogram(processed_raw, x="word_count", nbins=15, title="Distribution of Word Counts",
                                       labels={"word_count": "Word Count", "count": "Frequency"}, color_discrete_sequence=["#17bf63"])
                    fig.update_layout(height=300, margin=dict(t=50, b=20, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("Word count data unavailable.")

            # URL/Link Analysis
            st.markdown("#### Link Analysis")
            url_pattern = r'https?://[^\s]+'
            processed_raw['has_url'] = processed_raw['text'].str.contains(url_pattern)
            tweets_with_urls = processed_raw['has_url'].sum()
            total_tweets = len(processed_raw)
            url_pct = (tweets_with_urls / total_tweets * 100) if total_tweets > 0 else 0

            st.markdown(f"""
            <div class='metric-container'>
                <p><strong>Tweets containing URLs:</strong> {tweets_with_urls:,}</p>
                <p><strong>Percentage of tweets with links:</strong> {url_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("Content analysis requires the raw tweets dataset (sampled_tweets.csv) with a 'text' column.")


# --- Tab 4: User Analysis ---
with tab4:
    st.header("User Analysis (Raw Data)")

    if processed_raw is not None and 'user_name' in processed_raw.columns:
        user_activity = processed_raw['user_name'].value_counts().reset_index()
        user_activity.columns = ['user_name', 'tweet_count']

        total_users = user_activity['user_name'].nunique()
        tweets_per_user = len(processed_raw) / total_users if total_users > 0 else 0
        one_time_users = user_activity[user_activity['tweet_count'] == 1].shape[0]
        power_users = total_users - one_time_users # Users with > 1 tweet
        one_time_pct = one_time_users / total_users * 100 if total_users > 0 else 0
        power_user_pct = 100 - one_time_pct

        st.subheader("User Engagement Metrics")
        metric_cols = st.columns(4)
        user_metrics = [
            {"label": "Total Unique Users", "value": total_users, "format": "{:,.0f}"},
            {"label": "Avg Tweets per User", "value": tweets_per_user, "format": "{:.2f}"},
            {"label": "One-Time Posters", "value": one_time_users, "format": "{:,.0f}"},
            {"label": "Repeat Posters (>1)", "value": power_users, "format": "{:,.0f}"}
        ]
        for i, metric_info in enumerate(user_metrics):
             with metric_cols[i % len(metric_cols)]:
                 st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                 value = metric_info['value']
                 if value is not None:
                      display_val = metric_info['format'].format(value)
                      st.metric(label=metric_info['label'], value=display_val)
                 else:
                      st.metric(label=metric_info['label'], value="N/A")
                 st.markdown("</div>", unsafe_allow_html=True)


        st.subheader("User Activity Distribution")
        col1, col2 = st.columns([3, 2]) # Adjust ratio
        with col1:
            # Histogram of user tweet counts (log scale can be useful here if skewed)
            fig = px.histogram(user_activity, x="tweet_count", nbins=30, # More bins might be needed
                               title="Distribution of Tweets per User",
                               labels={"tweet_count": "Number of Tweets Posted by User", "count": "Number of Users"})
            fig.update_layout(yaxis_title="Number of Users", height=350, margin=dict(t=50, b=20, l=10, r=10))
            # Optional: Use log scale if distribution is highly skewed
            # fig.update_yaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Pie chart for user types
            if total_users > 0:
                 user_types_df = pd.DataFrame({
                      'type': ['One-time Users', 'Repeat Posters'],
                      'count': [one_time_users, power_users]
                 })
                 fig = go.Figure(data=[go.Pie(labels=user_types_df['type'], values=user_types_df['count'], hole=.4,
                                              marker_colors=['#AAB8C2', '#1DA1F2'])]) # Grey, Blue
                 fig.update_layout(title="User Engagement Type", height=350, margin=dict(t=50, b=20, l=10, r=10),
                                   annotations=[dict(text='USERS', x=0.5, y=0.5, font_size=16, showarrow=False)],
                                   legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
                 st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("No user data for engagement breakdown.")

        st.subheader("Top Active Users (Raw Data)")
        if not user_activity.empty:
            top_n_users = 15
            fig = px.bar(user_activity.head(top_n_users), x='user_name', y='tweet_count',
                         title=f"Top {top_n_users} Most Active Users (from Raw Data)",
                         color='tweet_count', color_continuous_scale='Blues',
                         labels={"user_name": "Username", "tweet_count": "Tweet Count"})
            fig.update_layout(xaxis_tickangle=-45, height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("No user activity data found.")

    else:
        st.error("User analysis requires the raw tweets dataset (sampled_tweets.csv) with a 'user_name' column.")


# --- Tab 5: Data Explorer ---
with tab5:
    st.header("Data Explorer (Raw Data)")

    if processed_raw is not None and not processed_raw.empty:
        st.markdown("Explore, filter, and analyze the raw tweet data in detail.")

        # Filtering options
        st.subheader("Filter Data")
        col1, col2, col3 = st.columns([2,2,1]) # Adjust width ratios

        with col1:
            search_term = st.text_input("Search in Tweet Text:", key="data_explorer_search", 
                                       help="Case-insensitive search in tweet content")

        with col2:
            # Convert all user names to strings first to avoid type comparison issues
            user_names = processed_raw['user_name'].astype(str).unique()
            all_users = ['All Users'] + sorted(user_names.tolist())
            selected_user = st.selectbox("Filter by User:", all_users, key="data_explorer_user")

        with col3:
            # Ensure sentiment values are strings for consistent display
            sentiment_values = processed_raw['sentiment'].astype(str).unique()
            sentiment_options = ['All'] + sorted(sentiment_values.tolist())
            selected_sentiment = st.selectbox("Filter by Sentiment:", sentiment_options, key="data_explorer_sentiment")

        # Additional filters in expandable section
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            with col1:
                # Date range filter if available
                if 'date' in processed_raw.columns and processed_raw['date'].notna().any():
                    # Filter out NaN values before calculating min/max
                    valid_dates = processed_raw['date'].dropna()
                    if not valid_dates.empty:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        
                        if pd.notna(min_date) and pd.notna(max_date):
                            date_filter = st.date_input(
                                "Filter by Date Range:",
                                value=(min_date, max_date),
                                min_value=min_date,
                                max_value=max_date,
                                key="explorer_date_range"
                            )
                            # Handle different return types
                            if isinstance(date_filter, tuple) and len(date_filter) == 2:
                                date_start, date_end = date_filter
                            elif isinstance(date_filter, list) and len(date_filter) >= 1:
                                date_start = date_filter[0]
                                date_end = date_filter[-1]
                            else:
                                date_start = date_end = date_filter
                    else:
                        st.info("No valid dates found in the data.")
                
                # Length filter
                if 'text_length' in processed_raw.columns:
                    min_len = int(processed_raw['text_length'].min())
                    max_len = int(processed_raw['text_length'].max())
                    length_range = st.slider(
                        "Tweet Length (characters):",
                        min_value=min_len,
                        max_value=max_len,
                        value=(min_len, max_len),
                        key="explorer_length_range"
                    )
            
            with col2:
                # Word count filter
                if 'word_count' in processed_raw.columns:
                    min_words = int(processed_raw['word_count'].min())
                    max_words = int(processed_raw['word_count'].max())
                    word_range = st.slider(
                        "Word Count:",
                        min_value=min_words,
                        max_value=max_words,
                        value=(min_words, max_words),
                        key="explorer_word_range"
                    )
                
                # Special content filters
                has_hashtag = st.checkbox("Contains hashtags (#)", key="filter_hashtags")
                has_mention = st.checkbox("Contains mentions (@)", key="filter_mentions")
                has_url = st.checkbox("Contains URLs", key="filter_urls")

        # Apply all filters
        filtered_df = processed_raw.copy()
        
        # Basic filters
        if search_term:
            filtered_df = filtered_df[filtered_df['text'].str.contains(search_term, case=False, na=False)]
        
        if selected_user != 'All Users':
            filtered_df = filtered_df[filtered_df['user_name'].astype(str) == selected_user]
        
        if selected_sentiment != 'All':
            filtered_df = filtered_df[filtered_df['sentiment'].astype(str) == selected_sentiment]
        
        # Advanced filters
        if 'date' in filtered_df.columns and 'date_start' in locals() and 'date_end' in locals():
            filtered_df = filtered_df[(filtered_df['date'] >= date_start) & 
                                      (filtered_df['date'] <= date_end)]
        
        if 'text_length' in filtered_df.columns and 'length_range' in locals():
            filtered_df = filtered_df[(filtered_df['text_length'] >= length_range[0]) & 
                                      (filtered_df['text_length'] <= length_range[1])]
        
        if 'word_count' in filtered_df.columns and 'word_range' in locals():
            filtered_df = filtered_df[(filtered_df['word_count'] >= word_range[0]) & 
                                      (filtered_df['word_count'] <= word_range[1])]
        
        # Content type filters
        if has_hashtag:
            filtered_df = filtered_df[filtered_df['text'].str.contains(r'#\w+', na=False)]
        
        if has_mention:
            filtered_df = filtered_df[filtered_df['text'].str.contains(r'@\w+', na=False)]
        
        if has_url:
            filtered_df = filtered_df[filtered_df['text'].str.contains(r'https?://\S+', na=False)]

        # Display filtered data
        count_message = f"Filtered Data ({len(filtered_df)} tweets, {len(filtered_df)/len(processed_raw):.1%} of total)"
        st.subheader(count_message)
        
        # Data insights before displaying the table
        if not filtered_df.empty:
            st.markdown("### Quick Insights")
            insight_cols = st.columns(3)
            
            with insight_cols[0]:
                # Sentiment distribution in filtered data
                if 'sentiment' in filtered_df.columns:
                    sent_counts = filtered_df['sentiment'].value_counts()
                    fig = px.pie(
                        values=sent_counts.values,
                        names=sent_counts.index,
                        title="Sentiment Distribution",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Blues
                    )
                    fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
            
            with insight_cols[1]:
                # Length distribution
                if 'text_length' in filtered_df.columns:
                    fig = px.histogram(
                        filtered_df, x='text_length',
                        nbins=20,
                        title="Length Distribution",
                        color_discrete_sequence=['#1DA1F2']
                    )
                    fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0),
                                     xaxis_title="Character Count")
                    st.plotly_chart(fig, use_container_width=True)
            
            with insight_cols[2]:
                # Word count or time distribution
                if 'word_count' in filtered_df.columns:
                    fig = px.histogram(
                        filtered_df, x='word_count',
                        nbins=15,
                        title="Word Count Distribution",
                        color_discrete_sequence=['#17bf63']
                    )
                    fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0),
                                     xaxis_title="Word Count")
                    st.plotly_chart(fig, use_container_width=True)

            # Text analysis section
            with st.expander("Text Analysis of Filtered Data", expanded=False):
                # Extract most common words
                if 'text' in filtered_df.columns:
                    # Simple word frequency analysis
                    all_words = ' '.join(filtered_df['text']).lower()
                    # Remove URLs, mentions, punctuation
                    all_words = re.sub(r'https?://\S+|@\w+|[^\w\s]', '', all_words)
                    # Split into words and count
                    word_list = all_words.split()
                    # Filter out common stop words
                    stop_words = {'the', 'and', 'is', 'in', 'to', 'a', 'of', 'for', 'on', 'with', 'as', 'at', 'by', 'an', 'be', 'this', 'that', 'it', 'or'}
                    word_list = [word for word in word_list if word not in stop_words and len(word) > 2]
                    
                    word_counts = Counter(word_list).most_common(20)
                    if word_counts:
                        word_df = pd.DataFrame(word_counts, columns=['word', 'count'])
                        fig = px.bar(
                            word_df.sort_values('count', ascending=True),
                            y='word', x='count',
                            orientation='h',
                            title="Most Common Words (excluding stop words)",
                            color='count',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Extract hashtags specifically
                    all_hashtags = []
                    for text in filtered_df['text']:
                        tags = re.findall(r'#(\w+)', text.lower())
                        all_hashtags.extend(tags)
                    
                    if all_hashtags:
                        hashtag_counts = Counter(all_hashtags).most_common(15)
                        hashtag_df = pd.DataFrame(hashtag_counts, columns=['hashtag', 'count'])
                        fig = px.bar(
                            hashtag_df.sort_values('count', ascending=False),
                            x='hashtag', y='count',
                            title="Most Common Hashtags",
                            color='count',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No hashtags found in the filtered data.")
        
        # Select and reorder columns for better display
        display_columns = ['timestamp_str', 'user_name', 'text', 'sentiment', 'text_length', 'word_count']
        # Add others if they exist and are useful
        for col in ['tweet_id', 'target']:
            if col in filtered_df.columns:
                display_columns.insert(1, col) # Insert after timestamp

        # Ensure columns exist before selecting
        display_columns = [col for col in display_columns if col in filtered_df.columns]
        
        # Show dataframe with column selection option
        col_select = st.multiselect(
            "Select columns to display:",
            options=display_columns,
            default=display_columns[:5]  # Default to first 5 columns
        )
        
        if col_select:
            st.dataframe(filtered_df[col_select], use_container_width=True)
        else:
            st.dataframe(filtered_df[display_columns], use_container_width=True)

        # Export options
        st.subheader("Export Options")
        export_cols = st.columns(3)
        
        with export_cols[0]:
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(filtered_df)
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=f'filtered_tweets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with export_cols[1]:
            # Excel export option
            @st.cache_data
            def to_excel(df):
                import io
                from datetime import datetime
                
                # Create a copy to avoid modifying the original
                df_excel = df.copy()
                
                # More thorough handling of datetime columns
                for col in df_excel.columns:
                    # Check if column contains datetime objects
                    if pd.api.types.is_datetime64_any_dtype(df_excel[col]):
                        # Convert timezone-aware datetimes to naive
                        try:
                            # First check if it has timezone info
                            if hasattr(df_excel[col].dtype, 'tz') and df_excel[col].dtype.tz is not None:
                                # Remove timezone info
                                df_excel[col] = df_excel[col].dt.tz_localize(None)
                        except Exception as e:
                            # If conversion fails, convert to string as fallback
                            st.warning(f"Converting datetime column '{col}' to string due to: {e}")
                            df_excel[col] = df_excel[col].astype(str)
                
                # Convert any remaining problematic values to strings
                for col in df_excel.select_dtypes(include=['object']).columns:
                    # Check if any values in this column are timezone-aware datetime objects
                    has_tz_aware = False
                    for val in df_excel[col].dropna().unique():
                        if isinstance(val, datetime) and val.tzinfo is not None:
                            has_tz_aware = True
                            break
                    
                    if has_tz_aware:
                        df_excel[col] = df_excel[col].astype(str)
                
                # Create Excel file
                buffer = io.BytesIO()
                try:
                    with pd.ExcelWriter(buffer) as writer:
                        df_excel.to_excel(writer, index=False, sheet_name="Filtered_Tweets")
                    return buffer.getvalue()
                except ValueError as e:
                    # If we still get timezone errors, convert all to string as last resort
                    if "timezone" in str(e).lower():
                        st.warning("Converting all date columns to string format for Excel export")
                        for col in df_excel.columns:
                            if pd.api.types.is_datetime64_any_dtype(df_excel[col]):
                                df_excel[col] = df_excel[col].astype(str)
                        
                        # Try again with string dates
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer) as writer:
                            df_excel.to_excel(writer, index=False, sheet_name="Filtered_Tweets")
                        return buffer.getvalue()
                    else:
                        raise
            
            excel_data = to_excel(filtered_df)
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name=f'filtered_tweets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                mime='application/vnd.ms-excel'
            )
        
        with export_cols[2]:
            # JSON export option
            @st.cache_data
            def to_json(df):
                return df.to_json(orient='records', date_format='iso').encode('utf-8')
            
            json_data = to_json(filtered_df)
            st.download_button(
                label="Download as JSON",
                data=json_data,
                file_name=f'filtered_tweets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json'
            )

    else:
        st.error("Data explorer requires the raw tweets dataset (sampled_tweets.csv).")


# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="small-text" style="text-align: center;">
    Simplified Tweet Analysis Dashboard | Based on Pig & Raw Data Analysis
</div>
""", unsafe_allow_html=True)