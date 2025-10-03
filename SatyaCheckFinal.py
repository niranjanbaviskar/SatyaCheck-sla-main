import streamlit as st
st.set_page_config(page_title="Satya Check", page_icon="🕵🏻", layout="wide")

# Load Google Font
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ------------------------------
# Other Imports and Setup
# ------------------------------
import re, io, csv, asyncio
import requests
from bs4 import BeautifulSoup
import spacy
import praw
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# API Configurations (replace with your own keys)
REDDIT_CLIENT_ID = "zUm08w-ffHWzi9jim_eq7w"
REDDIT_CLIENT_SECRET = "zjvo8yfii4iEHRLBXbC2-55gQf3LAQ"
REDDIT_USER_AGENT = "script:news_verifier_app:1.0 (by /u/Shlong_up )"
YOUTUBE_API_KEY = "AIzaSyCUmfjixPPCcpovy45VdHXaEqAelFaJgxY"

# Initialize API clients
reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Set up a TTL cache (max 100 items, 5 minutes TTL)
cache = TTLCache(maxsize=100, ttl=300)

# ------------------------------
# UI & Styling Functions
# ------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Global & Font */
    body, button, input, textarea { font-family: 'Montserrat', sans-serif; }
    body { background-color: #F0F2F5; color: #333; margin: 0; padding: 0; }
    .main .block-container { padding: 2rem 1rem; max-width: 1200px; margin: auto; }
    .stApp { background-color: #F0F2F5; }
    /* Header */
    .app-title { font-weight: 800; font-size: 3rem; background: linear-gradient(135deg, #5B51EB, #FF6A95);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;
                 margin-bottom: 0.5rem; text-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    .app-subtitle { text-align: center; color: #555; font-size: 1.2rem; margin-bottom: 2rem; }
    /* Sidebar */
    .css-1cypcdb { background-color: #fff !important; border-radius: 10px; padding: 1.5rem; 
                   box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .sidebar-description { font-size: 0.9rem; color: #555; margin-bottom: 1rem; }
    /* Cards */
    .result-section { background-color: #fff; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
                      box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .post-card { background-color: #fff; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
                 box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-left: 4px solid #5B51EB;
                 transition: transform 0.3s, box-shadow 0.3s; }
    .post-card:hover { transform: translateY(-3px); box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
    /* Buttons & Inputs */
    .stButton > button { background: linear-gradient(135deg, #5B51EB, #7E76F0); color: #fff;
                          border: none; border-radius: 50px; padding: 0.75rem 2rem; font-weight: 700;
                          font-size: 1rem; transition: transform 0.3s, box-shadow 0.3s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { border-radius: 8px; border: 1px solid #ccc;
                                                                           padding: 1rem; font-size: 1rem;
                                                                           box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                                                           transition: border-color 0.3s, box-shadow 0.3s; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: #5B51EB;
                                                                                      box-shadow: 0 0 0 3px rgba(91,81,235,0.15); }
    /* Credibility Badge */
    .credibility-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 50px;
                         font-weight: 700; text-align: center; margin-bottom: 1rem; }
    .credibility-badge.reliable { background: linear-gradient(135deg, #34D399, #10B981); color: #fff; }
    .credibility-badge.uncertain { background: linear-gradient(135deg, #F59E0B, #D97706); color: #fff; }
    .credibility-badge.unreliable { background: linear-gradient(135deg, #EF4444, #DC2626); color: #fff; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Profession-Specific Fake News Detection Functions
# ------------------------------
def advanced_fake_news_model(article_text: str) -> float:
    return 50.0  # Stub value

def predict_fake_news_by_profession(article_text: str, credibility: dict, sensationalism_score: int, profession: str) -> str:
    cred = credibility.get("credibility_score", 0)
    transformer_score = advanced_fake_news_model(article_text)
    combined_score = (cred + (100 - transformer_score)) / 2
    thresholds = {
        "Traders/Investors": (65, 45, 4),
        "Journalists": (55, 45, 2),
        "Political Analysts": (50, 40, 3),
        "Government Officials & Policy Makers": (60, 50, 3),
        "Researchers/Academics": (55, 45, 3),
        "Media and Broadcasting Organizations": (60, 50, 3),
        "Fact-Checking Organizations": (65, 50, 2),
        "Public Relations & Communications Professionals": (60, 50, 3),
        "Marketing & Advertising Professionals": (55, 45, 3),
        "Legal Professionals": (70, 55, 3),
        "Corporate Communications Teams": (60, 50, 3),
        "Social Media Analysts": (55, 45, 3),
        "Risk Management Specialists": (65, 50, 4),
        "General": (60, 50, 3)
    }
    t_true, t_fake, s_limit = thresholds.get(profession, thresholds["General"])
    if combined_score >= t_true and sensationalism_score < s_limit:
        return "Likely True"
    elif combined_score < t_fake and sensationalism_score >= s_limit:
        return "Likely Fake"
    else:
        return "Uncertain"

def get_profession_commentary(profession: str, credibility: dict, sensationalism_score: int, keywords: dict) -> str:
    commentary_dict = {
        "Traders/Investors": f"Market Analysis: Credibility Score is {credibility.get('credibility_score', 0)}. Verify market trends with financial data.",
        "Journalists": f"Journalistic Review: Credibility Score is {credibility.get('credibility_score', 0)}. Extracted keywords are hidden by default.",
        "Political Analysts": f"Political Analysis: Credibility Score is {credibility.get('credibility_score', 0)}. Review political context and bias carefully.",
        "Government Officials & Policy Makers": f"Policy Review: Credibility Score is {credibility.get('credibility_score', 0)}. Consider governance implications.",
        "Researchers/Academics": f"Academic Insight: Credibility Score is {credibility.get('credibility_score', 0)}. Evaluate methodological rigor in reporting.",
        "Media and Broadcasting Organizations": f"Media Assessment: Credibility Score is {credibility.get('credibility_score', 0)}. Ensure accuracy before broadcasting.",
        "Fact-Checking Organizations": f"Fact-Checking: Credibility Score is {credibility.get('credibility_score', 0)}. Additional investigation is recommended.",
        "Public Relations & Communications Professionals": f"PR Impact: Credibility Score is {credibility.get('credibility_score', 0)}. Monitor for potential reputational risk.",
        "Marketing & Advertising Professionals": f"Brand Analysis: Credibility Score is {credibility.get('credibility_score', 0)}. Verify news to safeguard brand image.",
        "Legal Professionals": f"Legal Risk: Credibility Score is {credibility.get('credibility_score', 0)}. Assess potential liability risks.",
        "Corporate Communications Teams": f"Corporate Analysis: Credibility Score is {credibility.get('credibility_score', 0)}. Ensure messages align with corporate integrity.",
        "Social Media Analysts": f"Social Media Trends: Credibility Score is {credibility.get('credibility_score', 0)}. Analyze the spread and influence.",
        "Risk Management Specialists": f"Risk Assessment: Credibility Score is {credibility.get('credibility_score', 0)}. Evaluate potential impacts on market or public perception.",
        "General": f"General Analysis: Credibility Score is {credibility.get('credibility_score', 0)}. Review overall trustworthiness."
    }
    return commentary_dict.get(profession, commentary_dict["General"])

# ------------------------------
# Caching & Parallel API Calls
# ------------------------------
def get_cached(key, fetch_func):
    if key in cache:
        return cache[key]
    data = fetch_func()
    cache[key] = data
    return data

def fetch_reddit_posts(keywords, max_posts, time_threshold):
    results = []
    query = " OR ".join(keywords.keys())
    for submission in reddit.subreddit('all').search(query, limit=max_posts, sort='new'):
        post_time = datetime.fromtimestamp(submission.created_utc)
        if post_time > time_threshold:
            full_text = f"{submission.title} {submission.selftext}"
            match_pct, matched_kw = calculate_keyword_match(full_text, keywords)
            if match_pct > 0:
                results.append({
                    'platform': 'Reddit',
                    'title': submission.title,
                    'text': submission.selftext,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'url': submission.url,
                    'timestamp': post_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'keyword_match_percentage': match_pct,
                    'matched_keywords': matched_kw
                })
    return results

def fetch_youtube_posts(keywords, max_posts, time_threshold):
    results = []
    query = " OR ".join(keywords.keys())
    search_response = youtube.search().list(
        q=query, part="id,snippet", maxResults=max_posts, type="video", order="date"
    ).execute()
    for item in search_response.get("items", []):
        video_id = item["id"]["videoId"]
        video_response = youtube.videos().list(part="statistics,snippet", id=video_id).execute()
        if video_response.get("items"):
            video = video_response["items"][0]
            published_time = datetime.strptime(video['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
            if published_time > time_threshold:
                full_text = f"{video['snippet']['title']} {video['snippet']['description']}"
                match_pct, matched_kw = calculate_keyword_match(full_text, keywords)
                if match_pct > 0:
                    results.append({
                        'platform': 'YouTube',
                        'title': video['snippet']['title'],
                        'description': video['snippet']['description'],
                        'views': int(video['statistics'].get('viewCount', 0)),
                        'likes': int(video['statistics'].get('likeCount', 0)),
                        'url': f"https://youtu.be/{video_id}",
                        'timestamp': published_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'keyword_match_percentage': match_pct,
                        'matched_keywords': matched_kw
                    })
    return results

def fetch_comprehensive_posts(keywords: dict, platforms: list, max_posts: int = 100, hours_back: int = 120):
    time_threshold = datetime.utcnow() - timedelta(hours=hours_back)
    results = []
    matched_keywords_data = {}
    platform_counts = {"Reddit": 0, "YouTube": 0}
    with ThreadPoolExecutor() as executor:
        futures = {}
        if "Reddit" in platforms:
            futures["Reddit"] = executor.submit(get_cached, f"reddit_{str(keywords)}",
                                                  lambda: fetch_reddit_posts(keywords, max_posts, time_threshold))
        if "YouTube" in platforms:
            futures["YouTube"] = executor.submit(get_cached, f"youtube_{str(keywords)}",
                                                   lambda: fetch_youtube_posts(keywords, max_posts, time_threshold))
        for key, future in futures.items():
            posts = future.result()
            for post in posts:
                for kw, wt in post['matched_keywords'].items():
                    matched_keywords_data[kw] = matched_keywords_data.get(kw, 0) + 1
            results.extend(posts)
            platform_counts[key] = len(posts)
    results = sorted(results, key=lambda p: p['keyword_match_percentage'], reverse=True)
    return results, matched_keywords_data, platform_counts

# ------------------------------
# Keyword Extraction & CSV Generation
# ------------------------------
def extract_comprehensive_keywords(article_text: str) -> dict:
    doc = nlp(article_text)
    keywords = {}
    for chunk in doc.noun_chunks:
        keyword = chunk.text.strip().lower()
        if 1 <= len(keyword.split()) <= 4:
            keywords[keyword] = keywords.get(keyword, 0) + 2
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
            keyword = ent.text.strip().lower()
            keywords[keyword] = keywords.get(keyword, 0) + 3
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and len(token.text) > 3:
            keyword = token.lemma_.lower()
            keywords[keyword] = keywords.get(keyword, 0) + 1
    return dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10])

def generate_full_csv(input_text, keywords, matched_posts):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Input Text", "Extracted Keywords", "Matched Site URL", "Matched Site Title", "Platform", "Keyword Match Percentage"])
    kw_str = "; ".join([f"{kw}: {wt}" for kw, wt in keywords.items()])
    if matched_posts:
        for post in matched_posts:
            writer.writerow([input_text, kw_str, post.get("url", ""), post.get("title", ""), post.get("platform", ""), post.get("keyword_match_percentage", "")])
    else:
        writer.writerow([input_text, kw_str, "", "", "", ""])
    return output.getvalue()

# ------------------------------
# Text Similarity & Heuristics
# ------------------------------
def calculate_keyword_match(post_text: str, original_keywords: dict) -> tuple:
    post_text = post_text.lower()
    total_weight = sum(original_keywords.values())
    matched_weight = 0
    matched_keywords = {}
    for keyword, weight in original_keywords.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, post_text):
            matched_weight += weight
            matched_keywords[keyword] = weight
    match_percentage = (matched_weight / total_weight) * 100 if total_weight else 0
    return round(match_percentage, 2), matched_keywords

# ------------------------------
# Fake News Detection Heuristics
# ------------------------------
def assess_news_credibility(posts: list) -> dict:
    if not posts:
        return {'credibility_score': 0, 'max_keyword_match': 0, 'avg_keyword_match': 0,
                'credibility_status': 'No Matching Posts', 'status_description': 'Insufficient data to verify the news.',
                'color': 'gray', 'class': 'uncertain'}
    match_percents = [post['keyword_match_percentage'] for post in posts]
    max_match = max(match_percents)
    avg_match = sum(match_percents) / len(match_percents)
    if max_match >= 60:
        return {'credibility_score': max_match, 'max_keyword_match': max_match, 'avg_keyword_match': round(avg_match, 2),
                'credibility_status': 'Reliable News', 'status_description': 'The news is well-supported by social media.',
                'color': '#10B981', 'class': 'reliable'}
    elif max_match >= 40:
        return {'credibility_score': max_match, 'max_keyword_match': max_match, 'avg_keyword_match': round(avg_match, 2),
                'credibility_status': 'Uncertain News', 'status_description': 'Limited corroboration. Further investigation recommended.',
                'color': '#F59E0B', 'class': 'uncertain'}
    else:
        return {'credibility_score': max_match, 'max_keyword_match': max_match, 'avg_keyword_match': round(avg_match, 2),
                'credibility_status': 'Likely Untrue', 'status_description': 'Minimal social media support detected.',
                'color': '#EF4444', 'class': 'unreliable'}

def override_credibility_by_post_count(posts: list) -> dict:
    if len(posts) < 3:
        return {'credibility_score': 0, 'max_keyword_match': 0, 'avg_keyword_match': 0,
                'credibility_status': 'Not Reliable News', 'status_description': 'Less than 3 supporting posts found.',
                'color': '#EF4444', 'class': 'unreliable'}
    return None

def calculate_sensationalism_score(text: str) -> int:
    exclamations = text.count("!")
    words = text.split()
    all_caps = sum(1 for word in words if word.isupper() and len(word) > 1)
    return exclamations + all_caps

# ------------------------------
# UI Display Functions
# ------------------------------
def display_big_credibility_status(credibility: dict):
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; margin: 2rem 0;">
        <div class="credibility-badge {credibility['class']}" style="font-size: 1.8rem; padding: 1rem 2rem;">
            {credibility['credibility_status']}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_statistics_visualizations(posts, keywords_data, platform_counts, chart_type="Bar Chart"):
    if not posts:
        st.warning("No data available for visualization.")
        return
    st.markdown("""<div class="result-section"><h3>📊 Statistical Analysis</h3></div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        platform_df = pd.DataFrame({
            'Platform': ['Reddit', 'YouTube'],
            'Posts': [platform_counts.get('Reddit', 0), platform_counts.get('YouTube', 0)]
        })
        if chart_type == "Pie Chart":
            fig = px.pie(platform_df, values='Posts', names='Platform', title='Platform Distribution', 
                         color='Platform', color_discrete_map={'Reddit': '#FF4500', 'YouTube': '#FF0000'}, hole=0.4)
            fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(platform_df, x='Platform', y='Posts', title='Platform Distribution', 
                         color='Platform', color_discrete_map={'Reddit': '#FF4500', 'YouTube': '#FF0000'}, text='Posts')
            fig.update_layout(xaxis_title=None, yaxis_title="Number of Posts", showlegend=False,
                              margin=dict(t=40, b=20, l=20, r=20), height=300)
            fig.update_traces(textposition='auto')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        match_ranges = ["0-25%", "26-50%", "51-75%", "76-100%"]
        match_counts = [len([p for p in posts if p['keyword_match_percentage'] < 25]),
                        len([p for p in posts if 25 <= p['keyword_match_percentage'] < 50]),
                        len([p for p in posts if 50 <= p['keyword_match_percentage'] < 75]),
                        len([p for p in posts if p['keyword_match_percentage'] >= 75])]
        match_df = pd.DataFrame({'Match Range': match_ranges, 'Count': match_counts})
        if chart_type == "Pie Chart":
            fig = px.pie(match_df, values='Count', names='Match Range', title='Keyword Match Distribution',
                         color='Match Range', color_discrete_map={'0-25%':'#EF4444','26-50%':'#F59E0B',
                                                                   '51-75%':'#3B82F6','76-100%':'#10B981'}, hole=0.4)
            fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(match_df, x='Match Range', y='Count', title='Keyword Match Distribution',
                         color='Match Range', color_discrete_map={'0-25%':'#EF4444','26-50%':'#F59E0B',
                                                                   '51-75%':'#3B82F6','76-100%':'#10B981'}, text='Count')
            fig.update_layout(xaxis_title=None, yaxis_title="Number of Posts", showlegend=False,
                              margin=dict(t=40, b=20, l=20, r=20), height=300)
            fig.update_traces(textposition='auto')
            st.plotly_chart(fig, use_container_width=True)
    with col3:
        dates = [datetime.strptime(p['timestamp'], "%Y-%m-%d %H:%M:%S").date() for p in posts]
        date_counts = {}
        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
        time_df = pd.DataFrame({'Date': list(date_counts.keys()), 'Posts': list(date_counts.values())})
        time_df['Date'] = pd.to_datetime(time_df['Date'])
        time_df = time_df.sort_values('Date')
        if chart_type in ["Line Chart", "Heat Map"]:
            fig = px.line(time_df, x='Date', y='Posts', title='Posts Over Time', markers=True)
            fig.update_layout(xaxis_title=None, yaxis_title="Number of Posts", margin=dict(t=40, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(time_df, x='Date', y='Posts', title='Posts Over Time', text='Posts')
            fig.update_layout(xaxis_title=None, yaxis_title="Number of Posts", margin=dict(t=40, b=20, l=20, r=20), height=300)
            fig.update_traces(textposition='auto')
            st.plotly_chart(fig, use_container_width=True)

def display_analyzed_posts(posts, keywords, show_details=True):
    if not posts:
        st.warning("No matching posts found.")
        return
    st.markdown("""<div class="result-section"><h3>🔍 Analyzed Social Media Posts</h3></div>""", unsafe_allow_html=True)
    st.markdown(f"<p>Showing top {min(10, len(posts))} most relevant posts:</p>", unsafe_allow_html=True)
    for post in posts[:10]:
        platform_class = "reddit" if post['platform'] == "Reddit" else "youtube"
        platform_icon = "🔴" if post['platform'] == "Reddit" else "▶️"
        st.markdown(f"""
        <div class="post-card {platform_class}">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 700;">{platform_icon} {post['platform']}</span>
                <span style="background: linear-gradient(135deg, {get_gradient_colors(post['keyword_match_percentage'])});
                      color: #fff; padding: 0.25rem 0.75rem; border-radius: 50px; font-weight: 700;">
                    Match: {post['keyword_match_percentage']}%
                </span>
            </div>
            <h4 style="margin: 0.5rem 0;">{post['title']}</h4>
            <div style="display: flex; gap: 1rem; margin-bottom: 0.5rem;">
                <span>📅 {post['timestamp']}</span>
                {format_platform_stats(post)}
            </div>
            {display_matched_keywords(post, keywords) if show_details else ''}
            <div style="margin-top: 1rem;">
                <a href="{post['url']}" target="_blank" style="color: #5B51EB; text-decoration: none; font-weight: 600;">
                    View Original Post →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_gradient_colors(match_percentage):
    if match_percentage >= 75:
        return "#10B981 0%, #059669 100%"
    elif match_percentage >= 50:
        return "#3B82F6 0%, #2563EB 100%"
    elif match_percentage >= 25:
        return "#F59E0B 0%, #D97706 100%"
    else:
        return "#EF4444 0%, #DC2626 100%"

def format_platform_stats(post):
    if post['platform'] == "Reddit":
        return f"""<span>👍 {post.get('score', 0)}</span>
                   <span>💬 {post.get('num_comments', 0)}</span>"""
    else:
        return f"""<span>👁️ {format_number(post.get('views', 0))}</span>
                   <span>👍 {format_number(post.get('likes', 0))}</span>"""

def format_number(num):
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

def display_matched_keywords(post, original_keywords):
    if not post.get('matched_keywords'):
        return ""
    matched_html = "<div style='margin-top: 0.75rem;'><strong>Matched Keywords:</strong> "
    for keyword, weight in post['matched_keywords'].items():
        importance = min(100, int((weight / max(original_keywords.values())) * 100))
        opacity = 0.5 + (importance / 200)
        matched_html += f"""<span class="keyword-badge" style="opacity: {opacity};">{keyword}</span>"""
    matched_html += "</div>"
    return matched_html

# ------------------------------
# Main App & Firebase Logging Integration
# ------------------------------
def main():
    apply_custom_css()
    professions = [
        "General", "Traders/Investors", "Journalists", "Political Analysts",
        "Government Officials & Policy Makers", "Researchers/Academics",
        "Media and Broadcasting Organizations", "Fact-Checking Organizations",
        "Public Relations & Communications Professionals", "Marketing & Advertising Professionals",
        "Legal Professionals", "Corporate Communications Teams", "Social Media Analysts",
        "Risk Management Specialists"
    ]
    st.sidebar.markdown("<div style='text-align: center;'><h2>⚙️ Analysis Settings</h2></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-description'>Adjust parameters for data retrieval and analysis:</div>", unsafe_allow_html=True)
    hours_back = st.sidebar.slider("Search posts from last (hours)", min_value=12, max_value=240, value=120, step=12)
    keyword_sensitivity = st.sidebar.slider("Keyword match threshold (%)", min_value=10, max_value=50, value=20, step=5)
    platforms = st.sidebar.multiselect("Select platforms", ["Reddit", "YouTube"], default=["Reddit", "YouTube"])
    chart_type = st.sidebar.selectbox("Visualization style", ["Bar Chart", "Pie Chart", "Line Chart", "Heat Map"], index=0)
    show_detailed_stats = st.sidebar.checkbox("Show detailed statistics", value=True)
    profession = st.sidebar.selectbox("Select your profession", options=professions)
    
    settings = {
        'hours_back': hours_back,
        'keyword_sensitivity': keyword_sensitivity,
        'platforms': platforms,
        'chart_type': chart_type,
        'show_detailed_stats': show_detailed_stats,
        'profession': profession
    }
    
    st.markdown("<div class='app-title'>🔍SatyaCheck</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Verify news credibility through social media cross-referencing</div>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Text Analysis", "About"])
    
    with tabs[0]:
        st.markdown("### 📝 Paste news article text to verify")
        article_text = st.text_area("News article text", height=300, placeholder="Paste the complete news article text here...")
        analyze_text_button = st.button("Analyze Text")
        
        if analyze_text_button and article_text:
            with st.spinner("Extracting keywords..."):
                keywords = extract_comprehensive_keywords(article_text)
                st.session_state["extracted_keywords"] = keywords
            with st.spinner("Fetching social media posts..."):
                posts, matched_keywords_data, platform_counts = fetch_comprehensive_posts(
                    keywords, settings['platforms'], max_posts=100, hours_back=settings['hours_back']
                )
                filtered_posts = [p for p in posts if p['keyword_match_percentage'] >= settings['keyword_sensitivity']]
                st.success(f"Found {len(filtered_posts)} relevant posts (matched ≥{settings['keyword_sensitivity']}% keywords)")
                credibility = assess_news_credibility(filtered_posts)
                override = override_credibility_by_post_count(filtered_posts)
                if override:
                    credibility = override
                display_big_credibility_status(credibility)
                create_statistics_visualizations(filtered_posts, matched_keywords_data, platform_counts, chart_type=settings['chart_type'])
                display_analyzed_posts(filtered_posts, keywords, settings['show_detailed_stats'])
            sensationalism_score = calculate_sensationalism_score(article_text)
            fake_news_prediction = predict_fake_news_by_profession(article_text, credibility, sensationalism_score, settings['profession'])
            st.markdown(f"### Final Fake News Detection Result: **{fake_news_prediction}**")
            st.markdown(f"*Credibility Score: {credibility.get('credibility_score', 0)} | Sensationalism Score: {sensationalism_score}*")
            prof_comment = get_profession_commentary(settings['profession'], credibility, sensationalism_score, keywords)
            st.markdown(f"#### {settings['profession']} Analysis: {prof_comment}")
            
            full_csv = generate_full_csv(article_text, keywords, filtered_posts)
            st.download_button("Download Full Results as CSV", full_csv, "full_results.csv", "text/csv")
            feedback = st.text_input("Have suggestions or feedback about the prediction? Tell us below:")
            if st.button("Submit Feedback"):
                if feedback:
                    st.session_state.feedback = st.session_state.get("feedback", [])
                    st.session_state.feedback.append({"text": article_text, "feedback": feedback, "timestamp": datetime.utcnow().isoformat()})
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Please enter your feedback.")
    
    with tabs[1]:
        st.markdown("""
        <div class="result-section">
            <h3>ℹ️ About SatyaCheck</h3>
            <p>SatyaCheck is a powerful tool designed to verify news credibility by cross-referencing social media. The name "Satya" means "truth" in Sanskrit, reflecting our commitment to reliable information.</p>
            <h4>How It Works</h4>
            <ol>
                <li>Paste the news article text.</li>
                <li>The system extracts key information and identifies important keywords (hidden by default).</li>
                <li>It concurrently searches social media platforms (e.g. Reddit & YouTube) using caching for efficiency.</li>
                <li>Heuristics—including sensationalism scoring and a (stub) transformer-based model—calculate a credibility score.</li>
                <li>Fake news detection is customized based on your selected profession.</li>
                                <li>You receive detailed data analysis along with options to download reports and provide feedback.</li>
            </ol>
            <h4>Who Can Benefit?</h4>
            <ul>
                <li>Traders/Investors</li>
                <li>Journalists</li>
                <li>Political Analysts</li>
                <li>Government Officials & Policy Makers</li>
                <li>Researchers/Academics</li>
                <li>Media & Broadcasting Organizations</li>
                <li>Fact-Checking Organizations</li>
                <li>Public Relations & Communications Professionals</li>
                <li>Marketing & Advertising Professionals</li>
                <li>Legal Professionals</li>
                <li>Corporate Communications Teams</li>
                <li>Social Media Analysts</li>
                <li>Risk Management Specialists</li>
            </ul>
            <p><strong>Note:</strong> This tool is part of an evolving media literacy toolkit. Always verify information with multiple sources.
                    We provide you a spreadsheet file with all of your searched data so you can build a dataset and improve our open-source model.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()