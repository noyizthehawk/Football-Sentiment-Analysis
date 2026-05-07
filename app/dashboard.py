from supabase import create_client
import pandas as pd
import os
import re
import json
from collections import Counter
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone


load_dotenv()

st.set_page_config(
    page_title="NewsPulse",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("NewsPulse")
st.markdown("Sentiment of All News Articles")



#connect to supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase=create_client(url,key)

def get_article_count(filter_team=None):
    response = supabase.table("sentiment").select("teams_mentioned").execute()
    data = response.data

    if not filter_team:
        return len(data)

    count = 0
    for row in data:
        teams = row.get("teams_mentioned") or []
        if filter_team in teams:
            count += 1

    return count

def get_recent_article_delta(hours=12, filter_team=None):
    now = datetime.now(timezone.utc)
    past = now - timedelta(hours=hours)

    response = supabase.table("sentiment") \
        .select("analyzed_at, teams_mentioned") \
        .gte("analyzed_at", past.isoformat()) \
        .execute()

    data = response.data

    if filter_team:
        data = [
            row for row in data
            if filter_team in (row.get("teams_mentioned") or [])
        ]

    recent_count = len(data)

    # total articles
    total = get_article_count(filter_team)

    return recent_count, total

# getting most mentioned team
def most_covered(number=1):
    mentioned_teams = (
        supabase.table("sentiment")
        .select("teams_mentioned")
        .eq("is_football", True)
        .execute()
    )
    data = mentioned_teams.data
    all_teams = []
    for row in data:
        teams = row.get("teams_mentioned") or []
        all_teams.extend(teams)

    team_counts = Counter(all_teams)
    # return just team names
    return team_counts.most_common(number)


# sentiment analysis (reusable)
def compute_sentiment_overall(filter_team=None):
    response = supabase.table("sentiment").select("sentiment, teams_mentioned").execute()
    data = response.data

    sentiments = []

    for row in data:
        teams = row.get("teams_mentioned") or []

        # if a team is specified, filter
        if filter_team:
            if filter_team not in teams:
                continue

        sentiments.append(row.get("sentiment"))

    counts = Counter(sentiments)
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    neutral = counts.get("neutral", 0)

    total = positive + negative + neutral

    if total == 0:
        return "No Data"

    pos_pct = positive / total
    neg_pct = negative / total

    if pos_pct > 0.7:
        return "Overwhelmingly Positive"
    elif pos_pct > 0.55:
        return "Mostly Positive"
    elif pos_pct > 0.45:
        return "Slightly Positive"
    elif neg_pct > 0.7:
        return "Overwhelmingly Negative"
    elif neg_pct > 0.55:
        return "Mostly Negative"
    elif neg_pct > 0.45:
        return "Slightly Negative"
    else:
        return "Neutral / Mixed"

def get_sentiment_counts(filter_team=None):
    response = supabase.table("sentiment").select("sentiment, teams_mentioned").execute()
    data = response.data

    sentiments = []
    for row in data:
        teams = row.get("teams_mentioned") or []
        if filter_team:
            if filter_team not in teams:
                continue
        sentiments.append(row.get("sentiment"))
    counts = Counter(sentiments)
    return {
        "positive": counts.get("positive", 0),
        "neutral": counts.get("neutral", 0),
        "negative": counts.get("negative", 0),
    }
# default (all articles)
overall = compute_sentiment_overall()

# ================= TEAM FILTER =================
team_list = [team for team, _ in most_covered(20)]
selected_team = st.selectbox("Filter by Team", ["All"] + team_list)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Total Articles")
    recent_count, total_articles = get_recent_article_delta(
        12,
        selected_team if selected_team != "All" else None
    )

    st.metric(
        label="Total Articles",
        value=get_article_count(selected_team if selected_team != "All" else None),
        label_visibility="collapsed",
        delta=f"+{recent_count} (last 12h)"
    )

with col2:
    st.markdown("### ⚽ Most Covered Team")
    st.metric(
        label="Most Covered Team",
        value=selected_team if selected_team != "All" else most_covered(1)[0][0],
        label_visibility="collapsed"
    )

with col3:
    st.markdown("### 🧠 Overall Sentiment")
    st.metric(
        label="Overall Sentiment",
        value=compute_sentiment_overall(selected_team if selected_team != "All" else None),
        label_visibility="collapsed"
    )
#================pie chart ==============
counts = get_sentiment_counts(selected_team if selected_team != "All" else None)
labels = ["Positive", "Neutral", "Negative"]
sizes = [
    counts["positive"],
    counts["neutral"],
    counts["negative"]
]
pie_fig, pie_ax = plt.subplots()
pie_fig.patch.set_facecolor("#0e1117")
pie_ax.set_facecolor("#0e1117")
colors = ["#2ecc71", "#95a5a6", "#e74c3c"]  # green, gray, red

pie_ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    wedgeprops={"width": 0.4},
    textprops={"color": "white"}
)
pie_ax.axis("equal")

#top 10 teams bar==========================
top_teams = most_covered(10)

teams = [team for team, count in top_teams]
team_counts = [count for team, count in top_teams]

bar_fig, bar_ax = plt.subplots()
colors = ["#e74c3c" if team == selected_team else "#4C78A8" for team in teams]
bar_ax.barh(teams, team_counts, color=colors)
bar_ax.set_xlabel("Number of Articles")
bar_ax.set_title("Top 10 Most Covered Teams")
bar_ax.invert_yaxis()

left, right = st.columns(2)
with left:
    st.subheader("📊 Sentiment Breakdown")
    st.pyplot(pie_fig)
with right:
    st.subheader("🏟️ Top 10 Teams")
    st.pyplot(bar_fig)
