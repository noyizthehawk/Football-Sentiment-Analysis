from supabase import create_client
import anthropic
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

@st.cache_data(ttl=18000)
def fetch_sentiment_data():
    return supabase.table("sentiment").select("*").execute().data

def get_article_count(filter_team=None):
    data = fetch_sentiment_data()

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

    data = fetch_sentiment_data()
    recent = []

    for row in data:
        ts = row.get("analyzed_at")
        if not ts:
            continue

        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except:
            continue

        # FORCE UTC for BOTH cases
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        if ts >= past:
            if filter_team:
                teams = row.get("teams_mentioned") or []
                if filter_team not in teams:
                    continue

            recent.append(row)

    return len(recent), get_article_count(filter_team)
# getting most mentioned team
def most_covered(number=1):
    data = fetch_sentiment_data()
    all_teams = []
    for row in data:
        teams = row.get("teams_mentioned") or []
        all_teams.extend(teams)

    team_counts = Counter(all_teams)
    # return just team names
    return team_counts.most_common(number)


# sentiment analysis (reusable)
def compute_sentiment_overall(filter_team=None):
    data = fetch_sentiment_data()

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
    data = fetch_sentiment_data()

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

# ================= RAG FUNCTIONS =================
def get_recent_articles_for_rag(filter_team=None, limit=10):
    data = fetch_sentiment_data()

    now = datetime.now(timezone.utc)
    past = now - timedelta(hours=24)

    filtered = []

    for row in data:
        ts = row.get("analyzed_at")
        if not ts:
            continue

        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except:
            continue

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        if ts < past:
            continue

        if filter_team:
            teams = row.get("teams_mentioned") or []
            if filter_team not in teams:
                continue

        filtered.append(row)

    filtered.sort(key=lambda x: x.get("analyzed_at", ""), reverse=True)
    return filtered[:limit]


def build_rag_context(articles):
    context = ""

    for art in articles:
        title = art.get("title", "")
        summary = art.get("summary", "")
        sentiment = art.get("sentiment", "")

        context += f"""
Title: {title}
Summary: {summary}
Sentiment: {sentiment}
---
"""

    return context[:4000]
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

#========team sentiment score===========================
def get_team_sentiment_score(team):
    data = fetch_sentiment_data()

    sentiments = []

    for row in data:
        teams = row.get("teams_mentioned") or []
        if team in teams:
            sentiments.append(row.get("sentiment"))

    counts = Counter(sentiments)

    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    neutral = counts.get("neutral", 0)

    total = positive + negative + neutral

    if total == 0:
        return 0

    return (positive - negative) / total
#get extremes
def get_extreme_teams():
    team_list = [team for team, _ in most_covered(20)]

    scores = []

    for team in team_list:
        score = get_team_sentiment_score(team)
        scores.append((team, score))

    most_positive = max(scores, key=lambda x: x[1])
    most_negative = min(scores, key=lambda x: x[1])

    return most_positive, most_negative
#display on dashboard

if selected_team == "All":
    pos_team, neg_team = get_extreme_teams()
    col4, col5 = st.columns(2)

    with col4:
        st.metric(
            label="🔥 Most Positive Team",
            value=pos_team[0],
            delta=f"{pos_team[1]:.2f}"
        )

    with col5:
        st.metric(
            label="❄️ Most Negative Team",
            value=neg_team[0],
            delta=f"{neg_team[1]:.2f}"
        )

else:
    score = get_team_sentiment_score(selected_team)
    st.metric(
        label=f"🧠 Sentiment Score for {selected_team}",
        value=selected_team,
        delta=f"{score:.2f}"
    )

# --------- Insight Section -----------
st.markdown("---")
st.subheader("🧠 Insight")

def generate_insight(filter_team=None):
    counts = get_sentiment_counts(filter_team)
    total = counts["positive"] + counts["neutral"] + counts["negative"]

    if total == 0:
        return "No data available for insights."

    pos_pct = counts["positive"] / total
    neg_pct = counts["negative"] / total
    neu_pct = counts["neutral"] / total

    team_text = filter_team if filter_team else "Football coverage"

    if pos_pct >= 0.65:
        return f"🧠 Insight: {team_text} is strongly positive ({pos_pct:.0%} positive sentiment)."
    elif neg_pct >= 0.65:
        return f"🧠 Insight: {team_text} is strongly negative ({neg_pct:.0%} negative sentiment)."
    elif pos_pct > neg_pct and pos_pct > 0.5:
        return f"🧠 Insight: {team_text} leans positive overall with mixed coverage."
    elif neg_pct > pos_pct and neg_pct > 0.5:
        return f"🧠 Insight: {team_text} leans negative overall with mixed coverage."
    else:
        return f"🧠 Insight: {team_text} has balanced or neutral coverage across recent articles."

filter_team = selected_team if selected_team != "All" else None
insight = generate_insight(filter_team)
st.info(insight)

# NOTE: AI summary can be connected to Anthropic API for advanced explanations
# using claude models. Keep it button-triggered to control cost.
# ================= OPTIONAL AI SUMMARY =================
st.markdown("---")
st.subheader("🤖 AI Summary")

@st.cache_data(ttl=21600)
def get_ai_summary_cached(summary_prompt):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[
            {"role": "user", "content": summary_prompt}
        ]
    )

    return message.content[0].text

if st.button("Generate AI Summary"):
    articles = get_recent_articles_for_rag(filter_team)
    context = build_rag_context(articles)
    summary_prompt = f"""
    You are analyzing recent football news sentiment using REAL articles.

    Here are the actual articles:
    {context}

    Sentiment Stats:
    Positive: {get_sentiment_counts(filter_team)['positive']}
    Negative: {get_sentiment_counts(filter_team)['negative']}
    Neutral: {get_sentiment_counts(filter_team)['neutral']}

    Instructions:
    - You MUST reference specific articles (mention titles or events)
    - Do NOT give generic explanations
    - Identify 2–4 concrete reasons for the sentiment
    - Mention real teams, players, or events from the data
    - If no strong signals exist, say that explicitly

    Write a concise but specific explanation grounded in the articles.
    if there are no articles presented you can just generalize using the sentiment counts
    """


    st.markdown(
        f"""
        <div style="
            padding: 12px;
            border-radius: 8px;
            background-color: #0e1117;
            border: 1px solid #2c2f36;
            color: #e6e6e6;
            font-size: 15px;
            line-height: 1.5;
        ">
            {get_ai_summary_cached(summary_prompt)}
        </div>
        """,
        unsafe_allow_html=True
    )


