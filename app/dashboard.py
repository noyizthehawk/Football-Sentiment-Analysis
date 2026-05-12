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

# --- Mode selector for Teams/Players ---
mode = st.radio("View Sentiment By:", ["Teams", "Players/Managers"], horizontal=True)


if mode == "Teams":
    entity_field = "teams_mentioned"
else:
    entity_field = "players_mentioned"

# --- Player images (add more as needed) ---
PLAYER_IMAGES = {
    # Replace with reliable URLs you prefer
    "Lamine Yamal": "https://upload.wikimedia.org/wikipedia/commons/e/e3/Lamine_Yamal_in_2025.jpg",
    "Erling Haaland": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Erling_Haaland_2023.jpg/320px-Erling_Haaland_2023.jpg",
    "Kylian Mbappé": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Kylian_Mbappe_2022.jpg/320px-Kylian_Mbappe_2022.jpg",
    "Kai Havertz" : "https://upload.wikimedia.org/wikipedia/commons/e/e8/2019-06-11_Fußball%2C_Männer%2C_Länderspiel%2C_Deutschland-Estland_StP_2059_LR10_by_Stepro.jpg"
}



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
        items = row.get(entity_field) or []
        if filter_team in items:
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
                items = row.get(entity_field) or []
                if filter_team not in items:
                    continue

            recent.append(row)

    return len(recent), get_article_count(filter_team)
def most_covered(number=1):
    data = fetch_sentiment_data()
    all_teams = []
    for row in data:
        teams = row.get("teams_mentioned") or []
        all_teams.extend(teams)

    team_counts = Counter(all_teams)
    # return just team names
    return team_counts.most_common(number)

# --- Generalized entity counter for Teams/Players
def get_entities(field, number=20):
    data = fetch_sentiment_data()
    all_items = []
    for row in data:
        items = row.get(field) or []
        all_items.extend(items)
    return Counter(all_items).most_common(number)


# sentiment analysis (reusable)
def compute_sentiment_overall(filter_team=None):
    data = fetch_sentiment_data()

    sentiments = []

    for row in data:
        items = row.get(entity_field) or []

        # if a team is specified, filter
        if filter_team:
            if filter_team not in items:
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
        items = row.get(entity_field) or []
        if filter_team:
            if filter_team not in items:
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
    past = now - timedelta(hours=168)

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
            items = row.get(entity_field) or []
            if filter_team not in items:
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


entity_list = [e for e, _ in get_entities(entity_field, 20)]
selected_entity = st.selectbox(f"Filter by {mode[:-1]}", ["All"] + entity_list)

# --- Selected player image ---
if mode == "Players" and selected_entity != "All":
    col_img, col_title = st.columns([1, 3])
    with col_img:
        img_url = PLAYER_IMAGES.get(selected_entity)
        if img_url:
            st.image(img_url, width=150)
        else:
            st.info("No image available for this player yet.")
    with col_title:
        st.subheader(selected_entity)

filter_entity = selected_entity if selected_entity != "All" else None

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Total Articles")
    recent_count, total_articles = get_recent_article_delta(
        12,
        filter_entity
    )

    st.metric(
        label="Total Articles",
        value=get_article_count(filter_entity),
        label_visibility="collapsed",
        delta=f"+{recent_count} (last 12h)"
    )

with col2:
    st.markdown(f"### {'⚽ Most Covered Team' if mode == 'Teams' else '⭐ Most Covered Player'}")
    st.metric(
        label=f"Most Covered {mode[:-1]}",
        value=selected_entity if selected_entity != "All" else entity_list[0],
        label_visibility="collapsed"
    )

with col3:
    st.markdown("### 🧠 Overall Sentiment")
    st.metric(
        label="Overall Sentiment",
        value=compute_sentiment_overall(filter_entity),
        label_visibility="collapsed"
    )
#================pie chart ==============
counts = get_sentiment_counts(filter_entity)
labels = ["Positive", "Neutral", "Negative"]
sizes = [
    counts["positive"],
    counts["neutral"],
    counts["negative"]
]
# FIX: handle empty or invalid data
if sum(sizes) == 0:
    sizes = [1, 1, 1]
    labels = ["No Data", "", ""]

pie_fig, pie_ax = plt.subplots()
pie_fig.patch.set_facecolor("#0e1117")
pie_ax.set_facecolor("#0e1117")
colors = ["#2ecc71", "#95a5a6", "#e74c3c"]  # green, gray, red

pie_ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%" if sum(sizes) > 0 else None,
    colors=colors,
    startangle=90,
    wedgeprops={"width": 0.4},
    textprops={"color": "white"}
)
pie_ax.axis("equal")

#top 10 entities bar==========================
top_entities = get_entities(entity_field, 10)
entities = [entity for entity, count in top_entities]
entity_counts = [count for entity, count in top_entities]

bar_fig, bar_ax = plt.subplots()
colors = ["#e74c3c" if entity == selected_entity else "#4C78A8" for entity in entities]
bar_ax.barh(entities, entity_counts, color=colors)
bar_ax.set_xlabel("Number of Articles")
bar_ax.set_title(f"Top 10 Most Covered {mode}")
bar_ax.invert_yaxis()

left, right = st.columns(2)
with left:
    st.subheader("📊 Sentiment Breakdown")
    st.pyplot(pie_fig)
with right:
    st.subheader(f"🏟️ Top 10 {mode}")
    st.pyplot(bar_fig)

#========entity sentiment score===========================
def get_entity_sentiment_score(entity, entity_field):
    data = fetch_sentiment_data()

    sentiments = []

    for row in data:
        items = row.get(entity_field) or []
        if entity in items:
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
def get_extreme_entities(entity_field):
    entity_list = [e for e, _ in get_entities(entity_field, 20)]

    scores = []

    for entity in entity_list:
        score = get_entity_sentiment_score(entity, entity_field)
        scores.append((entity, score))

    most_positive = max(scores, key=lambda x: x[1])
    most_negative = min(scores, key=lambda x: x[1])

    return most_positive, most_negative
#display on dashboard

if selected_entity == "All":
    pos_entity, neg_entity = get_extreme_entities(entity_field)
    col4, col5 = st.columns(2)

    with col4:
        st.metric(
            label=f"🔥 Most Positive {mode[:-1]}",
            value=pos_entity[0],
            delta=f"{pos_entity[1]:.2f}"
        )

    with col5:
        st.metric(
            label=f"❄️ Most Negative {mode[:-1]}",
            value=neg_entity[0],
            delta=f"{neg_entity[1]:.2f}"
        )

else:
    score = get_entity_sentiment_score(selected_entity, entity_field)
    st.metric(
        label=f"🧠 Sentiment Score for {selected_entity}",
        value=selected_entity,
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

insight = generate_insight(filter_entity)
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
    articles = get_recent_articles_for_rag(filter_entity)

    # Handle empty article case (common for players)
    if not articles:
        st.info("Not enough recent article data to generate a reliable AI summary.")
    else:
        context = build_rag_context(articles)

        entity_type = "team" if mode == "Teams" else "player"
        counts = get_sentiment_counts(filter_entity)

        summary_prompt = f"""
        You are analyzing recent football news sentiment for a {entity_type}.

        Entity: {filter_entity if filter_entity else 'All'}

        Here are the actual articles:
        {context}

        Sentiment Stats:
        Positive: {counts['positive']}
        Negative: {counts['negative']}
        Neutral: {counts['neutral']}

        Instructions:
        - If articles are available, reference specific titles or events
        - If data is weak, rely on sentiment trends instead
        - Identify 2–4 concrete reasons for the sentiment
        - Mention real teams, players, or events from the data when possible
        - If no strong signals exist, say that explicitly

        Write a concise but grounded explanation.
        """

        result = get_ai_summary_cached(summary_prompt)

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
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )
def get_article_title(article):
    return (
        article.get("title")
        or article.get("headline")
        or article.get("url", "").split("/")[-1].replace("-", " ").title()
        or "Untitled"
    )
def retrieve_articles(filter_entity=None, limit=10):
    data = fetch_sentiment_data()

    filtered = []

    for row in data:
        items = row.get(entity_field) or []
        if filter_entity and filter_entity not in items:
            continue
        filtered.append(row)

    # sort by most recent
    def get_time(x):
        ts = x.get("analyzed_at")
        if not ts:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except:
            return datetime.min.replace(tzinfo=timezone.utc)

    filtered.sort(key=get_time, reverse=True)

    return filtered[:limit]


@st.cache_data(ttl=18000)
def fetch_articles_with_details(filter_entity=None, limit=10):
    sentiment_data = retrieve_articles(filter_entity, limit)

    # Get all article IDs
    article_ids = [row.get("article_id") for row in sentiment_data if row.get("article_id")]

    if not article_ids:
        return []

    # Fetch full article data
    articles_response = supabase.table("articles").select("*").in_("id", article_ids).execute()
    articles_dict = {art["id"]: art for art in articles_response.data}

    # Merge sentiment + article data
    merged = []
    for sent in sentiment_data:
        article_id = sent.get("article_id")
        if article_id in articles_dict:
            merged.append({**articles_dict[article_id], **sent})

    return merged


# ================= ARTICLE VIEWER =================
st.markdown("---")
st.subheader("📰 Recent Articles")

num_articles = st.slider("Number of articles to display", 5, 50, 10)
articles = fetch_articles_with_details(filter_entity, num_articles)

if not articles:
    st.info("No articles found for the selected filter.")
else:
    for idx, article in enumerate(articles, 1):
        title = article.get("title") or article.get("headline") or "Untitled Article"
        summary = article.get("summary", "No summary available.")
        sentiment = article.get("sentiment", "unknown")
        entities = article.get(entity_field, [])
        url = article.get("url") or article.get("link", "")
        analyzed_at = article.get("analyzed_at", "")

        # Format timestamp
        try:
            ts = datetime.fromisoformat(analyzed_at.replace("Z", "+00:00"))
            time_str = ts.strftime("%b %d, %Y %I:%M %p")
        except:
            time_str = "Unknown time"

        # Sentiment color
        sentiment_colors = {
            "positive": "🟢",
            "neutral": "🟡",
            "negative": "🔴"
        }
        sentiment_icon = sentiment_colors.get(sentiment, "⚪")

        with st.expander(f"{sentiment_icon} {idx}. {title}"):
            st.markdown(f"**Published:** {time_str}")

            if entities:
                st.markdown(f"**{mode} Mentioned:** {', '.join(entities)}")

            st.markdown(f"**Sentiment:** {sentiment.capitalize()}")
            st.markdown(f"**Summary:**")
            st.write(summary)

            if url:
                st.markdown(f"[Read full article →]({url})")