from supabase import create_client
import anthropic
import pandas as pd
import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

#get client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase=create_client(url,key)

response = supabase.table("articles").select("*").execute()
data = response.data

ids = [article["id"] for article in data]
print("Total rows fetched:", len(ids))
print("Unique IDs:", len(set(ids)))

for article in data:
    try:
        article_text = article.get("description") or article.get("title", "")

        if not article_text:
            print(f"Skipping article {article['id']} - no content")
            continue
        article_text = article["description"]
        article_title = article["title"]
        prompt = (
            f"You are a football (soccer) news article analyzer. For this article: {article_text} "
            "determine whether it is related to football/soccer and extract structured insights. "
            "You must extract the following fields: "
            "is_football (boolean indicating if the article is about football/soccer), "
            "sentiment (positive, negative, or neutral), "
            "confidence (a float between 0 and 1 representing how confident you are in your analysis), "
            "teams_mentioned (list of football clubs mentioned), "
            "players_mentioned (list of football players mentioned), "
            "summary (a concise 1–3 sentence summary suitable for an LLM). "
            "Return ONLY a valid JSON object with no extra text before or after it."
        )
        claude_response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        text_output = claude_response.content[0].text.strip()
        text_output = re.sub(r"```json|```", "", text_output).strip()
        output = json.loads(text_output)
        if not output["is_football"]:
            print(f"Skipping non-football article: {article_title}")
            continue

        supabase.table("sentiment").upsert({
            "article_id": article["id"],
            "is_football": output["is_football"],
            "sentiment": output["sentiment"],
            "confidence": output["confidence"],
            "teams_mentioned": output["teams_mentioned"],
            "players_mentioned": output["players_mentioned"],
            "summary": output["summary"]
        }).execute()

        print(f"Processed article {article['id']}")

    except Exception as e:
        print(f"Error processing article {article['id']}: {e}")
        continue

