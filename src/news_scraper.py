import requests
import psycopg2
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("NEWS_API_KEY")
db_host = os.getenv("DB_HOST")
today = datetime.utcnow()

'''do the same for DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your-password
DB_PORT=5432'''

db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASSWORD")
db_port = os.getenv("DB_PORT")
keywords = [
    "football",
    "Premier League",
    "EPL",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "Champions League",
    "UCL"
]
conn = psycopg2.connect(
    dbname=db_name,
    user=db_user,
    password=db_pass,
    host=db_host,
    port=db_port
)
cur = conn.cursor()
for keyword in keywords:
    thirty_days_ago = today - timedelta(days=29)

    url = (
        f'https://newsapi.org/v2/everything?'
        f'q={keyword}&'
        f'from={thirty_days_ago.strftime("%Y-%m-%d")}&'
        f'to={today.strftime("%Y-%m-%d")}&'
        'sortBy=popularity&'
        f'apiKey={api_key}'
    )
    response = requests.get(url)
    data = response.json()
    if data.get("status") == "ok":

        for article in data.get("articles", []):
            title = article.get("title")
            description = article.get("description")
            url_link = article.get("url")
            source = article.get("source", {}).get("name")
            published_at = article.get("publishedAt")

            cur.execute("""
                        INSERT INTO articles (title, description, url, source, published_at, keyword)
                        VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (url) DO NOTHING
                        """, (title, description, url_link, source, published_at, keyword))

    else:
        print("API ERROR for keyword:", keyword)
        print(data)


conn.commit()
cur.close()
conn.close()





