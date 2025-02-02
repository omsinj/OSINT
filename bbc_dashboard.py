import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import nltk
import re
from nltk.corpus import stopwords
import streamlit as st

nltk.download("stopwords")

# Step 1: Web Scraping BBC Headlines
def scrape_bbc_headlines():
    """
    Scrapes the latest headlines from BBC News using Selenium.
    Returns a Pandas DataFrame containing the headlines.
    """
    driver_path = r"C:\Users\ojaga\Desktop\chromedriver-win64\chromedriver.exe"
    service = Service(driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get("https://www.bbc.com/news")
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h3"))
        )
        soup = BeautifulSoup(driver.page_source, "html.parser")
    finally:
        driver.quit()
    
    headlines = [item.get_text(strip=True) for item in soup.find_all("h3")]
    return pd.DataFrame({"Headline": headlines, "Scraped At": datetime.now()})

# Step 2: Preprocess Text Data
def preprocess_text(text):
    """Cleans and tokenizes text for clustering."""
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    return " ".join(word for word in text.split() if word not in stopwords.words("english"))

# Step 3: Apply Clustering to Headlines
def cluster_headlines(df, num_clusters=3):
    """Applies K-Means clustering to BBC news headlines."""
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["Processed_Text"])
    
    model = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster"] = model.fit_predict(X)
    return df, model, vectorizer, X

# Step 4: Visualise Word Frequency
def plot_wordcloud(text_data):
    """Generates and displays a word cloud."""
    text = " ".join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of BBC Headlines", fontsize=14)
    st.pyplot(plt)

# Step 5: Build Streamlit Dashboard
def run_dashboard(df):
    """Creates an interactive dashboard using Streamlit."""
    st.title("BBC OSINT Dashboard")
    st.subheader("Clustered BBC News Headlines")
    st.dataframe(df)
    
    # Word Cloud
    st.subheader("Word Cloud of Headlines")
    plot_wordcloud(df["Processed_Text"])
    
    # Cluster Distribution
    st.subheader("Cluster Distribution")
    st.bar_chart(df["Cluster"].value_counts())
    
    # Clustered Headlines
    for cluster in sorted(df["Cluster"].unique()):
        st.subheader(f"Cluster {cluster}")
        st.write(df[df["Cluster"] == cluster]["Headline"].tolist())

# Run the Pipeline
df = scrape_bbc_headlines()
df["Processed_Text"] = df["Headline"].apply(preprocess_text)
df, model, vectorizer, X = cluster_headlines(df, num_clusters=3)

# Run Streamlit Dashboard
run_dashboard(df)