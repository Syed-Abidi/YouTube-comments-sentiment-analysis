import streamlit as st
from googleapiclient.discovery import build
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# YouTube API details
API_KEY = 'AIzaSyB1T3lYOgRF9Kpg2-O4U89DCEDAVFefAUg'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Function to fetch comments from a YouTube video
def fetch_comments(video_id, uploader_channel_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    nextPageToken = None
    while len(comments) < 300:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        )
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            if comment['authorChannelId']['value'] != uploader_channel_id:
                comments.append(comment['textDisplay'])
        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break
    return comments

# Function to filter comments
def filter_comments(comments):
    hyperlink_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    threshold_ratio = 0.65
    relevant_comments = []
    for comment_text in comments:
        comment_text = comment_text.lower().strip()
        emojis = emoji.emoji_count(comment_text)
        text_characters = len(re.sub(r'\s', '', comment_text))
        if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
            if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
                relevant_comments.append(comment_text)
    return relevant_comments

# Function to analyze sentiments of comments
def analyze_sentiments(comments):
    analyzer = SentimentIntensityAnalyzer()
    polarity = []
    positive_comments = []
    negative_comments = []
    neutral_comments = []
    for comment in comments:
        sentiment_dict = analyzer.polarity_scores(comment)
        polarity_score = sentiment_dict['compound']
        polarity.append(polarity_score)
        if polarity_score > 0.05:
            positive_comments.append(comment)
        elif polarity_score < -0.05:
            negative_comments.append(comment)
        else:
            neutral_comments.append(comment)
    return polarity, positive_comments, negative_comments, neutral_comments

# Streamlit app
st.set_page_config(page_title="YouTube Comment Sentiment Analysis")

# Load image
image_url = "https://cdn3.iconfinder.com/data/icons/social-network-30/512/social-06-512.png"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Custom CSS
st.markdown(
    """
    <style>
    .title-font {
        font-family: 'Roboto', sans-serif;
        color: #333333;
        font-size: 36px;
    }
    .output-font {
        font-family: 'Roboto', sans-serif;
        color: #333333;
        font-size: 20px;
    }
    .highlight-font {
        font-family: 'Roboto', sans-serif;
        color: #333333;
        font-size: 22px;
        font-weight: bold;
    }
    .positive {
        color: green;
    }
    .negative {
        color: red;
    }
    .comment-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 4])
with col1:
    st.image(img, width=100)
with col2:
    st.markdown("<h1 class='title-font'>YouTube Comment Sentiment Analysis</h1>", unsafe_allow_html=True)

video_url = st.text_input("Enter YouTube Video URL:")
if video_url:
    video_id = video_url.split("v=")[-1]
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    video_response = youtube.videos().list(part='snippet', id=video_id).execute()
    video_snippet = video_response['items'][0]['snippet']
    video_title = video_snippet['title']
    channel_title = video_snippet['channelTitle']
    uploader_channel_id = video_snippet['channelId']
    
    st.markdown(f"<div class='output-font'>Video Title: {video_title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='output-font'>Channel Name: {channel_title}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='output-font'>Fetching Comments...</div>", unsafe_allow_html=True)
    comments = fetch_comments(video_id, uploader_channel_id)
    relevant_comments = filter_comments(comments)
    
    if relevant_comments:
        st.markdown("<div class='output-font'>Displaying 5 comments:</div>", unsafe_allow_html=True)
        for i, comment in enumerate(relevant_comments[:5]):
            st.markdown(f"<div class='output-font comment-box'>{i+1}. {comment}</div>", unsafe_allow_html=True)
        
        polarity, positive_comments, negative_comments, neutral_comments = analyze_sentiments(relevant_comments)
        
        avg_polarity = sum(polarity) / len(polarity)
        st.markdown(f"<div class='highlight-font'>Average Sentiment Polarity: {avg_polarity:.2f}</div>", unsafe_allow_html=True)
        
        if avg_polarity > 0.05:
            st.markdown("<div class='highlight-font'>The Video has got a Positive response</div>", unsafe_allow_html=True)
        elif avg_polarity < -0.05:
            st.markdown("<div class='highlight-font'>The Video has got a Negative response</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='highlight-font'>The Video has got a Neutral response</div>", unsafe_allow_html=True)
        
        st.markdown(
            f"<div class='highlight-font'>The comment with most <span class='positive'>positive</span> sentiment: <i>\"{positive_comments[0]}\"</i> with score <span class='positive'>{max(polarity):.2f}</span></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='highlight-font'>The comment with most <span class='negative'>negative</span> sentiment: <i>\"{negative_comments[0]}\"</i> with score <span class='negative'>{min(polarity):.2f}</span></div>",
            unsafe_allow_html=True
        )
        
        labels = ['Positive', 'Negative', 'Neutral']
        comment_counts = [len(positive_comments), len(negative_comments), len(neutral_comments)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Bar chart
        ax1.bar(labels, comment_counts, color=['blue', 'red', 'grey'])
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Comment Count')
        ax1.set_title('Sentiment Analysis of Comments')
        
        # Pie chart
        ax2.pie(comment_counts, labels=labels, autopct='%1.1f%%')
        ax2.set_title('Sentiment Distribution')
        
        st.pyplot(fig)
        
    else:
        st.markdown("<div class='output-font'>No relevant comments found.</div>", unsafe_allow_html=True)
