from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def sentiment_analyse(df):
    sid = SentimentIntensityAnalyzer()
    df["sentiments"] = df["review_clean"].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)