from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def sentiment_analyse(df):
    sid = SentimentIntensityAnalyzer()
    df["sentiments"] = df["review_clean"].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)
    return df

def tfidf_transform(df, loc):
    tfidf = TfidfVectorizer(min_df = 10)
    tfidf_model = tfidf.fit(df["review_clean"])

    with open(loc + 'Tfidf.pickle', 'wb') as f:
        pickle.dump(tfidf_model, f)

    tfidf_result = tfidf.transform(df["review_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = df.index
    df_concat = pd.concat([df, tfidf_df], axis=1)

    return df_concat, tfidf_model