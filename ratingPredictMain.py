from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nlp_transform import sentiment_analyse, tfidf_transform
from transform import col_transform, text_transform
import pandas as pd


def Predict_Review_main():
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv') #Predict on this file
    labels_drop = ['Review Title']
    index_col = ['id']

    model_loc = 'models/'

    df_colTransform = col_transform(df_train, labels_drop, index_col)
    df_cleanReview = text_transform(df_colTransform)
    df_sentiment = sentiment_analyse(df_cleanReview)
    df_tfidf, tfidf_model = tfidf_transform(df_sentiment, model_loc)

    label = "Star_Rating"
    ignore_cols = [label, 'App_Version_Code',
    'App_Version_Name', "Review_Text", "review_clean","compound"]
    features = [c for c in df_tfidf.columns if c not in ignore_cols]


    #Performing Grid Search to get the best parameter

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    X_train, X_test, y_train, y_test = train_test_split(df_tfidf[features], df_tfidf[label], test_size = 0.20, random_state = 42)

    # Fit the random search model
    rf_random.fit(X_train, y_train)
    pipeline = Pipeline([('chi',  SelectKBest(chi2, k='all')),
                     ('clf', rf_random.estimator)])

    # fitting our model and save it in a pickle for later use
    model = pipeline.fit(X_train, y_train)
    
    ytest = np.array(y_test)

    # confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(ytest, model.predict(X_test)))
    print(confusion_matrix(ytest, model.predict(X_test)))

    with open(model_loc+'RandomForest.pickle', 'wb') as f:
        pickle.dump(model, f)

    #Predicting the new file

    df_colTransform_test = col_transform(df_test, labels_drop, index_col)
    df_cleanReview_test = text_transform(df_colTransform_test)
    df_sentiment_test = sentiment_analyse(df_cleanReview_test)

    tfidf_result_test = tfidf_model.transform(df_sentiment_test["review_clean"]).toarray()
    tfidf_df_test = pd.DataFrame(tfidf_result_test, columns = tfidf_model.get_feature_names())
    tfidf_df_test.columns = ["word_" + str(x) for x in tfidf_df_test.columns]
    tfidf_df_test.index = df_sentiment_test.index
    df_test_predict = pd.concat([df_sentiment_test, tfidf_df_test], axis=1)

    df_predict = df_test_predict[features]

    y_pred_final = model.predict(df_predict)
    df_test['Star Rating'] = y_pred_final

    df_test.to_csv("../output/predictions.csv", index=False)

if __name__ == '__main__':
    Predict_Review_main()