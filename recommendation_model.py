import numpy as np
import pandas as pd
import pickle

# Expect these to be imported or passed in from app.py
# user_final_rating, df_sent_analysis, tfidf_vectorizer, rf_best_model

# Create function to recommend top 5 products to any user
def product_recommendations_user(user_name):
    """
    Recommend top 5 products for a given user based on:
    - Collaborative filtering (user_final_rating)
    - Sentiment analysis on top 20 candidate products (df_final + rf_model)

    Returns:
        - pd.DataFrame with columns ['name', 'pos_sent_percentage']
        - or None if user does not exist or no products found
    """

    # 1. Load the best recommendation matrix
    user_final_rating = pickle.load(open('models/user_final_rating.pkl', 'rb'))

    # 2. Validate user
    if user_name not in user_final_rating.index:
        return None

    # 3. Get top 20 recommended product *IDs* from user_final_rating
    #    (assuming user_final_rating was built with product IDs as columns)
    top20_product_ids = (
        user_final_rating.loc[user_name]
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    # 4. Load cleaned data containing reviews
    df_final = pickle.load(open('models/cleaned_data.pkl', 'rb'))

    #    Match on 'id' (product id), NOT 'name'
    df_top20_products = (
        df_final[df_final["id"].isin(top20_product_ids)]
        .drop_duplicates(subset=["cleaned_review"])
        .copy()
    )

    if df_top20_products.empty:
        return None

    # 5. Load TF-IDF vectorizer and transform cleaned reviews
    tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    X_tfidf = tfidf_vectorizer.transform(df_top20_products["cleaned_review"].astype(str).values)

    # Create DataFrame from TF-IDF transformed data with correct feature names
    X_df = pd.DataFrame(
        X_tfidf.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out()
    ).reset_index(drop=True)

    # 6. Numeric features (must match what was used during training)
    X_num = df_top20_products[["review_length"]].reset_index(drop=True)

    # Concatenate TF-IDF and numeric features
    df_top_20_products_final_features = pd.concat([X_df, X_num], axis=1)

    # 7. Load the best sentiment model (Random Forest)
    rf_model = pickle.load(open('models/sentiment_classification_random_forest_best_tuned.pkl', 'rb'))

    # Predict sentiment on the combined features
    df_top20_products["predicted_sentiment"] = rf_model.predict(df_top_20_products_final_features)

    # 8. Map sentiment labels to 1 (positive) / 0 (negative)
    # Assuming 1 = Positive, 0 = Negative
    df_top20_products["positive_sentiment"] = df_top20_products["predicted_sentiment"].apply(
        lambda x: 1 if x == 1 else 0
    )

    # 9. Aggregate sentiment by product name
    agg_df = df_top20_products.groupby("name").agg(
        pos_sent_count=("positive_sentiment", "sum"),
        total_sent_count=("predicted_sentiment", "count")
    )

    # 10. Compute positive sentiment percentage
    agg_df["pos_sent_percentage"] = np.round(
        agg_df["pos_sent_count"] / agg_df["total_sent_count"] * 100, 2
    )

    # 11. Reset index so 'name' is a column
    agg_df = agg_df.reset_index()

    # 12. Sort by positive sentiment percentage and keep top 5
    result = (
        agg_df.sort_values(by="pos_sent_percentage", ascending=False)
        .head(5)[["name", "pos_sent_percentage"]]
        .reset_index(drop=True)
    )

    return result
