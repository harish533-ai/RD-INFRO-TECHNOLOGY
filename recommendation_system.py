import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "Item": ["Python Course", "ML Course", "AI Course", "Web Dev Course"],
    "Category": ["Programming", "AI", "AI", "Web"]
}

df = pd.DataFrame(data)
encoded = pd.get_dummies(df["Category"])
similarity = cosine_similarity(encoded)

def recommend(item):
    index = df[df["Item"] == item].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return df.iloc[scores[1][0]]["Item"]

print("Recommendation for AI Course:", recommend("AI Course"))
