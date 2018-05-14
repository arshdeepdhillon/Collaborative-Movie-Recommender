import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader, Dataset, SVD, model_selection

nl = "\n"
# ratings = pd.read_csv('/resources/ratings.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
print(ratings.head(), nl)

# The Reader class is used to parse a file containing ratings.
# Such a file is assumed to specify only one rating per line, and each line needs to respect the following structure:
#     user ; item ; rating ; [timestamp]
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)

svd = SVD()
print("CROSS VALIDATING THE DATASET", nl)
model_selection.cross_validate(svd, data, measures=['rmse'], verbose=True)
print("DONE CROSS VALIDATING THE DATASET", nl)

print("TRAINING THE MODEL")
trainset = data.build_full_trainset()
svd.fit(trainset)
print("DONE TRAINING THE MODEL", nl)

print(ratings[ratings['userId'] == 1])

# r_ui -> The true rating
# iid  -> The item id (movie id)
# uid  -> The user id
print("Predicting rating for ",svd.predict(uid=1, iid=302, r_ui=1))

"""
print("PRINTING RECOMMENDATION")
from collections import defaultdict

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = svd.test(testset)

top_n = get_top_n(predictions[0:1], n=10)

# Print the recommended items for each user
print("PRINTING RECOMMENDATION")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
"""

