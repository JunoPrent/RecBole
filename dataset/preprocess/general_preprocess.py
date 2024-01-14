import math


def item_preprocess(item_info, inter_info):
    # Stores all given ratings for each item
    all_item_ratings = {}
    for item_idx in item_info.index:
        all_item_ratings[item_idx] = list(inter_info[inter_info["item_id:token"] == item_idx]["rating:float"])

    # Average rating per item
    item_info["average rating"] = {item_idx: sum(all_item_ratings[item_idx]) / len(all_item_ratings[item_idx]) if len(all_item_ratings[item_idx]) >= 10 else None for item_idx in all_item_ratings.keys()}
    # Positive item (avg rating > 3.8), negative item (avg rating < 2.8) or None
    item_info["pos/neg item"] = {item_idx: "pos" if item_info.loc[item_idx]["average rating"] > 3.8 else "neg" if item_info.loc[item_idx]["average rating"] < 2.8 else None for item_idx in all_item_ratings.keys()}
    # Percentage of all ratings that belong to each item
    item_info["total ratings (%)"] = {item_idx: len(all_item_ratings[item_idx]) / len(inter_info.index) for item_idx in all_item_ratings.keys()}

    # Finds the most popular items that comprise 20% of the total amount of ratings (113 total)
    pop_threshold = 0.2
    cumulative_total = 0.0
    pop_labels = {}
    for i in item_info.sort_values("total ratings (%)", ascending=False).index:
        if cumulative_total > pop_threshold:
            pop_labels[i] = 0
        else:
            cumulative_total += item_info.loc[i]["total ratings (%)"]
            pop_labels[i] = 1

    item_info["popular item"] = pop_labels


def user_preprocess(user_info, item_info, inter_info):
    # Threshold for what fraction of a user's items should be popular for the user to be mainstream
    user_pop_threshold = 0.25

    # Stores all items each user has rated 
    user_items = {user_idx: list(inter_info[inter_info["user_id:token"] == user_idx]["item_id:token"]) for user_idx in user_info.index}    
    
    for u in user_items.keys():
        if len(user_items[u]) == 0:
            print(u)

    # Fraction of popular items rated by each user
    user_info["popular items rated (%)"] = {user_idx: (None if not user_items[user_idx] else item_info.loc[user_items[user_idx]]["popular item"].sum() / len(user_items[user_idx])) for user_idx in user_info.index}
    # Whether a user is mainstream (1) or not (0) based on the threshold
    user_info["mainstream user"] = {user_idx: int(user_info["popular items rated (%)"].loc[user_idx] >= user_pop_threshold) for user_idx in user_info.index}

    # Splits the users into mainstream classes based upon fraction of popular items rated by them (0-4, 4 means 40%+ popular items)
    user_info["mainstream class (thresholds)"] = {user_idx: math.floor(user_info["popular items rated (%)"].loc[user_idx] * 10) for user_idx in user_info.index}
    user_info.loc[user_info["mainstream class (thresholds)"] > 4, "mainstream class (thresholds)"] = 4

    # Assigns mainstream classes based on fraction thresholds (every 0.2 a new class)
    N = 5
    indices = user_info.sort_values("popular items rated (%)", ascending=False).index
    mainstream_classes = {}

    for i in range(0, N):
        slices = (round(len(indices) / N * i), round(len(indices) / N * (i + 1)))
        for idx in user_info.loc[indices[slices[0]: slices[1]]].index:
            mainstream_classes[idx] = (N - 1) - i

    user_info["mainstream class (even groups)"] = mainstream_classes