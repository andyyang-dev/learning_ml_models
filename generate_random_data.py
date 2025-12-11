from typing import List, Tuple, Dict
from enum import Enum
import random
import numpy as np
import pandas as pd

class GirlType(Enum):
    WIFEY_MATERIAL = "Wifey Material"
    PROFESSIONAL_PARTNER = "Professional Partner"
    SEXUALLY_ENTICING = "Sexually Enticing"
    STRANGER = "Stranger"

def get_linear_data(n: int, range_tuple: Tuple[float, float, float], parameter: np.ndarray) -> Tuple[List, List]:
    min_x_data, max_x_data, max_deviation = range_tuple
    b0, b1 = parameter

    def adjusted_random_x(random_x):
        is_adjusted = random.random() > 0.5
        if is_adjusted:
            if 0.4 < random_x < 0.6:
                random_x += 0.2
            elif min_x_data < random_x < 0.4:
                random_x += 0.3
        return random_x

    deviations, x_data, y_data = [], [], []

    for _ in range(n):
        deviation = random.random() * max_deviation
        deviations.append(deviation)
        deviations.append(deviation * -1)

    for deviation in deviations:
        random_y = -1
        while random_y < 0.05 or random_y > 0.95:
            random_x = random.random() * (max_x_data - min_x_data) + min_x_data
            random_x = adjusted_random_x(random_x)
            random_y = random_x*b1 + b0 + deviation
        x_data.append(random_x)
        y_data.append(random_y)
    
    return (x_data, y_data)

def get_logistic_data(n: int) -> Tuple[List, List]:
    x_data = [random.random() for _ in range(50)]
    y_data = []
    for x in x_data:
        is_outliers = random.random() < 0.3
        if x < 0.5: 
            if x < 0.3 or not is_outliers: y_data.append(0)
            else: y_data.append(1)
        else:
            if x > 0.7 or not is_outliers: y_data.append(1)
            else: y_data.append(0)
    return (x_data, y_data)

def get_decision_tree_data(n: int) -> pd.DataFrame:
    def get_distribution(n: int) -> Dict:
        first_distribution_diff = random.random() * 3 / 100 + 0.02
        second_distribution_diff = random.random() * 2 / 100
        third_distribution_diff = random.random() * 2 / 100
        more_distribution = 0.5 + first_distribution_diff
        less_distribution = 0.5  - first_distribution_diff
        n_professional_partner = round(n  * more_distribution * (0.5 + second_distribution_diff))
        n_sexually_enticing = round(n  * more_distribution * (0.5 - second_distribution_diff))
        n_wifey_material = round(n  * less_distribution * (0.5 - third_distribution_diff))
        return {
            GirlType.PROFESSIONAL_PARTNER.value: n_professional_partner,
            GirlType.SEXUALLY_ENTICING.value: n_sexually_enticing,
            GirlType.WIFEY_MATERIAL.value: n_wifey_material,
            GirlType.STRANGER.value: n - n_wifey_material - n_professional_partner - n_sexually_enticing
        }
    def adjust_df(df: pd.DataFrame) -> pd.DataFrame:
        mask = df["Girl Type"] == GirlType.STRANGER.value 
        df.loc[mask, features] = df.loc[mask, features] * 0.40
        mask = df["Girl Type"] == GirlType.PROFESSIONAL_PARTNER.value
        df.loc[mask, features[2]] = df.loc[mask, features[2]] * 0.65 + 0.45
        df.loc[mask, features[3]] = df.loc[mask, features[3]] * 0.50
        df.loc[mask & (df[features[2]] < 0.5), features[4]] = df.loc[mask & (df[features[2]] < 0.5), features[4]] * 0.34 + 0.66
        df.loc[mask & (df[features[2]] >= 0.5), features[4]] = df.loc[mask & (df[features[2]] >= 0.5), features[4]] * 0.50 + 0.50
        mask = df["Girl Type"] == GirlType.SEXUALLY_ENTICING.value
        df.loc[mask, features[1]] = df.loc[mask, features[1]] * 0.45 + 0.55
        df.loc[mask, features[2]] = df.loc[mask, features[2]] * 0.70
        df.loc[mask & (df[features[1]] < 0.66), features[0]] = df.loc[mask & (df[features[1]] < 0.66), features[0]] * 0.30 + 0.70
        df.loc[mask & (df[features[1]] >= 0.66), features[0]] = df.loc[mask & (df[features[1]] >= 0.66), features[0]] * 0.45 + 0.55
        df.loc[mask, features[4]] = df.loc[mask, features[4]] * 0.55
        mask = df["Girl Type"] == GirlType.WIFEY_MATERIAL.value
        df.loc[mask, features[4]] = df.loc[mask, features[4]] * 0.30 + 0.70
        df.loc[mask, features[1]] = df.loc[mask, features[1]] * 0.45 + 0.25
        df.loc[mask & (df[features[4]] < 0.66), features[3]] = df.loc[mask & (df[features[4]] < 0.66), features[3]] * 0.20 + 0.80
        df.loc[mask & (df[features[4]] >= 0.66), features[3]] = df.loc[mask & (df[features[4]] >= 0.66), features[3]] * 0.40 + 0.60
        df.loc[mask, features[0]] = df.loc[mask, features[0]] * 0.45 + 0.55
        return df
        
    girl_type_arr = []
    for girl_type, count in get_distribution(n).items():
        girl_type_arr += [girl_type] * count
    
    features = ["Facial Attractiveness", "Body Attractiveness", "Professional Scale", "Feminime Energy Scale", "Attitude Scale"]
    data_dict = {feature: [random.random() for _ in range(n)] for feature in features}
    data_dict["Girl Type"] = girl_type_arr

    df = pd.DataFrame(data_dict)
    df = adjust_df(df)
    scale_categories = {
        "Facial Attractiveness": [["Ugly", "Mid", "Beautiful"], 3],
        "Body Attractiveness": [["Bad", "Present", "Hot"], 3],
        "Professional Scale": [["Not Professional", "Professional"], 2],
        "Feminime Energy Scale": [["Masculine", "Ambivert", "Feminime"], 3],
        "Attitude Scale": [["Bad", "Normal", "Good"], 3],
    }
    for category, (labels, bins) in scale_categories.items():
        df[category] = pd.cut(df[category], bins=bins,labels=labels, right=False)
    return df.sample(frac=1).reset_index(drop=True)

def get_naive_bayes_data(n: int) -> pd.DataFrame:
    def get_distribution(n: int) -> Dict:
        first_distribution_diff = random.random() * 3 / 100 + 0.02
        second_distribution_diff = random.random() * 2 / 100
        third_distribution_diff = random.random() * 2 / 100
        more_distribution = 0.5 + first_distribution_diff
        less_distribution = 0.5  - first_distribution_diff
        n_professional_partner = round(n  * more_distribution * (0.5 + second_distribution_diff))
        n_sexually_enticing = round(n  * more_distribution * (0.5 - second_distribution_diff))
        n_wifey_material = round(n  * less_distribution * (0.5 - third_distribution_diff))
        return {
            GirlType.PROFESSIONAL_PARTNER.value: n_professional_partner,
            GirlType.SEXUALLY_ENTICING.value: n_sexually_enticing,
            GirlType.WIFEY_MATERIAL.value: n_wifey_material,
            GirlType.STRANGER.value: n - n_wifey_material - n_professional_partner - n_sexually_enticing
        }
    distributions = get_distribution(n)
    scalars = ["Facial Attractiveness", "Body Attractiveness", "Professional Scale", "Feminime Energy Scale", "Attitude Scale"]
    data = {key: np.array([]) for key in scalars}
    data["Girl Type"] = []
    set_label = lambda scalar, label, distribution, p: np.concat([data[scalar], np.random.choice(label, distribution, p=p)])
    for girl_type, distribution in distributions.items():
        if girl_type == GirlType.PROFESSIONAL_PARTNER.value:
            data["Facial Attractiveness"] = set_label("Facial Attractiveness", ["Ugly", "Mid", "Beautiful"], distribution, [.38, .47, .15])
            data["Body Attractiveness"] = set_label("Body Attractiveness", ["Not Ideal", "Normal", "Hot"], distribution, [.35, .45, .2])
            data["Professional Scale"] = set_label("Professional Scale", ["Not Professional", "Professional"], distribution, [.05, .95])
            data["Feminime Energy Scale"] = set_label("Feminime Energy Scale", ["Masculine", "Feminime"], distribution, [.7, .3])
            data["Attitude Scale"] =set_label("Attitude Scale", ["Normal", "Nice"], distribution, [.15, .85])
        elif girl_type == GirlType.SEXUALLY_ENTICING.value:
            data["Facial Attractiveness"] = set_label("Facial Attractiveness", ["Mid", "Beautiful"], distribution, [.15, .85])
            data["Body Attractiveness"] = set_label("Body Attractiveness", ["Normal", "Hot"], distribution, [.05, .95])
            data["Professional Scale"] = set_label("Professional Scale", ["Not Professional", "Professional"], distribution, [.6, .4])
            data["Feminime Energy Scale"] = set_label("Feminime Energy Scale", ["Masculine", "Feminime"], distribution, [.4, .6])
            data["Attitude Scale"] =set_label("Attitude Scale", ["Not Nice", "Normal", "Nice"], distribution, [.35, .45, .2])
        elif girl_type == GirlType.WIFEY_MATERIAL.value:
            data["Facial Attractiveness"] = set_label("Facial Attractiveness", ["Mid", "Beautiful"], distribution, [.1, .9])
            data["Body Attractiveness"] = set_label("Body Attractiveness", ["Normal", "Hot"], distribution, [.3, .7])
            data["Professional Scale"] = set_label("Professional Scale", ["Not Professional", "Professional"], distribution, [.5, .5])
            data["Feminime Energy Scale"] = set_label("Feminime Energy Scale", ["Masculine", "Feminime"], distribution, [.1, .9])
            data["Attitude Scale"] = set_label("Attitude Scale", ["Normal", "Nice"], distribution, [.05, .95])
        elif girl_type == GirlType.STRANGER.value:
            data["Facial Attractiveness"] = set_label("Facial Attractiveness", ["Ugly", "Mid"], distribution, [.7, .3])
            data["Body Attractiveness"] = set_label("Body Attractiveness", ["Not Ideal", "Normal"], distribution, [.7, .3])
            data["Professional Scale"] = set_label("Professional Scale", ["Not Professional", "Professional"], distribution, [.7, .3])
            data["Feminime Energy Scale"] = set_label("Feminime Energy Scale", ["Masculine", "Feminime"], distribution, [.7, .3])
            data["Attitude Scale"] =set_label("Attitude Scale", ["Not Nice", "Normal"], distribution, [.7, .3])
        data["Girl Type"] += [girl_type] * distribution
    return pd.DataFrame(data)

import random
import pandas as pd

def get_svm_data(n: int) -> pd.DataFrame:
    n_short_not_ideal = round(n * 0.25)
    n_ideal = round(n * 0.4)
    n_tall_not_ideal = round(n * 0.25)
    n_outliers = n - n_short_not_ideal - n_ideal - n_tall_not_ideal
    
    x_data, y_data = [], []

    for _ in range(n_ideal):
        x_data.append(random.random() * 25 + 150)
        y_data.append("Ideal")

    for _ in range(n_short_not_ideal):
        x_data.append(random.random() * 25 + 120)
        y_data.append("Too short")

    for _ in range(n_tall_not_ideal):
        x_data.append(random.random() * 30 + 180)
        y_data.append("Too tall")

    for _ in range(n_outliers):
        if random.random() < 0.5:
            x_point = random.random() * 10 + 145
            x_data.append(x_point)
            y_data.append("Ideal" if x_point >= 150 else "Too short")
        else:
            x_point = random.random() * 15 + 175
            x_data.append(x_point)
            y_data.append("Ideal" if x_point <= 180 else "Too tall")

    return pd.DataFrame({"Height (cm)": x_data, "Class": y_data})

def get_knn_kmeans_data(n: int) -> pd.DataFrame:
    data_x_1 = np.random.rand(n//3) * 0.38 + 0.52
    adjustments = np.random.rand(n//3)
    data_y_1 = 1.20 - data_x_1
    data_y_1 = ((0.97 - data_y_1) * adjustments) + data_y_1 
    data_y_1[data_y_1 <= 0.45] += 0.07


    data_x_2 = np.random.rand(n//3) * 0.27 + 0.10
    adjustments = np.random.rand(n//3)
    data_y_2 = data_x_2 * 1 + 0.15
    data_y_2 = ((0.97 - data_y_2) * adjustments) + data_y_2 

    data_x_3 = np.random.rand(n//3) * 0.39 + 0.43
    adjustments = np.random.rand(n//3)
    data_y_3 = data_x_3 / - 4.4 + 0.53
    data_y_3 = (data_y_3 - 0.05) - adjustments * (data_y_3 - 0.08)

    careers = []
    for career in ["Model", "Artist", "Flight Attendance"]:
        careers += [career] * (n//3)

    df = pd.DataFrame({
        "Attractiveness Scale": np.concat([data_x_1, data_x_2, data_x_3]),
        "Creativity Scale": np.concat([data_y_1, data_y_2, data_y_3]),
        "Career": careers
    })

    return df.sample(frac=1).reset_index(drop=True)
