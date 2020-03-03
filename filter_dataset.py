import pandas as pd

from Utils import extract_parameter_value_as_int


gains1 = pd.read_csv("gains.csv", ",")[
    ["dataset", "clf", "clf_family", "clf_params", "od_name", "od_params", "removed", "accuracy", "accuracy_old",
     "gain"]]
gains2 = pd.read_csv("gains_nearest_neighbor_OLD.csv", ",")[
    ["dataset", "clf", "clf_family", "clf_params", "od_name", "od_params", "removed", "accuracy", "accuracy_old",
     "gain"]]
gains3 = pd.merge(gains1, gains2, how="outer",
                  on=["dataset", "clf", "clf_family", "clf_params", "od_name", "od_params", "removed", "accuracy",
                      "accuracy_old", "gain"])
gains3.drop_duplicates(inplace=True)
#gains3 = gains3.sort_values(["dataset", "clf", "clf_family", "clf_params", "od_name", "od_params", "removed"])
gains3['temp'] = gains3['od_params'].map(lambda x: extract_parameter_value_as_int(x, parameter="n_neighbors"))\
    .map(lambda x: 567 if type(x) == str else x)
gains3.sort_values(by="temp", inplace=True)
gains3.drop(['temp'], axis=1)
gains3 = gains3[gains3['removed'] != 0.0]
# "BayesNet", "AdaBoostM1", "DecisionTable", "JRip", "OneR", "RandomTree", "REPTree", "ZeroR"

# "badges2", "car", "car-evaluation", "cm1-req", "dermatology", "iris", "kr-vs-kp", "lung-cancer", "mushroom", "ozone-level-8hr"

gains3 = gains3[gains3['dataset'].map(lambda x: x not in ["badges2", "car", "car-evaluation", "cm1-req", "dermatology", "iris", "kr-vs-kp", "lung-cancer", "mushroom", "ozone-level-8hr"])]
gains3 = gains3[gains3['clf'].map(lambda x: x not in ["BayesNet", "AdaBoostM1", "DecisionTable", "JRip", "OneR", "RandomTree", "REPTree", "ZeroR"])]




gains3.to_csv('gains-nn-merged-filtered.csv', index=False)
