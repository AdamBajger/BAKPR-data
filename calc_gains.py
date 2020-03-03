import pandas as pd
import numpy as np

# set float formatting to show
pd.options.display.float_format = '{:0.5f}'.format

# read dataset
data = pd.read_csv("results-NN.csv", ",")
data_zero = pd.read_csv("results-NN.csv")
print(data_zero[['accuracy', 'removed']])
data_zero[['accuracy', 'removed']] = data_zero[['accuracy', 'removed']].apply(pd.to_numeric, errors='coerce', axis=1)
data_zero = data_zero[data_zero['removed'] == 0]
print(data_zero[['accuracy', 'removed']])
data[['accuracy', 'removed']] = data[['accuracy', 'removed']].apply(pd.to_numeric, errors='coerce', axis=1)
print(data[['removed', 'accuracy']])



# extract data
#g = data.groupby(["removed"])

#zero_removed_df = None
#positive_removed_dfs = []
#for group, group_df in g:
 #   print(group)
#    if group == 0.0:
#        zero_removed_df = group_df
#        break
    #else:
    #    positive_removed_dfs.append(group_df)





data = pd.merge(data, data_zero[['accuracy', 'dataset', 'clf', 'clf_family', ]], how="outer", on=['dataset', 'clf', 'clf_family'], suffixes=('', '_old'))
data['gain'] = data['accuracy'] - data['accuracy_old']
# map float values to fixed decimal places strings
for column_name in ['accuracy', 'od_time', 'clf_time', 'total_time', 'accuracy_old', 'gain']:
    data[column_name] = data[column_name].map('{:0.5f}'.format)
data.drop_duplicates(inplace=True)
data.to_csv('gains.csv', index=False)
