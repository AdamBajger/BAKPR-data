import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from Utils import extract_parameter_value_as_int

## agg backend is used to create plot as a .png file
mpl.use('agg')

def print_nice_scatterplot(data: pd.DataFrame,
                           graph_filename: str,
                           col_examined:str,
                           col_grouped_by:str,
                           col_related:str,
                           title:str,
                           legend_title:str,
                           x_title: str,
                           y_title: str,
                           max_val=None,
                           min_val=None):

    # preset configuration
    scale=2
    max_marker_size = 1000*scale
    min_marker_size = 1*scale

    if max_val is None:
        max_val = data[col_examined].max()
    if min_val is None:
        min_val = data[col_examined].min()
    tick = (max_val - min_val) / 40
    y_ticks = np.concatenate([ np.arange(0, min_val-tick, -tick)[::-1], np.arange(0, max_val, tick)])


    # Scatterplot create figure
    fig = plt.figure( figsize=(8*scale,60*scale))

    # Create an axes instance
    ax1 = fig.add_subplot(111)
    ax1.set_title(title,
                 fontsize=25*scale)
    ax1.set_xlabel(x_title, fontsize=25*scale)
    ax1.set_ylabel(y_title, rotation=90, fontsize=25*scale)
    # this sorts times and labels for display in the boxplot by the parameters of the boxplots
    #data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr,labels), key=lambda e: e[1] ))


    groups = []
    # get the dataframes with their group names into list
    for group, group_df in data.groupby(col_grouped_by):
        groups.append((group, group_df))

    # sort the list by the parameter so we can apply reasonable coloring
    groups = sorted(groups, key=lambda x: x[0])
    current_size = max_marker_size
    # use seaborn to generate list of enough colors from a color pallete - it is graded
    colors=sns.color_palette(sns.dark_palette('cyan', n_colors=len(groups)), n_colors=len(groups))
    for group, group_df in groups:
        # Create the scatterplot
        ax1.scatter(x=group_df[col_related], y=group_df[col_examined], label=str(group)+' % ', color=colors.pop(), s=current_size)
        current_size -= (max_marker_size-min_marker_size)/len(groups)

    #ax1.set_xticklabels(['1', '2', '5', '10', '50', '100', '500', '200'])
    ax1.set_yticks(y_ticks)
    ax1.tick_params(axis='x', labelsize=22*scale)
    ax1.tick_params(axis='y', labelsize=22*scale)
    #ax1.grid(True)
    legend = plt.legend(loc="upper center", title=legend_title, ncol=2, prop={'size': 16*scale})
    legend.get_title().set_fontsize(22*scale)
    fig.savefig(graph_filename, bbox_inches="tight")

# read dataset
dataset = pd.read_csv("gains-filteredClassifiersAndDatasets.csv", ",")



dataset['od_params'] = dataset['od_params'].map(extract_parameter_value_as_int)
dataset = dataset.sort_values('od_params')
dataset['od_params'] = dataset['od_params'].map(str)
print_nice_scatterplot(data=dataset,
                       graph_filename='IsoForest-accuracy-groupedByRemoved.png',
                       col_examined="gain",
                       col_related="od_params",
                       col_grouped_by = "removed",
                       x_title='n_estimators parameter value ',
                       y_title='Increase in gain after OD',
                       #y_ticks=np.arange(-0.26, 0.26, 0.01),
                       title="Changes in accuracy based on % of removed outliers\n for different parameter n_estimators of OD method IsolationForest\n",
                       legend_title="% of removed outliers")


dataset = dataset.sort_values('removed')
dataset['removed'] = dataset['removed'].map(str)
print_nice_scatterplot(data=dataset,
                       graph_filename='IsoForest-accuracy-groupedByODParam.png',
                       col_examined="gain",
                       col_related="removed",
                       col_grouped_by = "od_params",
                       x_title='% of removed outliers',
                       y_title='Increase in gain after OD',
                       #y_ticks=np.arange(-0.26, 0.26, 0.01),
                       title="Changes in accuracy based on parameter n_estimators \n of OD method IsolationForest for different % of removed outliers\n",
                       legend_title="n_estimators parameter value")
