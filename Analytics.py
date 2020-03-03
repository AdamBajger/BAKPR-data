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
    _fig = plt.figure( figsize=(8*scale,40*scale))

    # Create an axes instance
    ax1 = _fig.add_subplot(111)
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
    legend = plt.legend(loc="lower center", title=legend_title, ncol=2, prop={'size': 16*scale})
    legend.get_title().set_fontsize(22*scale)
    _fig.savefig(graph_filename, bbox_inches="tight")
    plt.close(_fig)




"""
argument sort_func is a comparator function applied to a tuple of two elements: (data series, name). It sorts data in the graph.
"""
def print_boxplots(data: pd.DataFrame,
                   graph_filename: str,
                   col_examined: str,
                   col_related: str,
                   sort_func,
                   title: str,
                   x_title: str,
                   y_title: str,
                   min_val=None,
                   max_val=None
                   ):
    g = data.groupby([col_related])  # ["accuracy"].sum().reset_index()

    # graph parameters
    scale = 1
    show_fliers = True
    mean_color='b'
    mean_marker='o'


    labels = []
    data_to_plot_arr = []
    #switch = True


    for group, group_df in g:
        data_to_plot_arr.append(group_df[col_examined])
        labels.append(group)

    # dynamically set parameters of the graphs so that they are uniform across all graphs, but are minimalised
    figsize = ((len(g)) * scale, 25 * scale)  # originally (60, 30)
    if max_val is None:
        max_val = data[col_examined].max()
    if min_val is None:
        min_val = data[col_examined].min()
    tick = (max_val - min_val) / 40
    y_labels = np.concatenate([ np.arange(0, min_val-tick, -tick)[::-1], np.arange(0, max_val+6*tick, tick)])

    # Create a figure instance
    _fig = plt.figure( figsize=figsize)

    # Create an axes instance
    _ax = _fig.add_subplot(111)
    _ax.set_xlabel(col_related, fontsize=20*scale)
    # this sorts times and labels for display in the boxplot by the parameters of the boxplots
    data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr,labels), key=sort_func ))

    # Create the boxplot

    bp = _ax.boxplot(data_to_plot_arr, positions=[x for x in range(len(labels))], showfliers=show_fliers)
    # following function is described here: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    _ax.plot([x for x in range(len(labels))], list(map(lambda x: x.mean(), list(data_to_plot_arr))), marker=mean_marker, color=mean_color)
    _ax.set_title(title,
                  fontsize=25 * scale)
    _ax.set_xlabel(x_title, fontsize=25 * scale)
    _ax.set_ylabel(y_title, rotation=90, fontsize=25 * scale)
    _ax.set_xticklabels(labels, rotation=90)
    _ax.set_yticks(y_labels)
    _ax.tick_params(axis='x', labelsize=22*scale)
    _ax.tick_params(axis='y', labelsize=22*scale)

    # custom legend elements gymnastics (it is really awful, but I coudl not find better solution)
    colors = [mean_color]
    sizes = [6*scale]
    texts = ["Mean"]
    patches = [plt.plot([], [], marker=mean_marker, ms=sizes[i], ls="", mec=None, color=colors[i],
                        label="{:s}".format(texts[i]))[0] for i in range(len(texts))]

    legend = plt.legend(handles=patches,
                        bbox_to_anchor=[0.5, -0.12],
                        loc='center',
                        title="Boxplots show first and third quartile,\n with variability represented with whiskers",
                        ncol=2,
                        prop={'size': 16 * scale})
    legend.get_title().set_fontsize(16 * scale)
    _ax.grid(True)



    # Save the figure
    _fig.savefig(graph_filename+'.png', bbox_inches='tight')
    plt.close(_fig)



    # # this sorts times and labels for display in the boxplot by the max of the boxplots
    # data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr,labels), key=lambda e: e[0].max()))
    #
    # # Create a figure instance
    # fig2 = plt.figure( figsize=figsize)
    # # Create an axes instance
    # ax2 = fig2.add_subplot(111)
    # ax2.set_xlabel(col_related, fontsize=x_title_font_size)
    #
    # show_fliers = True
    # ax2.boxplot(data_to_plot_arr, showfliers=show_fliers)
    # ax2.set_xticklabels(labels, rotation=90)
    # ax2.set_yticks(y_labels)
    # ax2.tick_params(axis='x', labelsize=22)
    # ax2.grid(True)
    #
    # # Save the figure
    # fig2.savefig(graph_filename + 'boxplots-'  + col_related + '-maxSorted-' + ('fliers' if show_fliers else'') + '.png', bbox_inches='tight')
    #
    #
    #
    # # this sorts times and labels for display in the boxplot by the max of the boxplots
    # data_to_plot_arr, labels = zip(*sorted(zip(data_to_plot_arr,labels), key=lambda e: e[0].mean()))
    #
    # # Create a figure instance
    # fig3 = plt.figure( figsize=figsize)
    # # Create an axes instance
    # ax3 = fig3.add_subplot(111)
    # ax3.set_xlabel(col_related, fontsize=x_title_font_size)
    #
    # show_fliers = False
    # ax3.boxplot(data_to_plot_arr, showfliers=show_fliers)
    # ax3.set_xticklabels(labels, rotation=90)
    # ax3.set_yticks(np.arange(-0.03, 0.02, 0.001 ))
    # ax3.tick_params(axis='x', labelsize=22)
    # ax3.grid(True)
    #
    # # Save the figure
    # fig3.savefig(graph_filename + 'boxplots-'  + col_related + '-meanSorted-' + ('fliers' if show_fliers else'') + '.png', bbox_inches='tight')


# read dataset
df = pd.read_csv("gains-nn-merged.csv", ",")
#df = df.sort_values('removed')
df['removed'] = df['removed'].map(str)

df['od_params'] = df['od_params'].map(lambda x: extract_parameter_value_as_int(x, parameter="n_neighbors"))
#df = df.sort_values('od_params')
df['od_params'] = df['od_params'].map(str)

od_method_name = 'NearestNeighbors'

print_boxplots(data=df,
               graph_filename= od_method_name + '-boxplots-od_params',
               col_examined="gain",
               col_related = "od_params",
               sort_func=lambda e: -e[0].mean(),
               title="Accuracy of classifiers\n for different n_estimators parameter\nof OD method IsolationForest\nsorted on the mean values\n",
               x_title="parameter n_neighbors ",
               y_title="Increase in accuracy after applying OD")
#exit()

print_nice_scatterplot(data=df,
                       graph_filename=od_method_name + "-scatterplot-od_params",
                       col_examined="gain",
                       col_grouped_by="removed",
                       col_related="od_params",
                       title="Accuracy of classifiers\n for different n_neighbors parameter\nof OD method NearestNeighbors\nsorted on the mean values\n",
                       x_title="parameter n_neighbors",
                       y_title="change in gain after OD",
                       legend_title="% of removed outliers")


print_boxplots(data=df,
               graph_filename= od_method_name +'-boxplots-removed',
               col_examined="gain",
               col_related = "removed",
               sort_func=lambda e: -e[0].mean(),
               title="Accuracy of classifiers\n for different n_estimators parameter\nof OD method IsolationForest\nsorted on the mean values\n",
               x_title="% removed ",
               y_title="Increase in accuracy after applying OD")


df.sort_values(by="removed", inplace=True)
print_nice_scatterplot(data=df,
                       graph_filename=od_method_name + '-scatterplot-removed',
                       col_examined="gain",
                       col_grouped_by="od_params",
                       col_related="removed",
                       title="Accuracy of classifiers\n for different n_neighbors parameter\nof OD method NearestNeighbors\nsorted on the mean values\n",
                       x_title="% of removed outliers",
                       y_title="change in gain after OD",
                       legend_title="parameter n_neighbors")

exit()

# preset configuration
grouped_by = "clf"

# extract data
gbc = df.groupby(grouped_by)

for group, group_df in gbc:
    print_nice_scatterplot(data=group_df,
                           graph_filename='clf_od_params/IsoForest-scatter-od_params-'+group+'.png',
                           col_examined="gain",
                           col_related="od_params",
                           col_grouped_by="removed",
                           x_title='n_estimators value',
                           y_title='Increase in gain after OD',
                           title="Changes in accuracy of classifier "+group+"\nbased on parameter n_estimators \n of OD method IsolationForest for different % of removed outliers\n",
                           legend_title="% of removed outliers")