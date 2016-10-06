import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from scipy import stats

def get_data(feature_file, feature_name):
    excel_df = pd.read_excel(open('../../Assignment_1/sentimentAnnotations_rev_v03.xlsx','rb'),
                            sheetname='Sheet1')
    excel_df = excel_df['majority vote']
    feat_df = pd.read_csv(feature_file)
    feat_df = feat_df[feature_name]
    return feat_df, excel_df

def calculate_anova(x,y,feature_name):
    classes = np.unique(y.values)[:-1]
    print classes
    classes = map(np.int,classes)

    z = pd.concat([x,y], axis=1)
    z = z.replace([np.inf, -np.inf], np.nan)
    z = pd.concat([x,y], axis=1)
    z = z.groupby(by='majority vote')

    data = []
    for key, item in z:
        temp_df = z.get_group(key)
        data.append(temp_df.iloc[:,0].values)

    f_val, p_val = stats.f_oneway(data[0], data[1], data[2])
    print p_val
    with open('../results/anova.dat', 'a+') as f:
        f.write(feature_name + '\t' + str(p_val) + '\n')


def plot_box(x,y,feature_name):
    classes = np.unique(y.values)[:-1]
    print classes
    classes = map(np.int,classes)

    z = pd.concat([x,y], axis=1)
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna()
    z = z.groupby(by='majority vote')

    data = []

    for key, item in z:
        temp_df = z.get_group(key)
        data.append(temp_df.iloc[:,0].values)

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data)
    ax.set_xticklabels(classes)
    ax.set_ylabel(feature_name)
    ax.set_xlabel('Majority Vote')

    for box in bp['boxes']:
        box.set(color='red', linewidth=2)

    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=2)

    for cap in bp['caps']:
        cap.set(color='black', linewidth=2)

    for median in bp['medians']:
        median.set(color='blue', linewidth=2)

    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    fig.savefig('../plots/' + feature_name + '.png')


if __name__ == '__main__':
    feature_name = sys.argv[2]
    feature_file = sys.argv[1]
    feat_df, excel_df = get_data(feature_file, feature_name)
    plot_box(feat_df, excel_df, feature_name)
    calculate_anova(feat_df, excel_df, feature_name)
