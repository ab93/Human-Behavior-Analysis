import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import csv
from scipy import stats
import csv

ANNOT_FILE_PATH = '../../Assignment_1/sentimentAnnotations_rev_v03.xlsx'

def get_data(feature_file, feature_name):

    excel_df = pd.read_excel(open(ANNOT_FILE_PATH,'rb'), sheetname='Sheet1')

    excel_df = pd.read_excel(open('../sentimentAnnotations_rev_v03.xlsx','rb'),
                            sheetname='Sheet1')

    excel_df = excel_df['majority vote']
    feat_df = pd.read_csv(feature_file)
    feat_df = feat_df[feature_name]
    return feat_df, excel_df



def plot_histogram(x, y, feature_name):
    classes = np.unique(y.values)[:-1]
    classes = map(np.int,classes)

    z = pd.concat([x,y], axis=1)
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna()
    z = z.groupby(by='majority vote')

    data = []
    for key, item in z:
        temp_df = z.get_group(key)
        data.append(temp_df.iloc[:,0].values)

    plt.hist(data[2],normed=True)
    plt.show()


def calculate_stat_scores(x,y,feature_name):
    z = pd.concat([x,y], axis=1)
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna()
    z = z.groupby(by='majority vote')

    data = []
    for key, item in z:
        temp_df = z.get_group(key)
        data.append(temp_df.iloc[:,0].values)

    with open('../results/stats.dat', 'a+') as f:
        f.write(feature_name + '\t')
        for class_data in data:
            f.write(str(np.mean(class_data)) + '\t'
            + str(np.median(class_data)) + '\t'
            + str(np.std(class_data)) + '\t')
        f.write('\n')

def calculate_anova(x,y,feature_name):
    classes = np.unique(y.values)[:-1]
    classes = map(np.int,classes)

    z = pd.concat([x,y], axis=1)
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna()
    z = z.groupby(by='majority vote')

    data = []
    for key, item in z:
        temp_df = z.get_group(key)
        data.append(temp_df.iloc[:,0].values)

    f_val, p_val = stats.f_oneway(data[0], data[1], data[2])
    print p_val
    with open('../results/anova_std.dat', 'a+') as f:
        f.write(feature_name + '\t' + str(p_val) + '\n')

def calculate_t(x, y, feature_name):
    classes = np.unique(y.values)[:-1]
    classes = map(np.int,classes)

    z = pd.concat([x,y], axis=1)
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna()
    z = z.groupby(by='majority vote')

    data = []
    for key, item in z:
        temp_df = z.get_group(key)
        data.append(temp_df.iloc[:,0].values)

    print data[1]
    print data[2]
    ans_neg_neu=stats.ttest_ind(data[0], data[1], None, False)
    ans_neu_pos=stats.ttest_ind(data[1], data[2], None, False)
    ans_neg_pos=stats.ttest_ind(data[0], data[2], None, False)

    print ans_neg_neu, ans_neu_pos, ans_neg_pos
    with open('../results/t_test_results.dat','a') as f:
        writer=csv.writer(f, delimiter='\t')
        writer.writerow([feature_name, ans_neg_neu, ans_neu_pos, ans_neg_pos])


def plot_box(x,y,feature_name):
    classes = np.unique(y.values)[:-1]
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
    bp = ax.boxplot(data, notch=True, sym='+', vert=True, whis=1.5,
                    patch_artist=True)
    ax.set_xticklabels(['Negative','Neutral','Positive'])
    ax.set_ylabel(feature_name)
    #ax.set_ylabel('Mouth Openness')
    ax.set_xlabel('Class Label (Majority Vote)')

    plt.grid(axis='y',
        linestyle='--',
        which='major',
        color='black',
        alpha=0.25)

    colors = ['red', 'blue', 'green']
    for box,color in zip(bp['boxes'],colors):
        box.set(color='black', linewidth=0.5)
        box.set_facecolor(color)

    for whisker in bp['whiskers']:
        whisker.set(color='grey', linewidth=1.5, linestyle='--')

    for cap in bp['caps']:
        cap.set(color='black', linewidth=2)

    for median in bp['medians']:
        median.set(color='black', linewidth=3)

    for flier in bp['fliers']:
        flier.set(marker='o', color='green', alpha=0.7)
    fig.savefig('../plots/' + feature_name + '.png')


if __name__ == '__main__':
    feature_name = sys.argv[2]
    feature_file = sys.argv[1]
    feat_df, excel_df = get_data(feature_file, feature_name)

    plot_box(feat_df, excel_df, feature_name)
    calculate_anova(feat_df, excel_df, feature_name)
    calculate_stat_scores(feat_df, excel_df, feature_name)

    #plot_box(feat_df, excel_df, feature_name)
    calculate_anova(feat_df, excel_df, feature_name)
    #plot_histogram(feat_df, excel_df, feature_name)
    #calculate_stat_scores(feat_df, excel_df, feature_name)
    #calculate_t(feat_df, excel_df, feature_name)

    