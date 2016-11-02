import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

ANNOT_FILE_PATH = '../data/sentimentAnnotations_rev_v03.xlsx'
ACOUSTIC_FILE_PATH = '../results/acou_mean.csv'
VISUAL_FILE_PATH = '../results/okao_mean.csv'

def split_data(exp=3):
    df = pd.read_excel(ANNOT_FILE_PATH, sheetname='Sheet1')
    split = [None] * 4
    split[0] = np.arange(67)
    split[1] = np.arange(67,140)
    split[2] = np.arange(140,210)
    split[3] = np.arange(210,280)
    if exp == 0:
        return np.append(split[0],split[1]), split[2], split[3]
    elif exp == 1:
        return np.append(split[0],split[3]), split[1], split[2]
    elif exp == 2:
        return np.append(split[2],split[3]), split[0], split[1]
    else:
        return np.append(split[1],split[2]), split[3], split[0]


def fuse_features():
    labels_df = pd.read_excel(ANNOT_FILE_PATH, sheetname='Sheet1')
    labels_df = labels_df['majority vote']

    acou_feat_df = pd.read_csv(ACOUSTIC_FILE_PATH)
    #acou_feat_df = acou_feat_df[['naq','frequency','energy']]

    acou_feat_df = acou_feat_df.replace([np.inf, -np.inf], np.nan)
    acou_feat_df = acou_feat_df.apply(lambda x: x.fillna(x.mean()),axis=0)
    acou_feat_df = acou_feat_df['frequency']

    visual_feat_df = pd.read_csv(VISUAL_FILE_PATH)
    #visual_feat_df = visual_feat_df[['face_up_down','mouth_open','smile_level']]
    visual_feat_df = visual_feat_df[['mouth_open']]

    return pd.concat([acou_feat_df, visual_feat_df, labels_df], axis=1)


def model():
    df = fuse_features()
    train_idx, val_idx, test_idx = split_data()

    train_features = df.iloc[train_idx,:-1].values
    train_labels = df.iloc[train_idx,-1].values

    val_features = df.iloc[val_idx,:-1].values
    val_labels = df.iloc[val_idx,-1].values

    test_features = df.iloc[test_idx,:-1].values
    test_labels = df.iloc[test_idx,-1].values

    #scaler = preprocessing.StandardScaler().fit(train_features)
    #x_train = scaler.transform(train_features)
    #x_val = scaler.transform(val_features)
    #print train_features[:,0].shape
    #raw_input()

    plt.figure(2, figsize=(8, 6))
    plt.scatter(train_features[:, 0], train_features[:, 1], c=train_labels, cmap=plt.cm.Paired)
    plt.xlabel('naq')
    plt.ylabel('mouth_open')

    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    #plt.show()

    clf = svm.SVC(C=1)
    #clf = svm.LinearSVC(C=0.1, dual=False, multi_class='ovr',)
    #clf = RandomForestClassifier()
    clf.fit(train_features, train_labels)
    print clf.predict(val_features)
    print val_labels
    #print train_labels

    print clf.score(val_features, val_labels)


if __name__ == '__main__':
    model()
