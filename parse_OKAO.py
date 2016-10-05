import pandas as pd
import numpy as np
import sys
import os
import re


def get_indices():
    indices = ['rec_l', 'rec_t', 'rec_r', 'rec_d', 'rec_conf', 'face_pose']
    ffp = [['fp_x'+str(i), 'fp_y'+str(i), 'fp_conf'+str(i)] for i in range(38)]
    indices.extend([item for sublist in ffp for item in sublist])
    indices.extend(['face_up_down','face_left_right','face_roll','face_conf'])
    indices.extend(['gaze_up_down','gaze_left_right','gaze_conf'])
    indices.extend(['left_eye_open','left_eye_conf','right_eye_open',
                'right_eye_conf','mouth_open','mouth_conf', 'smile_level'])

    return indices

def parse(path):
    indices = get_indices()
    df = pd.read_csv(path, delimiter=" ", header=None, names=indices)
    return df
    #print df.columns

def get_id(id):
    if id < 10:
        return '0' + str(id)
    else:
        return str(id)

def generate_segment_features(measure='mean'):
    excel_df = pd.read_excel(open('../Assignment_1/sentimentAnnotations_rev_v03.xlsx','rb'),
                            sheetname='Sheet1')
    excel_df = excel_df[['video','start frame','end frame']]
    output_df = pd.DataFrame()
    func = np.mean
    if measure == 'std':
        func = np.std

    #for filename in os.listdir('../features/VisualFeatures/'):
    for vid_idx in range(1,49):
        seg_df = pd.DataFrame()
        #patt = re.compile(r'_okao_output')
        #m = patt.search(filename)
        filename = 'video' + get_id(vid_idx) + '_okao_output.txt'

        df = parse(os.path.join('../features/VisualFeatures/', filename))
        vid_num = int(filename.split('_')[0][-2:])
        vid_df = excel_df[excel_df['video'] == vid_num]

        print 'vid_num:',vid_num
        for index in range(len(vid_df)):
            start_frame = vid_df.iloc[index]['start frame']
            end_frame = vid_df.iloc[index]['end frame']
            seg_df = seg_df.append(df.iloc[start_frame-1:end_frame].dropna().drop('rec_l',1).apply(func),
                                                            ignore_index=True)

        seg_df.insert(0, 'video', pd.Series(map(int,[vid_num] * len(vid_df))))
        output_df = output_df.append(seg_df, ignore_index=True)

    indices = get_indices()
    indices.insert(0,'video')
    indices.remove('rec_l')
    output_df[indices].to_csv('okao_' + measure + '.csv',sep=",",index=False)

if __name__ == '__main__':
    generate_segment_features(measure='std')
    #parse('../features/VisualFeatures/video01_okao_output.txt')
