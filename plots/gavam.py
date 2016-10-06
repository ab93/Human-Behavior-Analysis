import pandas as pd
import os,re

filelist={}
p=re.compile('.*GAVAM_output.txt')
for (path,dir,files) in os.walk('/home/achyut/Desktop/535H/features/'):
    #print("path", path)
    for each in files:
        #print("file match", each)
        if p.match(each):
            #print("file match", each)

            filelist[each[0:each.index('(')+1]]=path+'/'+each
#print filelist


headers=[]
headers.append('frame no')
headers.append('time_process(milli)')
headers.append('Gavam(1)/Clm(2)')

headers.append('face_displacement_X_Axis')
headers.append('face_displacement_Y_Axis')
headers.append('face_displacement_Z_Axis')

headers.append('face_Angular_displacement_X_Axis')
headers.append('face_Angular_displacement_Y_Axis')
headers.append('face_Angular_displacement_Z_Axis')

headers.append('confidence')

'''for i in range(9):
    headers.append(str(i))'''
print headers

#print data
table = pd.read_excel(open('sentimentAnnotations_rev_v03.xlsx','rb'),sheetname="Sheet1")
#print len(table.index)
cur = 0
changev = 0

vi_mean = open('gavam_mean.csv', 'wb')
vi_mean.write("video no")
vi_mean.write(",")
for head in headers[1:]:
    vi_mean.write(head)
    vi_mean.write(",")
vi_mean.write("polarity")
vi_mean.write("\n")

vi_std = open('gavam_std.csv', 'wb')
vi_std.write("video no")
vi_std.write(",")
for head in headers[1:]:
    vi_std.write(head)
    vi_std.write(",")
vi_std.write("polarity")
vi_std.write("\n")
print headers
#print filelist
for i in range(0,len(table.index)):

    video = table.iloc[i]['video']

    if changev != video:
        vidname='video'+str(int(video))+'('
        data = pd.read_csv(filelist[vidname], sep=' ', header=None)
        data.columns = headers
        cur = 0
        changev=video

    sframe = table.iloc[i]['start frame']
    eframe = table.iloc[i]['end frame']
    stime = table.iloc[i]['start time (second)']
    etime = table.iloc[i]['end time (second)']
    pol = table.iloc[i]['majority vote']
    findex_ac = int(stime*100)
    lindex_ac = int(etime*100)
    findex = cur

    while sframe <= data.iloc[cur]['frame no'] and eframe >= data.iloc[cur]['frame no']:
        cur+=1
    
    lindex = cur
    temp = data[findex:lindex]
    vi_mean.write(str(int(video)))
    vi_mean.write(",")
    vi_std.write(str(int(video)))
    vi_std.write(",")
    
    for head in headers[1:]:
        #print temp[head].mean()
        vi_mean.write(str(temp[head].mean()))
        vi_mean.write(",")
        vi_std.write(str(temp[head].std()))
        vi_std.write(",")

    vi_mean.write(str(pol))
    vi_mean.write("\n")
    vi_std.write(str(pol))
    vi_std.write("\n")
    
vi_mean.close()
#ac_mean.close()
vi_std.close()
#ac_std.close()