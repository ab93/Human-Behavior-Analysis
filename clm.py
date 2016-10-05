import pandas as pd
import os,re

headers=['frame no']
filelist={}
p=re.compile('.*_CLM_output.txt')
for (path,dir,files) in os.walk('/Users/Indhu/Downloads/Homework_1/features/VisualFeatures'):
    for each in files:
        if p.match(each):
            filelist[each[0:each.index('(')+1]]=path+'/'+each
print filelist

for i in range(0,66):
    headers.append(str(i)+"x")
    headers.append(str(i)+"y")
headers.append("uncertainity")

headers_ac=[]
filelist_ac={}
for (path,dir,files) in os.walk('/Users/Indhu/Downloads/Homework_1/features/AcousticFeatures'):
    for each in files:
        filelist_ac[each[0:each.index('(')+1]]=path+"/"+each
print filelist_ac

for i in range(7):
    headers_ac.append(str(i))
print headers_ac

#print data
table = pd.read_excel(open('sentimentAnnotations_rev_v03.xlsx','rb'),sheetname="Sheet1")
#print len(table.index)
cur = 0
changev = 0

vi_mean = open('clm_mean.csv', 'wb')
vi_mean.write("video no")
vi_mean.write(",")
for head in headers[1:-1]:
    vi_mean.write(head)
    vi_mean.write(",")
vi_mean.write("polarity")
vi_mean.write("\n")

ac_mean = open('acou_mean.csv', 'wb')
ac_mean.write("video no")
ac_mean.write(",")
for head in headers_ac:
    ac_mean.write(head)
    ac_mean.write(",")
ac_mean.write("polarity")
ac_mean.write("\n")

vi_std = open('clm_std.csv', 'wb')
vi_std.write("video no")
vi_std.write(",")
for head in headers[1:-1]:
    vi_std.write(head)
    vi_std.write(",")
vi_std.write("polarity")
vi_std.write("\n")

ac_std = open('acou_std.csv', 'wb')
ac_std.write("video no")
ac_std.write(",")
for head in headers_ac:
    ac_std.write(head)
    ac_std.write(",")
ac_std.write("polarity")
ac_std.write("\n")

for i in range(0,len(table.index)):
    video = table.iloc[i]['video']
    if changev != video:
        vidname='video'+str(int(video))+'('
        data = pd.read_csv(filelist[vidname], sep=' ', header=None)
        data.columns = headers
        vidname_ac='video'+str(int(video))+'('
        data_ac = pd.read_csv(filelist_ac[vidname_ac], sep=',', header=None)
        data_ac.columns = headers_ac
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
    for head in headers[1:-1]:
        #print temp[head].mean()
        vi_mean.write(str(temp[head].mean()))
        vi_mean.write(",")
        vi_std.write(str(temp[head].std()))
        vi_std.write(",")
    vi_mean.write(str(pol))
    vi_mean.write("\n")
    vi_std.write(str(pol))
    vi_std.write("\n")
    temp = data_ac[findex_ac:lindex_ac]
    ac_mean.write(str(int(video)))
    ac_mean.write(",")
    ac_std.write(str(int(video)))
    ac_std.write(",")
    for head in headers_ac:
        # print temp[head].mean()
        ac_mean.write(str(temp[head].mean()))
        ac_mean.write(",")
        ac_std.write(str(temp[head].std()))
        ac_std.write(",")
    ac_mean.write(str(pol))
    ac_mean.write("\n")
    ac_std.write(str(pol))
    ac_std.write("\n")
vi_mean.close()
ac_mean.close()
vi_std.close()
ac_std.close()