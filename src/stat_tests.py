import pandas as pd
from scipy import stats

m = []
for i in range(4):
    m.append(pd.read_csv(open('../results/scores/SVM_linear_hold_Exp_1_'+str(i+1)+'.txt','rb'),sep = "\t",header=None))

a = []
for i in range(4):
    a.append(pd.read_csv(open('../results/scores/SVM_linear_hold_Exp_2a_'+str(i+1)+'.txt','rb'),sep = "\t",header = None))

v = []
for i in range(4):
    v.append(pd.read_csv(open('../results/scores/SVM_linear_hold_Exp_2b_'+str(i+1)+'.txt','rb'),sep = "\t",header=None))

v_df = pd.concat([v[0],v[1],v[2],v[3]], axis=0)
a_df = pd.concat([a[0],a[1],a[2],a[3]], axis=0)
m_df = pd.concat([m[0],m[1],m[2],m[3]], axis=0)
print v_df

def acou_visual_t_test():
    print "Visual vs Acoustic ",t_test_values(v_df.iloc[:,1].values,a_df.iloc[:,1].values)
    print "Multimodal vs Acoustic ", t_test_values(m_df.iloc[:, 1].values, a_df.iloc[:, 1].values)
    print "Multomodal vs Visual ", t_test_values(m_df.iloc[:, 1].values, v_df.iloc[:, 1].values)
#print m[0]

def t_test_values(pred1,pred2):

    temp = stats.ttest_ind(pred1,pred2,None ,True)

    return temp[1]

# def acou_visual_t_test():
#     acou_vis_res,acou_mul_res,vis_mul_res = {},{},{}
#     acou_vis,acou_mul,vis_mul = [],[],[]
#     acou_total,vis_total,mul_total = [],[],[]
#     print a
#     for i in range(4):
#         acou_vis.append(t_test_values(a[i].iloc[:,1].values,v[i].iloc[:,1].values))
#         acou_mul.append(t_test_values(a[i].iloc[:,1].values,m[i].iloc[:,1].values))
#         vis_mul.append(t_test_values(v[i].iloc[:,1].values,m[i].iloc[:,1].values))
#         #print acou_total,a[i].iloc[:,1].values
#         acou_total.extend(a[i].iloc[:,1].values)
#         vis_total.extend( v[i].iloc[:, 1].values)
#         mul_total.extend( m[i].iloc[:,1].values)
#
#
#     acou_vis_res['4 exp'] = acou_vis
#     acou_mul_res['4 exp'] = acou_mul
#     vis_mul_res['4 exp'] = vis_mul
#
#     acou_vis_res['mean'] = sum(acou_vis)/float(len(acou_vis))
#     acou_mul_res['mean'] = sum(acou_mul)/float(len(acou_mul))
#     vis_mul_res['mean'] = sum(vis_mul)/float(len(vis_mul))
#
#     acou_vis_res['total'] = t_test_values(acou_total,vis_total)
#     acou_mul_res['total'] = t_test_values(acou_total,mul_total)
#     vis_mul_res['total'] = t_test_values(vis_total,mul_total)
#
#     print acou_vis_res
#     print acou_mul_res
#     print vis_mul_res
#

acou_visual_t_test()