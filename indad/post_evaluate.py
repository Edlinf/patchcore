import numpy as np
import os
import torch
import shutil
def post_evaluate(path, thres=5):
    # save_path = '/disk/8T/xuhy/Projects/ind_knn_ad-master/results/results2'
    save_path = '/disk/8T/xuyy/anomaly/results/resultgpu50ms'
    avg_score = [[] for _ in range(3)] #0:good, 1:small, 2:large
    accs = [[] for _ in range(4)] #0:good, 1:small, 2:large, 3:all
    g = [[] for _ in range(2)] #0:score list, 1:correct
    s = [[] for _ in range(2)] #0:score list, 1:correct
    l = [[] for _ in range(2)] #0:score list, 1:correct

    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [item.replace('\n', '') for item in lines]
    for line in lines:
        type = line.split('_')[0]
        score = float(line.split('_')[-1])
        # old_path = os.path.join('/disk/8T/xuhy/Projects/ind_knn_ad-master/datasets/defect2/test', type, line.split('_')[1])
        old_path = os.path.join('/disk/8T/xuyy/anomaly/datasets/defect2/test', type, line.split('_')[1])
        if type=='good':
            g[0].append(score)
            if score>thres:
                g[1].append(0)
                new_path = os.path.join(save_path, type, 'Wrong', str(score)+'_'+line.split('_')[1])
                shutil.copy(old_path, new_path)
            else:
                g[1].append(1)
                new_path = os.path.join(save_path, type, 'True', str(score)+'_'+line.split('_')[1])
                shutil.copy(old_path, new_path)
        elif type=='small':
            s[0].append(score)
            if score<=thres:
                s[1].append(0)
                new_path = os.path.join(save_path, type, 'Wrong', str(score)+'_'+line.split('_')[1])
                shutil.copy(old_path, new_path)
            else:
                s[1].append(1)
                new_path = os.path.join(save_path, type, 'True', str(score)+'_'+line.split('_')[1])
                shutil.copy(old_path, new_path)
        elif type=='large':
            l[0].append(score)
            if score<=thres:
                l[1].append(0)
                new_path = os.path.join(save_path, type, 'Wrong', str(score)+'_'+line.split('_')[1])
                shutil.copy(old_path, new_path)
            else:
                l[1].append(1)
                new_path = os.path.join(save_path, type, 'True', str(score)+'_'+line.split('_')[1])
                shutil.copy(old_path, new_path)

    avg_score[0].append(sum(g[0])/len(g[0]))
    avg_score[1].append(sum(s[0])/len(s[0]))
    avg_score[2].append(sum(l[0])/len(l[0]))
    accs[0].append(sum(g[1])/len(g[1]))
    accs[1].append(sum(s[1])/len(s[1]))
    accs[2].append(sum(l[1])/len(l[1]))
    accs[3].append((sum(g[1])+sum(s[1])+sum(l[1]))/(len(g[1])+len(s[1])+len(l[1])))
    print(avg_score)
    print(accs)

    print(max(g[0]))
    print(min(s[0]))
    print(s[0])




if __name__=='__main__':
    post_evaluate('/disk/8T/xuyy/anomaly/results/results_PaDim_xception.txt')
    # post_evaluate('../results/results_PaDim_resnet18.txt')/disk/8T/xuyy/anomaly/results/results_PaDim_resnet18.txt
    # post_evaluate('../results/patchcore_02_04_2022_19_39_34.txt')