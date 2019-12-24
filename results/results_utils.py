import os
import sys
import re

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def get_stats(resdir, skip_first_frames_num=2, testfilename='test.npy', stepnums=(1,6,12)):
    res_tuples = []
    STEPNUMS = stepnums

    def mae_stepn(y_true, y_pred, stepn, errfunc, skip_first_frames_num=2):
        maes = []
        for y_true_sub, y_pred_sub in zip(y_true, y_pred):
            if stepn > 0:
                if y_true_sub.shape[1] >= stepn + skip_first_frames_num:
                    maes.append(errfunc(y_true_sub[:, skip_first_frames_num:skip_first_frames_num+stepn, :].flatten(), 
                                        y_pred_sub[:, skip_first_frames_num:skip_first_frames_num+stepn, :].flatten()))
            else:
                if y_true_sub.shape[1] > skip_first_frames_num:
                    maes.append(errfunc(y_true_sub[:, skip_first_frames_num:, :].flatten(), 
                                        y_pred_sub[:, skip_first_frames_num:, :].flatten()))
        return np.mean(maes)

    def print_log(root, res_tuples):
        fns = os.listdir(root)
        for fn in fns:
            fpath = os.path.join(root, fn)
            if os.path.isdir(fpath):
                print_log(fpath, res_tuples)
            elif fn == 'log.txt':
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                    confline = lines[3]
                    confpath = os.path.join(root, '../conf.yaml')
                    if os.path.exists(confpath):
                        with open(confpath, 'r') as conff:
                            conf = yaml.load(conff, Loader=yaml.FullLoader)
                            targetname, modelname = conf['noaa_target_features'], confpath.split('/')[-5]
                    else:
                        try:
                            targetname = re.search('noaa_target_features\=\'(.*)\'', confline)[1]
                        except:
                            targetname = re.search('\'noaa_target_features\': \'(.*)\', \'noaa_given_input_features\'', confline)[1]
                        try:
                            modelname = re.search('model\=\'(.*)\', noaa_given', confline)[1]
                        except:
                            modelname = re.search('\'model\': \'(.*)\', \'hidden_dim\'', confline)[1]
                testfilepath = os.path.join(root, '../test/{}'.format(testfilename))
                if os.path.exists(testfilepath):
                    testres = np.load(testfilepath).item()
                    model_target = testres['target']
                    model_output = testres['output']
                    res_tuple = [targetname, modelname, root]
                    for stepn in STEPNUMS:
                        error = mae_stepn(model_target, model_output, stepn=stepn, errfunc=mean_absolute_error, skip_first_frames_num=skip_first_frames_num)
                        res_tuple.append(error)
                    res_tuple = tuple(res_tuple)
                    res_tuples.append(res_tuple)
                else:
                    res_tuple = [targetname, modelname, root]
                    for stepn in STEPNUMS:
                        res_tuple.append(np.nan)
                    res_tuples.append(res_tuple)

    print_log(resdir, res_tuples)
    res_tuples = sorted(res_tuples, key=lambda x: (x[0], x[1], x[2]))
    indices = [x[0:3] for x in res_tuples]
    vals = np.array([x[3:] for x in res_tuples], dtype=np.float64)
    index = pd.MultiIndex.from_tuples(indices, names=['target', 'model', 'path'])
    res_df = pd.DataFrame(vals, index=index, columns=STEPNUMS)
    
    avg_df = res_df.groupby(['target', 'model']).mean().add_suffix('_mean')
    std_df = res_df.groupby(['target', 'model']).std().add_suffix('_std').fillna(0)
    avg_std_df = pd.concat((avg_df, std_df), axis=1)
    for stepi in STEPNUMS:
        avg_std_df['{}'.format(stepi)] = avg_std_df['{}_mean'.format(stepi)].apply(lambda x: '{:.04f}±'.format(x)) + avg_std_df['{}_std'.format(stepi)].apply(lambda x: '{:.04f}'.format(x))
        avg_std_df.pop('{}_mean'.format(stepi))
        avg_std_df.pop('{}_std'.format(stepi))
    return res_df, avg_std_df