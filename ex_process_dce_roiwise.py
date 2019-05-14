#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as pl
from scipy.io import loadmat
from compartmentmodels.compartmentmodels import TwoCXModel, TwoCUModel, CompartmentModel, ModifiedTofts
# from dce import dce_fit
import pandas as pd


# Data example path
if os.path.isdir('/Users/sameerkhanna'):
    data_path = r'/Users/sameerkhanna/Desktop/Test/dce_wrist/data/wrist_pt6/Baseline/Tresol5s_Registered.mat'
    roi_path =  r'/Users/sameerkhanna/Desktop/Test/dce_wrist/data/wrist_pt6/Baseline/sixROIs.mat'


# load data
print('Loading data...')
dat_dict = loadmat(data_path)
dat = dat_dict['RegI']
# data size
sx, sy, sz, st = dat.shape

# load roi
roi_dict = loadmat(roi_path)
rois = roi_dict['masks'][0, 0]
rois_name = list(rois.dtype.names)
print('...done')

# time
dt = 5.
time = np.arange(st) * dt

# extract baseline
n0 = 10
dat0 = np.mean(dat[:, :, :, :n0], axis=3)

# remove baseline
dat = dat - dat0[..., None]
# Average signal in rois
curves = {}
for roi_name in rois_name:
    c_3d_roi = rois[roi_name]
    ix, iy, iz = np.where(c_3d_roi > 0)
    curves[roi_name] = np.mean(dat[ix, iy, iz, :], axis=0)

# Measure vascular input function (we consider here aif = vei)
#TODO: AIF definition impact, could be refined ?
vei_roi = rois['VEI']
ix, iy, iz = np.where(vei_roi == 1)
aif_mean = np.mean(dat[ix, iy, iz, :], axis=0)

# choose model currently implemented:
# model = CompartmentModel
curr_model = ModifiedTofts
initial_values = {'vp': .1, 've': 30.0, 'ktrans': .1}

# turn on interactive plot
pl.ion()
# Loop through rois
models = []
for roi in curves.keys():
    model = curr_model(time=time, curve=curves[roi], aif=aif_mean)
    model.fit_model(initial_values)
    models.append(model)
    pl.figure()
    pl.plot(model.time, model.curve, 'bo')
    pl.plot(model.time, model.fit, 'g-')
    pl.title("One compartment fit")
    pl.xlabel("time [s]")
    pl.ylabel("concentration [a.u.]")
    pl.legend(('simulated curve', 'model fit'))
    pl.title('ROI: {0}'.format(roi))
    pl.show()


# save the data
para_names = list(initial_values.keys())

df = pd.DataFrame(columns=['roi_name'] + para_names)
for i, roi_name in enumerate(rois_name):
    # init row
    row = []
    # append roi name
    row.append(roi_name)

    for j, para_name in enumerate(para_names):
        row.append(models[i].get_parameters()[para_name])
    df.loc[i] = row

df.to_csv('test1.csv', index=False)





