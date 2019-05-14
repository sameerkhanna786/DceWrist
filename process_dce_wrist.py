import numpy as np
import matplotlib as pl
from scipy.io import loadmat
from compartmentmodels.compartmentmodels import TwoCXModel, TwoCUModel, CompartmentModel
from dce import dce_fit


# Data example path
data_path = r'/Users/sameerkhanna/Desktop/Test/dce_wrist/data/wrist_pt6/Baseline/Tresol5s_Registered.mat'
roi_path =  r'/Users/sameerkhanna/Desktop/Test/dce_wrist/data/wrist_pt6/Baseline/sixROIs.mat'


# load data
dat_dict = loadmat(data_path)
dat = dat_dict['RegI']
# data size
sx, sy, sz, st = dat.shape

# load roi
roi_dict = loadmat(roi_path)
rois = roi_dict['masks'][0, 0]
rois_name = list(rois.dtype.names)

# Threshold (rough)
thres = 2000
mask = np.where(dat[:, :, :, 0] > thres, 1, 0)

# time
dt = 5.
time = np.arange(st) * dt

# extract baseline
n0 = 10
dat0 = np.mean(dat[:, :, :, :n0], axis=3)

# remove baseline
dat = dat - dat0[..., None]

# Measure vascular input function (we consider here aif = vei)
#TODO: AIF definition impact, could be refined ?
vei_roi = rois['VEI']
ix, iy, iz = np.where(vei_roi == 1)
aif_mean = np.mean(dat[ix, iy, iz, :], axis=0)

# choose model (currently implemented:
model = CompartmentModel

par, res, suc = dce_fit(dat, mask, aif_mean, time, model=model)





