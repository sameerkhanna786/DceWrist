import os
import numpy as np
import matplotlib as pl
from scipy.io import loadmat
from compartmentmodels.compartmentmodels import TwoCXModel, TwoCUModel, CompartmentModel
from dce import dce_fit


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

# Threshold (rough)
# thres = 2000
# mask = np.where(dat[:, :, :, 0] > thres, 1, 0)

# Mask on all rois
mask = np.zeros(rois[0].shape)
for i, roi in enumerate(rois):
    mask = mask + roi

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

# choose model currently implemented:
model = CompartmentModel

par, res, suc, para_names = dce_fit(dat, mask, aif_mean, time, model=model)

# save
save_path = r'/Users/sameerkhanna/Desktop/Test/dce_wrist/results'
img = nib.Nifti1Image(par, np.eye(4))
img.to_filename(os.path.join(save_path, 'para_' + model.__name__ + '.nii'))
img = nib.Nifti1Image(res, np.eye(4))
img.to_filename(os.path.join(save_path, 'resi_' + model.__name__ + '.nii'))
img = nib.Nifti1Image(suc.astype(float), np.eye(4))
img.to_filename(os.path.join(save_path, 'succ_' + model.__name__ + '.nii'))


# save the data
df = pd.DataFrame(columns=['roi_name'] + para_names)

for roi_name in rois_name:

    row = []
    row.append(roi_name)

    c_3d_roi = rois[roi_name]
    ix, iy, iz = np.where(c_3d_roi > 0)

    for j, para_name in enumerate(para_names):
        tmp = par[ix, iy, iz, j]
        tmp = tmp[np.isfinite(tmp)]
        row.append(np.mean(tmp))

    df.loc[i] = row

df.to_csv('test.csv', index=False)





