import numpy as np
from compartmentmodels.compartmentmodels import TwoCXModel, TwoCUModel, CompartmentModel
from progressbar import ProgressBar, Percentage, Bar, ETA


def dce_fit(data, mask, aif, time, model=TwoCXModel, init_val={}):
    """Fit DCE value based on compartmentmodels

    Args:
        - data (np.array, float): 4d input data
        - mask (np.array): 3d mask data (=1, in mask)
        - aif (np.array): Arterial input function
        - model (object): of type CompartmentModel
        - init_val (dict): initial parameter values use for fit

    Returns:
        - (np.array): 4d parameters map (last dimension matches parameters of
        the model)
        - (np.array): residuals

    """

    # If initial values not defined, use following default value
    if not init_val:
        if model.__name__ == 'CompartmentModel':
            init_val = {"F": 50.0, "v": 12.0}
        if model.__name__ == 'TwoCUModel':
            init_val = {'Fp': 51.0, 'v': 11.2, 'PS': 4.9}
        if model.__name__ == 'TwoCXModel':
            init_val = {'Fp': 51.0, 'vp': 11.2, 'PS': 4.9, 'VE': 13.2}

    # Get voxels to fit from mask
    ix, iy, iz = np.where(mask[:, :, :] > 0)

    # Initialize model
    gp = model(time=time, curve=np.zeros(time.shape), aif=aif)

    # Initialize output
    results = np.zeros(data.shape[:3] + (len(init_val),))
    success = np.zeros(data.shape[:3], dtype=np.bool)
    residuals = np.zeros(data.shape[:3], dtype=np.float)

    # Init progress bar
    pbar = ProgressBar(widgets=[Percentage(), Bar(), ' ', ETA()], maxval=len(ix),
                       redirect_stdout=False).start()
    bar_count = 0
    # loop
    for i in range(len(ix)):
        gp.curve = data[ix[i], iy[i], iz[i], :]
        gp.fit_model(init_val, fft=False)

        # save fit success
        success[ix[i], iy[i], iz[i]] = gp._fitted

        # get parameters
        res = gp.get_parameters()
        tmp = [res[x] for x in init_val.keys()]
        results[ix[i], iy[i], iz[i], :] = np.array(tmp)
        residuals[ix[i], iy[i], iz[i]] = gp._calc_residuals(gp._parameters, gp.curve)

        # update progress bar
        pbar.update(bar_count+1)
        bar_count += 1

    return results, residuals, success, list(init_val.keys())