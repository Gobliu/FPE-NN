import sys
import os

import numpy as np
from scipy import signal
# from keras import callbacks, backend, losses
import matplotlib.pyplot as plt

from NonGridModules.PDM_NG import PDM_NG
from NonGridModules.PxtData_NG import PxtData_NG
from NonGridModules.FPLeastSquare_NG import FPLeastSquare_NG
from NonGridModules.FPENet_NG import FPENet_NG
from NonGridModules.Loss import Loss

import OU_config as config
# import B_config as config
import Boltz_config as config

from GridModules.GaussianSmooth import GaussianSmooth

np.set_printoptions(suppress=True)

true_pxt = np.load('/GitHub/IFPE-Net/Pxt/OU/4_pxt_2015_sigma0.5.npy')
noisy_pxt = np.load('/GitHub/IFPE-Net/Pxt/OU/4_noisy_2015_sigma0.5.npy')
print(true_pxt.shape, noisy_pxt.shape)
print(true_pxt[0, 0, :])
print(true_pxt[-1, 0, :])
print(noisy_pxt[0, 0, :])
print(noisy_pxt[-1, 0, :])

# np.savez_compressed('./Pxt/Bessel_id{}_{}_sigma{}'.format(run_id, seed, sigma), x=x[:range_], t=t,
#                         true_pxt=f_true_pxt[:, :, :range_], noisy_pxt=f_noisy_pxt[:, :, :range_])
np.savez_compressed('./Pxt/OU_id2015_sigma0.5', x=true_pxt[0, 0, :], true_pxt=true_pxt[:, 1:, :],
                    noisy_pxt=noisy_pxt[:, 1:, :])
