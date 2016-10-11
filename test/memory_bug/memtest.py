#!/usr/bin/env python

from __future__ import division, print_function
import sys
import numpy as np
from numpy.linalg import pinv

import nibabel as nib
from sys import getsizeof
from resource import getrusage, RUSAGE_SELF
from os.path import join, normpath
#from memory_profiler import profile
#from memprof import memprof
sys.path.insert(0, normpath('../..'))

#@profile
#@memprof
def reg_filter(data, design, components, aggressive=False, mask=True):
    """Filter functional data by regressing out noise components.
       A replacement for `fsl_regfilt`.

    Parameters
    ----------
    data: rank 4 numpy array
        Input/output data file to be denoised organised (nt, nz, ny, nz) and c-contiguous
    design: rank 2 numpy array
        design (or melodic mixing) matrix organised (nt, nc) and c-contiguous
    denoise_indices: sequence of integers
        Indices of the components that should be regressed out
    aggressive: bool
        Whether to do aggressive denoising
    mask: bool
        Whether to mask out low intensity voxels
    Returns
    -------
    None - in-place update

    Warning
    -------
    For reasons of memory efficiency the input data array is modified in-place
    """

    nt, nz, ny, nx = data.shape
    ntd, nc = design.shape
    components = sorted(set(components))

    assert ntd == nt
    assert data.flags['C_CONTIGUOUS']

    assert components and all(0 <= i < nc for i in components)

    # mask out background voxels in-place at less then 1%
    if mask:
        mean_image = data.mean(axis=0)
        min_, max_ = mean_image.min(), mean_image.max()
        mask = mean_image > (min_ + (max_ - min_) / 100)
        data *= mask[None, :]

    # flatten image volumes so we can treat as ordinary matrix
    data = data.reshape((nt, -1))

    # de-mean data
    data_means = data.mean(axis=0)
    data -= data_means

    # de-mean model
    design = design - design.mean(axis=0)

    # noise components of design
    noise_design = design[:, components]
    
    # filter data in place
    if aggressive:
        data -= noise_design.dot(pinv(noise_design).dot(data))
    else:
        data -= noise_design.dot(pinv(design).dot(data)[components])

    # re-mean data
    data += data_means


def main():
    #print(getrusage(RUSAGE_SELF).ru_maxrss)
    #
    # Caller is holding reference to data so it won't be deallocated
    # masking should write back in place
    # reshape is probably just a view
    # demeaning is another copy unless we use data[:] = 
    # filtered data is another copy
    # also temporaries
    #
    #

    infile = join('MAC-M006_TASK_preproc.feat', 'filtered_func_data.nii.gz')
    outfile = 'ica_filter_out.nii.gz'
    mixfile = join('MAC-M006_TASK_preproc.feat', 'ICA_AROMA', 'melodic.ica', 'melodic_mix')
    mix = np.loadtxt(mixfile)
    indicesfile = join('MAC-M006_TASK_preproc.feat', 'ICA_AROMA', 'classified_motion_ICs.txt')
    denoise_indices = list(np.loadtxt(indicesfile, dtype=int, delimiter=',') - 1)

    #print(len(denoise_indices))

    nii = nib.load(infile)
    #print(nii.shape)

    #print('After load::', getrusage(RUSAGE_SELF).ru_maxrss)

    #data = nii.get_data().T

    #print('Original data shape:', data.shape, ', bytes = %d' % data.nbytes)
    #print('After get_data::', getrusage(RUSAGE_SELF).ru_maxrss)

    #filtered_data = reg_filter(data, design=mix, components=denoise_indices)
    reg_filter(nii.get_data().T, design=mix, components=denoise_indices)
    #print(np.allclose(filtered_data, data))

    #print(np.may_share_memory(data, filtered_data))
    #print(np.shares_memory(data, filtered_data))

    #print(np.may_share_memory(nii.get_data(), filtered_data))
    #print(np.shares_memory(nii.get_data(), filtered_data))
    #nii.get_data()[:] = filtered_data.T

   # print(np.may_share_memory(nii.get_data(), filtered_data))

    #print(np.allclose(nii.get_data(), filtered_data.T))
    #print(np.shares_memory(nii.get_data(), filtered_data))

    #print('After reg_filter::', getrusage(RUSAGE_SELF).ru_maxrss)

    #print('Filtered data shape:', filtered_data.shape, ', bytes = %d' % data.nbytes)
    #del filtered_data
    #del data
    del nii

#
# may be an issue with writing the data back to the nibabel image. Are we sharing data? What happens whem we modify in place?
# does the data change in plcae in the nibabel image? When we write back are we trying to copy to ourselves or is new space allocated?
# if new space is allcoated then we have a twofold memory inflation at that point undel we release 'data'.
#
# looks ok, we'll want a byte by byte comparison of normal and in-place version ...
# also retain or document normal version as in-place code is obscure.
main()

