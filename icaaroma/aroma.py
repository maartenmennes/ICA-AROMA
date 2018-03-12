#!/usr/bin/env python
"""
aroma.py: filter fmri datasets based on ICA analysis.
"""
from __future__ import division, print_function

import sys
import os
from os.path import join, isfile, isdir, expanduser, dirname
import shutil
from functools import partial

from tempfile import mkdtemp
from subprocess import check_call, CalledProcessError

import logging
import argparse
import random

import numpy as np
from numpy.linalg import pinv
import nibabel as nib

__version__ = '0.4.0'

# FSL commands and environment (defaults are suitable for ubuntu)
FSLBINDIR = join(os.environ.get("FSLDIR", '/usr/share/fsl/5.0'), 'bin')
FSLTEMPLATEDIR = join(os.environ.get("FSLDIR", '/usr/share/fsl/5.0'), 'data', 'standard')

# MNI152 T1 2mm template file
FSLMNI52TEMPLATE = join(FSLTEMPLATEDIR, 'MNI152_T1_2mm_brain.nii.gz')

MELODIC   = join(FSLBINDIR, 'melodic')
FSLROI    = join(FSLBINDIR, 'fslroi')
FSLMERGE  = join(FSLBINDIR, 'fslmerge')
FSLMATHS  = join(FSLBINDIR, 'fslmaths')
FLIRT     = join(FSLBINDIR, 'flirt')
APPLYWARP = join(FSLBINDIR, 'applywarp')
BET       = join(FSLBINDIR, 'bet')


def is_writable_file(path):
    exists_and_writable = path and isfile(path) and os.access(path, os.W_OK)
    parent = path and (os.path.dirname(path) or os.getcwd())
    creatable = parent and os.access(parent, os.W_OK)
    return exists_and_writable or creatable


def is_writable_directory(path):
    return path and isdir(path) and os.access(path, os.W_OK)


def nifti_dims(filename):
    """Matrix dimensions of image in nifti file"""
    return tuple(nib.load(filename).header['dim'][1:5])


def nifti_pixdims(filename):
    """Pixel dimensions of image in nifti file"""
    return tuple(nib.load(filename).header['pixdim'][1:5])


def zsums(filename, masks=(None,)):
    """Sum of Z-values within the total Z-map or within a subset defined by a mask.

    Calculated via the mean and number of non-zero voxels.

    Parameters
    ----------
    filename: str
        zmap nifti file
    masks: Optional(sequence of str)
        mask files (None indicates no mask)

    Returns
    -------
    tuple of numpy arrays
        sums of pixels across the whole images or just within the mask
    """
    assert isfile(filename)

    img = np.abs(nib.load(filename).get_data())
    maskarrays = [
        nib.load(fname).get_data().astype('bool')[..., None] if fname is not None else np.ones_like(img)
        for fname in masks
    ]
    zsums = [(img * mask).sum(axis=(0, 1, 2)) for mask in maskarrays]

    return tuple(zsums)


def cross_correlation(a, b):
    """Cross Correlations between columns of two matrices"""
    assert a.ndim == b.ndim == 2
    _, ncols_a = a.shape
    # nb variables in columns rather than rows hence transpose
    # extract just the cross terms between cols in a and cols in b
    return np.corrcoef(a.T, b.T)[:ncols_a, ncols_a:]


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
    aggressive: bool, optional
        Whether to do aggressive denoising (default: False)
    mask: bool, optional
        Whether to mask out low intensity voxels (default: True)
    Returns
    -------
    None - in-place update

    Notes
    -------
    For reasons of memory efficiency the input data array is modified in-place
    """

    nt, nz, ny, nx = data.shape
    ntd, nc = design.shape
    components = sorted(set(components))

    assert ntd == nt
    assert data.flags['C_CONTIGUOUS']

    assert components and all(0 <= i < nc for i in components)

    # mask out background voxels at less then 1%
    if mask:
        mean_image = data.mean(axis=0)
        min_, max_ = mean_image.min(), mean_image.max()
        mask = mean_image > (min_ + (max_ - min_) / 100)
        data *= mask[None, :]

    # flattened view of image volumes so we can treat as an ordinary matrix
    data = data.reshape((nt, -1))

    # de-mean data
    data_means = data.mean(axis=0)
    data -= data_means

    # de-mean model
    design = design - design.mean(axis=0)

    # just the noise components of the design
    design_n = design[:, components]

    # filter noise components from data
    # D_n @ (pinv(D_n) @ data) or D_n @ ((pinv(D) @ data)_n)
    if aggressive:
        data -= design_n.dot(pinv(design_n).dot(data))
    else:
        data -= design_n.dot(pinv(design).dot(data)[components])

    # re-mean data
    data += data_means


def is_valid_melodic_dir(dirpath):
    """Check for all the files needed in melodic directory"""
    return (
        dirpath and
        isdir(dirpath) and
        isfile(join(dirpath, 'melodic_IC.nii.gz')) and
        isfile(join(dirpath, 'melodic_mix')) and
        isfile(join(dirpath, 'melodic_FTmix')) and
        isdir(join(dirpath, 'stats'))
    )


def is_valid_feat_dir(dirpath):
    """Check for all the files needed in feat directory"""
    return (
        dirpath and
        isdir(dirpath) and
        isfile(join(dirpath, 'filtered_func_data.nii.gz')) and
        isdir(join(dirpath, 'mc')) and
        isfile(join(dirpath, 'mc', 'prefiltered_func_data_mcf.par')) and
        isdir(join(dirpath, 'reg')) and
        isfile(join(dirpath, 'reg', 'example_func2highres.mat')) and
        isfile(join(dirpath, 'reg', 'highres2standard_warp.nii.gz'))
    )


def _find_aroma_dir(aromadir=None):
    """Find location of aroma directory with mask files"""
    locations = [
        '/usr/share/aroma',
        '/usr/local/share/aroma',
        join(expanduser('~'), '.local', 'share', 'aroma'),
        dirname(__file__),
        os.getcwd()
    ]
    if aromadir is not None:
        locations.insert(0, aromadir)

    testfile = 'mask_csf.nii.gz'

    for location in locations:
        if isdir(location) and isfile(join(location, testfile)):
            return location
    return None


def run_ica(infile, outfile, maskfile, t_r, ndims_ica=0, melodic_indir=None, seed=None):
    """Runs MELODIC and merges the mixture modelled thresholded ICs into a single 4D nifti file.

    Parameters
    ----------
    infile:  str
        fMRI data file (nii.gz) on which MELODIC should be run
    outdir:  str
        Output directory
    maskfile: str
        Mask file to be applied during MELODIC
    t_r: float
        Repetition Time (in seconds) of the fMRI data
    ndims_ica: int
        Dimensionality of ICA
    melodic_indir: str
        MELODIC directory in case it has been run before
    seed: Optional(unsigned int)
        Seed for RNG in melodic

    Returns
    -------
    tuple of numpy arrays
        mix, ftmix

    Output
    ------
    Merged file containing the mixture modelling thresholded Z-stat maps
    """
    assert isfile(maskfile)
    assert 0.5 < t_r < 10
    assert 0 <= ndims_ica < 100

    working_dir = mkdtemp(prefix='run_ica')

    if is_valid_melodic_dir(melodic_indir):
        for f in ['melodic_IC.nii.gz', 'melodic_mix', 'melodic_FTmix']:
            shutil.copy(join(melodic_indir, f), join(working_dir, f))
        shutil.copytree(join(melodic_indir, 'stats'), join(working_dir, 'stats'))
    else:
        cmdline = [
            MELODIC, '--in=%s' % infile, '--outdir=%s' % working_dir,
            '--mask=%s' % maskfile, '--dim=%d' % ndims_ica,
            '--Ostats', '--nobet', '--mmthresh=0.5', '--report', '--tr=%f' % t_r
        ]
        if seed is not None:
            cmdline.append('--seed=%u' % seed)
        check_call(cmdline)

    assert is_valid_melodic_dir(working_dir)

    melodic_ics_file   = join(working_dir, 'melodic_IC.nii.gz')
    melodic_ftmix_file = join(working_dir, 'melodic_FTmix')
    melodic_mix_file   = join(working_dir, 'melodic_mix')

    # Normally, there will be only one spatial map per file but if the mixture modelling did not converge
    # there will be two, the latter being the results from a simple null hypothesis test and the first one empty.
    # To handle this we'll get the last map from each file.
    # NB Files created by MELODIC are labelled with integers, base 1, no zero padding ...
    ncomponents = nifti_dims(melodic_ics_file)[3]
    zfiles_in  = [join(working_dir, 'stats', 'thresh_zstat%d.nii.gz' % i) for i in range(1, ncomponents+1)]
    zfiles_out = [join(working_dir, 'stats', 'thresh_zstat_fixed%d.nii.gz' % i) for i in range(1, ncomponents+1)]
    for zfile_in, zfile_out in zip(zfiles_in, zfiles_out):
        nmaps = nifti_dims(zfile_in)[3]  # will be 1 or 2
        #                      input, output, first frame (base 0), number of frames
        check_call([FSLROI, zfile_in, zfile_out, '%d' % (nmaps-1), '1'])

    # Merge all mixture modelled Z-maps within the output directory (NB: -t => concatenate in time)
    melodic_thr_file = join(working_dir, 'melodic_IC_thr.nii.gz')
    check_call([FSLMERGE, '-t', melodic_thr_file] + zfiles_out)

    # Apply the mask to the merged file (in case pre-run melodic was run with a different mask)
    check_call([FSLMATHS, melodic_thr_file, '-mas', maskfile, melodic_thr_file])

    # Outputs
    shutil.copyfile(melodic_thr_file, outfile)
    mix = np.loadtxt(melodic_mix_file)
    ftmix = np.loadtxt(melodic_ftmix_file)

    shutil.rmtree(working_dir)
    return mix, ftmix


def register_to_mni(infile, outfile, template=FSLMNI52TEMPLATE, affmat=None, warp=None):
    """Registers an image (or a time-series of images) to MNI152 T1 2mm.

    If no affmat is specified, it only warps (i.e. it assumes that the data has been registered to the
    structural scan associated with the warp-file already). If no warp is specified either, it only
    resamples the data to 2mm isotropic if needed (i.e. it assumes that the data has been registered
    to a MNI152 template). In case only an affmat file is specified, it assumes that the data has to be
    linearly registered to MNI152 (i.e. the user has a reason not to use non-linear registration on the data).

    Parameters
    ----------
    infile: str
        Input file (nii.gz) which is to be registered to MNI152 T1 2mm
    outfile: str
        Output file registered to MNI152 T1 2mm (.nii.gz)
    template: Optional(str)
        MNI52 template file
    affmat: str
        Mat file describing the linear registration to structural space (if image still in native space)
    warp: str
        Warp file describing the non-linear registration to MNI152 space (if image not yet in MNI space)

    Returns
    -------
    None

    Output
    ------
    File containing the mixture modelling thresholded Z-stat maps registered to 2mm MNI152 template
    """
    assert isfile(infile)
    assert is_writable_file(outfile)

    if affmat is None and warp is None:
        # No affmat- or warp-file specified, assume already in MNI152 space
        if np.allclose(nifti_pixdims(infile)[:3], [2.0, 2.0, 2.0]):
            shutil.copyfile(src=infile, dst=outfile)
        else:
            # Resample to 2mm if need be
            check_call([
                FLIRT, '-ref', template, '-in', infile, '-out', outfile,
                '-applyisoxfm', '2', '-interp', 'trilinear'
            ])
    elif warp is not None and affmat is None:
        # Only a warp-file, assume already registered to structural, apply warp only
        check_call([
            APPLYWARP, '--ref=%s' % template, '--in=%s' % infile, '--out=%s' % outfile,
            '--warp=%s' % warp, '--interp=trilinear'
        ])
    elif affmat is not None and warp is None:
        # Only a affmat-file, perform affine registration to MNI
        check_call([
            FLIRT, '-ref', template, '-in', infile, '-out', outfile,
            '-applyxfm', '-init', affmat, '-interp', 'trilinear'
        ])
    else:
        # Both an affmat and a warp file specified, apply both
        check_call([
            APPLYWARP, '--ref=%s' % template, '--in=%s' % infile, '--out=%s' % outfile,
            '--warp=%s' % warp, '--premat=%s' % affmat, '--interp=trilinear'
        ])


def feature_time_series(mix, rparams, seed=None):
    """Maximum 'Realignment' parameters correlation feature scores.

    Determines the maximum robust correlation of each component time-series with
    a model of 72 realignment parameters.

    Parameters
    ----------
    mix: rank 2 numpy array
        Melodic_mix array
    rparams: rank 2 nump array
        Realignment parameters (n rows of 6 parameters)
    seed: Optional(int)
        Random number generator seed for python random module
    Returns
    -------
    rank 1 numpy.array
        Maximum RP correlation feature scores for the components of the melodic_mix file
    """
    assert mix.ndim == rparams.ndim == 2

    _, nparams = rparams.shape

    if seed is not None:
        random.seed(seed)

    # RP model including the RPs, their derivatives, and time shifted versions of each
    rp_derivs = np.vstack((
        np.zeros(nparams),
        np.diff(rparams, axis=0)
    ))
    rp12 = np.hstack((rparams, rp_derivs))
    rp12_1fw = np.vstack((
        np.zeros(2*nparams),
        rp12[:-1]
    ))
    rp12_1bw = np.vstack((
        rp12[1:],
        np.zeros(2*nparams)
    ))
    rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

    # Determine the maximum correlation between RPs and IC time-series
    nsplits = 1000
    nmixrows, nmixcols = mix.shape
    nrows_to_choose = int(round(0.9 * nmixrows))

    # Max correlations for multiple splits of the dataset (for a robust estimate)
    max_correls = np.empty((nsplits, nmixcols))
    for i in range(nsplits):
        # Select a random subset of 90% of the dataset rows (*without* replacement)
        chosen_rows = random.sample(population=range(nmixrows), k=nrows_to_choose)

        # Combined correlations between RP and IC time-series, squared and non squared
        correl_nonsquared = cross_correlation(mix[chosen_rows], rp_model[chosen_rows])
        correl_squared = cross_correlation(mix[chosen_rows]**2, rp_model[chosen_rows]**2)
        correl_both = np.hstack((correl_squared, correl_nonsquared))

        # Maximum absolute temporal correlation for every IC
        max_correls[i] = np.abs(correl_both).max(axis=1)

    # Feature score is the mean of the maximum correlation over all the random splits
    # Avoid propagating occasional nans that arise in artificial test cases
    return np.nanmean(max_correls, axis=0)


def feature_frequency(ftmix, t_r):
    """High-frequency content feature scores.

    It determines the frequency, as fraction of the Nyquist frequency, at which the higher and lower
    frequencies explain half of the total power between 0.01Hz and Nyquist.

    Parameters
    ----------
    ftmix: rank 2 numpy array
        melodic ft mix array
    t_r: float
        Repetition time (in seconds) of the fMRI data

    Returns
    -------
    rank 1 numpy.array
        HFC ('High-frequency content') feature scores for the components of the melodic_FTmix file
    """
    assert ftmix.ndim == 2
    assert 0.5 < t_r < 10

    sample_frequency = 1 / t_r
    nyquist = sample_frequency / 2

    # Determine which frequencies are associated with every row in the melodic_FTmix file
    # (assuming the rows range from 0Hz to Nyquist)
    # TODO: RHD: How many rows? Off by one? is the first row 0Hz or nyquist/n and the last (n-1)/n * nyquist or nyquist?
    frequencies = nyquist * (np.arange(ftmix.shape[0]) + 1) / ftmix.shape[0]

    # Include only frequencies above 0.01 Hz
    ftmix = ftmix[frequencies > 0.01, :]
    frequencies = frequencies[frequencies > 0.01]

    # Set frequency range to [0, 1]
    normalised_frequencies = (frequencies - 0.01) / (nyquist - 0.01)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fraction = np.cumsum(ftmix, axis=0) / np.sum(ftmix, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
    index_cutoff = np.argmin((fcumsum_fraction - 0.5)**2, axis=0)

    # Now get the fractions associated with those indices index, these are the final feature scores
    hfc = normalised_frequencies[index_cutoff]

    # 'High-frequency content' scores
    return hfc


def feature_spatial(melodic_ic_file, aroma_dir=None):
    """Spatial feature scores.

    For each IC determine the fraction of the mixture modelled thresholded Z-maps respectively located within
    the CSF or at the brain edges, using predefined standardized masks.

    Parameters
    ----------
    melodic_ic_file: str
        nii.gz file containing mixture-modelled thresholded (p>0.5) Z-maps, registered to the MNI152 2mm template
    aroma_dir:  str
        ICA-AROMA directory, containing the mask-files (mask_edge.nii.gz, mask_csf.nii.gz & mask_out.nii.gz)

    Returns
    -------
    tuple (rank 1 array like, rank 1 array like)
        Edge and CSF fraction feature scores for the components of the melodic_ics_file file
    """
    assert isfile(melodic_ic_file)
    if aroma_dir is None:
        aroma_dir = AROMADIR
    assert isdir(aroma_dir)

    csf_mask  = join(aroma_dir, 'mask_csf.nii.gz')
    edge_mask = join(aroma_dir, 'mask_edge.nii.gz')
    out_mask  = join(aroma_dir, 'mask_out.nii.gz')

    total_sum, csf_sum, edge_sum, outside_sum = zsums(
        melodic_ic_file, masks=[None, csf_mask, edge_mask, out_mask]
    )

    edge_fraction = np.where(total_sum > csf_sum, (outside_sum + edge_sum) / (total_sum - csf_sum), 0)
    csf_fraction = np.where(total_sum > csf_sum, csf_sum / total_sum, 0)

    return edge_fraction, csf_fraction


def classification(max_rp_correl, edge_fraction, hfc, csf_fraction):
    """Classify a set of components into motion and non-motion components.

    Classification is based on four features:
     - maximum RP correlation
     - high-frequency content,
     - edge-fraction
     - CSF-fraction

    Parameters
    ----------
    max_rp_correl: rank 1 array like
        Maximum RP Correlation feature scores of the components
    edge_fraction: rank 1 array like
        Edge Fraction feature scores of the components
    hfc: rank1 array like
        High-Frequency Content feature scores of the components
    csf_fraction:  ranke 1 array like
        CSF fraction feature scores of the components

    Return
    ------
    rank 1 numpy array
        Indices of the components identified as motion components
    """
    assert len(max_rp_correl) == len(edge_fraction) == len(hfc) == len(csf_fraction)

    # Criteria for classification (thresholds and hyperplane-parameters)
    csf_threshold = 0.10
    hfc_threshold = 0.35
    hyperplane = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    # Project edge and max_rp_correl feature scores to new 1D space
    projection = hyperplane[0] + np.vstack([max_rp_correl, edge_fraction]).T.dot(hyperplane[1:])

    # NB np.where() with single arg returns list of indices satisfying condition
    return np.where((projection > 0) | (csf_fraction > csf_threshold) | (hfc > hfc_threshold))[0]


def save_classification(outdir, max_rp_correl, edge_fraction, hfc, csf_fraction, motion_ic_indices):
    """Save classification results in text files.

    Parameters
    ----------
    outdir: str
        Output directory
    max_rp_correl: rank 1 array like
        Maximum RP Correlation feature scores of the components
    edge_fraction: rank 1 array like
        Edge Fraction feature scores of the components
    hfc: rank1 array like
        High-frequency content' feature scores of the components
    csf_fraction:  rank 1 array like
        CSF fraction feature scores of the components
    motion_ic_indices:  rank 1 array like
        list of indices of components classified as motion

    Return
    ------
    None

    Output (within the requested output directory)
    ------
    text file containing the original feature scores (feature_scores.txt)
    text file containing the indices of the identified components (classified_motion_ICs.txt)
    text file containing summary of classification (classification_overview.txt)
    """
    assert is_writable_directory(outdir)
    assert max(motion_ic_indices) < len(max_rp_correl)
    assert len(max_rp_correl) == len(edge_fraction) == len(hfc) == len(csf_fraction)

    # Feature scores
    np.savetxt(join(outdir, 'feature_scores.txt'),
               np.vstack((max_rp_correl, edge_fraction, hfc, csf_fraction)).T)

    # Indices of motion-classified ICs
    with open(join(outdir, 'classified_motion_ICs.txt'), 'w') as file_:
        if len(motion_ic_indices) > 0:
            print(','.join(['%.0f' % (idx+1) for idx in motion_ic_indices]), file=file_)

    # Summary overview of the classification (nb this is *not* valid TSV!)
    is_motion = np.zeros_like(csf_fraction, dtype=bool)
    is_motion[motion_ic_indices] = True
    with open(join(outdir, 'classification_overview.txt'), 'w') as file_:
        print(
            'IC', 'Motion/noise', 'maximum RP correlation', 'Edge-fraction',
            '', 'High-frequency content', 'CSF-fraction',
            sep='\t', file=file_
        )
        for i in range(len(csf_fraction)):
            print(
                '%d\t%s\t\t%.2f\t\t\t%.2f\t\t\t%.2f\t\t\t%.2f' %
                (i+1, is_motion[i], max_rp_correl[i],
                 edge_fraction[i], hfc[i], csf_fraction[i]),
                file=file_
            )


def denoising(infile, outfile, mix, denoise_indices, aggressive=False):
    """Apply ica denoising using the specified components

    Parameters
    ----------
    infile: str
        Input data file (nii.gz) to be denoised
    outfile: str
        Output file
    mix: rank 2 numpy array
        Melodic mix matrix
    denoise_indices:  rank 1 numpy array like
        Indices of the components that should be regressed out
    aggressive: bool
        Whether to do aggressive denoising
    Returns
    -------
    None

    Output
    ------
    A nii.gz format file of the denoised fMRI data
    """
    nii = nib.load(infile)
    if len(denoise_indices) < 1:
        nib.save(nii, outfile)
        return

    # for memory efficiency we update image data in-place
    # get_data() returns the numpy image data as F-contig (nx, ny, nz, nt)
    # the transpose gives us a C-contig (nt, nz, ny, nx) view.
    reg_filter(nii.get_data().T, design=mix, components=denoise_indices, aggressive=aggressive)
    nib.save(nii, outfile)


def _valid_infile(arg):
    path = os.path.abspath(os.path.normpath(arg))
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError("%s does not exist or is not a file" % path)
    return path


def _valid_indir(arg):
    path = os.path.abspath(os.path.normpath(arg))
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("%s does not exist or is not a directory" % path)
    return path


def _valid_outdir(arg):
    path = os.path.abspath(os.path.normpath(arg))
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise argparse.ArgumentTypeError("%s is not a valid output directory" % path)
    return path


def _valid_feat_dir(arg):
    path = os.path.abspath(os.path.normpath(arg))
    if not is_valid_feat_dir(path):
        raise argparse.ArgumentTypeError("%s is not a valid feat directory" % path)
    return path


def _valid_melodic_dir(arg):
    path = os.path.abspath(os.path.normpath(arg))
    if not is_valid_melodic_dir(path):
        raise argparse.ArgumentTypeError("%s is not a valid melodic directory" % path)
    return path


def _valid_float_in_interval(min_, max_, arg):
    val = float(arg)
    if min_ <= val <= max_:
        return val
    else:
        raise argparse.ArgumentTypeError("%f is outside interval [%f, %f]" % (val, min_, max_))

_valid_tr = partial(_valid_float_in_interval, 0.5, 10)


def _valid_logging_level(arg):
    loglevels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
    if arg.upper() in loglevels:
        return arg.upper()
    else:
        raise argparse.ArgumentTypeError("%s is not a valid loging level" % arg)


def parse_cmdline(args):
    """Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            ('ICA-AROMA %s ("ICA-based Automatic Removal Of Motion Artefacts" on fMRI data).' % __version__) +
            ' See the companion manual for further information.'))

    # Required arguments
    requiredargs = parser.add_argument_group('Required arguments')
    requiredargs.add_argument(
        '-o', '--out', dest="outdir", required=True, type=_valid_outdir,
        help='Output directory name'
    )

    # Required arguments in non-Feat mode
    nonfeatargs = parser.add_argument_group('Required arguments - generic mode')
    nonfeatargs.add_argument(
        '-i', '--in', dest="infile", type=_valid_infile,
        help='Input file name of fMRI data (.nii.gz)'
    )
    nonfeatargs.add_argument(
        '-p', '--motionparams', dest="mc", type=_valid_infile,
        help='mc motion correction file eg prefiltered_func_data_mcf.par')
    nonfeatargs.add_argument(
        '-a', '--affmat', dest="affmat", type=_valid_infile,
        help=(
            'Mat file of the affine registration (eg FLIRT) of the functional data to structural space.' +
            ' (.mat file eg subj.feat/reg/example_func2highres.mat)'))
    nonfeatargs.add_argument(
        '-w', '--warp', dest="warp", type=_valid_infile,
        help=(
            'Warp file of the non-linear registration (eg FNIRT) of the structural data to MNI152 space .' +
            ' (.nii.gz file eg subj.feat/reg/highres2standard_warp.nii.gz)'))


    nonfeatargs.add_argument(
        '-m', '--mask', dest="existing_mask", type=_valid_infile,
        help='Mask file for MELODIC (denoising will be performed on the original/non-masked input data)')

    # Required options in Feat mode
    featargs = parser.add_argument_group('Required arguments - FEAT mode')
    featargs.add_argument(
        '-f', '--feat', dest="featdir", type=_valid_feat_dir,
        help='Existing Feat folder (Feat should have been run without temporal filtering and including' +
             'registration to MNI152)')

    # Optional options
    optionalargs = parser.add_argument_group('Optional arguments')
    optionalargs.add_argument('-T', '--tr', dest="TR", help='TR in seconds', type=_valid_tr)
    optionalargs.add_argument(
        '-t', '--denoisetype', dest="denoise_type", default="nonaggr",
        choices=['none', 'nonaggr', 'aggr', 'both'],
        help=("Denoising strategy: 'none': classification only; 'nonaggr':" +
              " non-aggresssive; 'aggr': aggressive; 'both': both (seperately)"))
    optionalargs.add_argument(
        '-M', '--melodicdir', dest="melodic_dir", default=None, type=_valid_melodic_dir,
        help='MELODIC directory name if MELODIC has been run previously.')
    optionalargs.add_argument(
        '-D', '--dimreduction', dest="dim", default=0, type=int,
        help='Dimensionality reduction into #num dimensions when running MELODIC (default: automatic estimation)')
    optionalargs.add_argument(
        '-s', '--seed', dest="seed", default=None, type=int, help='Random number seed')
    optionalargs.add_argument(
        '-L', '--log', dest="loglevel", default='INFO', type=_valid_logging_level, help='Logging Level')

    parsed_args = parser.parse_args(args)

    # Either a feat directory or all inputs explicit
    if parsed_args.featdir is None:
        if any([parsed_args.infile is None,
                parsed_args.mc is None,
                parsed_args.affmat is None,
                parsed_args.warp is None]):
            print(
                'Either a feat directory or separate input image, ' +
                'motion parameters, affine transform and warp files must specified.',
                file=sys.stderr)
            parser.print_help()
            sys.exit(2)

    return parsed_args


def feat_args(args):
    """File and directory names to use from a feat directory.
    """
    featdir = args.featdir
    assert is_valid_feat_dir(featdir)

    # The names of the input files that should be already present in the Feat directory as we assume it is valid
    infile = join(featdir, 'filtered_func_data.nii.gz')
    mc = join(featdir, 'mc', 'prefiltered_func_data_mcf.par')
    affmat = join(featdir, 'reg', 'example_func2highres.mat')
    warp = join(featdir, 'reg', 'highres2standard_warp.nii.gz')

    melodic_dir = join(featdir, 'filtered_func_data.ica')
    melodic_dir = melodic_dir if isdir(melodic_dir) else args.melodic_dir

    return infile, mc, affmat, warp, melodic_dir


def nonfeat_args(args):
    """File and directory names to use when passed explicitly.
    """
    infile = args.infile
    mc = args.mc
    affmat = args.affmat
    warp = args.warp
    melodic_dir = args.melodic_dir

    assert all(arg is not None for arg in [infile, mc, affmat, warp])
    assert melodic_dir is None or is_valid_melodic_dir(melodic_dir)

    return infile, mc, affmat, warp, melodic_dir


def create_mask(infile, outfile, featdir=None):
    """Create a mask.
    """
    assert isfile(infile)
    assert is_writable_file(outfile)

    if featdir is None:
        # RHD TODO: just binarize stddev of input file?
        check_call([FSLMATHS, infile, '-Tstd', '-bin', outfile])
        return

    # Try and use example_func in feat dir to create a mask
    example_func = join(featdir, 'example_func.nii.gz')
    if isfile(example_func):
        temp_dir = mkdtemp(prefix='create_mask')
        check_call([BET, example_func, join(temp_dir, 'bet'), '-f', '0.3', '-n', '-m', '-R'])
        shutil.move(src=join(temp_dir, 'bet_mask.nii.gz'), dst=outfile)
        shutil.rmtree(temp_dir)
    else:
        logging.warning(
            'No example_func was found in the Feat directory.' +
            ' A mask will be created including all voxels with varying intensity over time in the fMRI data.' +
            ' Please check!'
        )
        check_call([FSLMATHS, infile, '-Tstd', '-bin', outfile])


def run_aroma(infile, outdir, mask, dim, t_r, melodic_dir, affmat, warp, mc, denoise_type, seed=None, verbose=True):
    """Run aroma denoising.

    Parameters
    ----------
    infile: str
        Input data file (nii.gz) to be denoised
    outdir: str
        Output directory
    mask: str
        Mask file to be applied during MELODIC
    dim: int
        Dimensionality of ICA
    t_r: float
        Repetition Time (in seconds) of the fMRI data
    existing_melodic_dir: str
        MELODIC directory in case it has been run before, otherwise define empty string
    affmat: str
        Mat file describing the linear registration to structural space (if image still in native space)
    warp: str
        Warp file describing the non-linear registration to MNI152 space (if image not yet in MNI space)
    mc: str
        Text file containing the realignment parameters
    denoise_type: str
        Type of requested denoising ('aggr', 'nonaggr', 'both', 'none')
    seed: Optional(int)
        Seed for both MELODIC and python RNGs
    verbose: Optional(bool)
        Log info messages and save classification to text files in output directory

    Returns
    -------
    None

    Output (within the requested output directory)
    ------
    A nii.gz file of the denoised fMRI data (denoised_func_data_<denoise_type>.nii.gz) in outdir
    """

    assert isfile(infile)
    assert is_writable_directory(outdir)
    assert isfile(mask)
    assert 0 <= dim < 100
    assert 0.5 <= t_r < 10
    assert melodic_dir is None or isdir(melodic_dir)
    assert affmat is None or isfile(affmat)
    assert warp is None or isfile(warp)
    assert isfile(mc)
    assert denoise_type in ['none', 'aggr', 'nonaggr', 'both']

    tempdir = mkdtemp(prefix='run_aroma')

    logging.info('Step 1) MELODIC')
    melodic_ics_file = join(tempdir, 'thresholded_ics.nii.gz')
    mix, ftmix = run_ica(
        infile, outfile=melodic_ics_file, maskfile=mask, t_r=t_r,
        ndims_ica=dim, melodic_indir=melodic_dir, seed=seed
    )

    logging.info('Step 2) Automatic classification of the components')
    melodic_ics_file_mni = join(tempdir, 'melodic_IC_thr_MNI2mm.nii.gz')
    register_to_mni(melodic_ics_file, melodic_ics_file_mni, affmat=affmat, warp=warp)

    edge_fraction, csf_fraction = feature_spatial(melodic_ics_file_mni)
    max_rp_correl = feature_time_series(mix=mix, rparams=np.loadtxt(mc), seed=seed)
    hfc = feature_frequency(ftmix, t_r=t_r)

    motion_ic_indices = classification(max_rp_correl, edge_fraction, hfc, csf_fraction)

    logging.info('Step 3) Data denoising')
    if denoise_type in ['nonaggr', 'both']:
        outfile = join(outdir, 'denoised_func_data_nonaggr.nii.gz')
        denoising(infile, outfile, mix, motion_ic_indices, aggressive=False)
    if denoise_type in ['aggr', 'both']:
        outfile = join(outdir, 'denoised_func_data_aggr.nii.gz')
        denoising(infile, outfile, mix, motion_ic_indices, aggressive=True)

    shutil.rmtree(tempdir)

    if verbose:
        save_classification(outdir, max_rp_correl, edge_fraction, hfc, csf_fraction, motion_ic_indices)


def main(argv=sys.argv):
    """Command line entry point."""

    args = parse_cmdline(argv[1:])

    level = getattr(logging, args.loglevel, None)
    if level is not None:
        print('Logging Level is %s (%d)' % (args.loglevel, level))
        logging.basicConfig(level=level)

    using_feat = args.featdir is not None
    try:
        infile, mc, affmat, warp, melodic_dir = feat_args(args) if using_feat else nonfeat_args(args)
    except ValueError as exception:
        print('%s: %s' % (sys.argv[0], exception), file=sys.stderr)
        sys.exit(1)

    # Get TR of the fMRI data, if not specified and check
    if args.TR is not None:
        TR = args.TR
    else:
        TR = nifti_pixdims(infile)[3]
        if not (0.5 <= TR <= 10):
            logging.critical(
                ('Unexpected TR value (%f not in [0.5, 10]) found in nifti header. ' % TR) +
                'Check the header, or define the TR explicitly as an additional argument.' +
                'Exiting ...'
            )
            sys.exit(1)

    mask = join(args.outdir, 'mask.nii.gz')
    if args.existing_mask is not None:
        shutil.copyfile(src=args.existing_mask, dst=mask)
    elif using_feat:
        create_mask(infile, outfile=mask, featdir=args.featdir)
    else:
        create_mask(infile, outfile=mask)

    try:
        run_aroma(
            infile=infile,
            outdir=args.outdir,
            mask=mask,
            dim=args.dim,
            t_r=TR,
            melodic_dir=melodic_dir,
            affmat=affmat,
            warp=warp,
            mc=mc,
            denoise_type=args.denoise_type,
            seed=args.seed
        )
    except CalledProcessError as e:
        logging.critical('Error in calling external FSL command: %s. Exiting ...' % e)
        return 1
    except Exception as e:
        logging.critical('Internal Error: %s. Exiting ...' % e)
        return 1
    return 0

if __name__ == '__main__':
    # installed as standalone script
    import signal

    def handler(signum, frame):
        print('Interrupted. Exiting ...', file=sys.stderr)
        sys.exit(1)
    signal.signal(signal.SIGINT, handler)

    AROMADIR = _find_aroma_dir(os.environ.get("AROMADIR", None))
    if AROMADIR is None:
        logging.critical(
            'Unable to find aroma data directory with mask files. ' +
            'Exiting ...')
        sys.exit(1)
    sys.exit(main())
else:
    # installed as python package
    try:
        import pkg_resources
        PKGDATA = pkg_resources.resource_filename(__name__, 'data')
    except ImportError:
        PKGDATA = join(dirname(__file__), 'data')
    AROMADIR = os.environ.get("AROMADIR", PKGDATA)
