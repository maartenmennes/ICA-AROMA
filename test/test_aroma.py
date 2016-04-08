from __future__ import print_function, division

import sys
sys.path.append('..')
import os
from os.path import join, normpath, isfile
from tempfile import mkstemp, mkdtemp
import filecmp
import shutil
import subprocess

import numpy as np
import nibabel as nib

import aroma

# Monkey patching aroma for testing
aroma.AROMADIR = normpath('..')
def _call(*args, **kwargs):
    #print(' '.join(args[0]), file=sys.stderr)
    subprocess.call(*args, **kwargs)
def _check_output(*args, **kwargs):
    #print(' '.join(args[0]), file=sys.stderr)
    return subprocess.check_output(*args, **kwargs)
aroma.call = _call
aroma.check_output = _check_output

def setup():
    pass
 
def teardown():
    pass


def test_nifti_info():
    dtype = aroma.nifti_info('refin/filtered_func_data.nii.gz', 'data_type')
    assert dtype == 'FLOAT32'


def test_nifti_dims():
    dims = aroma.nifti_dims('refin/filtered_func_data.nii.gz')
    assert dims == (64, 64, 34, 180)


def test_nifti_pixdims():
    pixdims = aroma.nifti_pixdims('refin/filtered_func_data.nii.gz')
    assert pixdims == (3.0, 3.0, 3.5, 2.0)


def test_is_writable_file():
    assert aroma.is_writable_file('/tmp/gagaga')
    assert not aroma.is_writable_file(None)
    assert not aroma.is_writable_file('')
    assert not aroma.is_writable_file('/nopath/gaga')
    assert not aroma.is_writable_file('/root/gaga')



def test_is_writable_directory():
    assert aroma.is_writable_directory('/tmp')
    assert not aroma.is_writable_directory(None)
    assert not aroma.is_writable_directory('')
    assert not aroma.is_writable_directory('/nopath')
    assert not aroma.is_writable_directory('/root')


def test_zsums():
    total_sum = aroma.zsums('refout/melodic_IC_thr_MNI2mm.nii.gz')
    edge_sum  = aroma.zsums('refout/melodic_IC_thr_MNI2mm.nii.gz', '../mask_edge.nii.gz')
    csf_sum   = aroma.zsums('refout/melodic_IC_thr_MNI2mm.nii.gz', '../mask_csf.nii.gz')
    outside_sum   = aroma.zsums('refout/melodic_IC_thr_MNI2mm.nii.gz', '../mask_out.nii.gz')
    edge_fractions = np.where(total_sum > csf_sum, (outside_sum + edge_sum) / (total_sum - csf_sum), 0)
    csf_fractions = np.where(total_sum > csf_sum, csf_sum / total_sum, 0)
    assert np.all((0 <= edge_fractions) & (edge_fractions <= 1))
    assert len(edge_fractions) == len(csf_fractions) == 45
    scores = np.loadtxt('refout/feature_scores.txt').T
    assert ((edge_fractions - scores[1])**2).sum() < 1e-6
    assert ((csf_fractions - scores[3])**2).sum() < 1e-6
    

def test_cross_correlation():
    # Using numpy random here shouldn't interfere with stdlib random
    a = np.random.rand(100, 5) - 0.5
    b = a[:, [0, 2, 4, 3, 1]]
    xcorr = aroma.cross_correlation(a, b)
    expected = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0]
    ], dtype=bool)
    assert np.all((np.abs(xcorr) > 0.75) == expected)


def test_is_valid_melodic_dir():
    assert aroma.is_valid_melodic_dir('refout/melodic.ica')
    assert not aroma.is_valid_melodic_dir(None)
    assert not aroma.is_valid_melodic_dir('')
    assert not aroma.is_valid_melodic_dir('/nopath')
    assert not aroma.is_valid_melodic_dir('/etc')


def test_run_ica():
    outdir = mkdtemp(prefix='test_run_ica')
    outfile = join(outdir, 'melodic_IC_thr.nii.gz')
    mix, ftmix = aroma.run_ica(
        infile='refin/filtered_func_data.nii.gz', 
        outfile=outfile,
        maskfile='refin/mask.nii.gz', # also one in refout/
        t_r=2.0,
        ndims_ica=0,
        melodic_indir=None,
        seed=31415926
    )
    assert filecmp.cmp(outfile, 'refout/melodic_IC_thr.nii.gz', shallow=False)
    shutil.rmtree(outdir)


def test_register_to_mni():
    outdir = mkdtemp(prefix='test_register_to_mni')
    outfile = join(outdir, 'melodic_IC_thr_MNI2mm.nii.gz')
    aroma.register_to_mni(
        infile='refout/melodic_IC_thr.nii.gz',
        outfile=outfile,
        affmat='refin/reg/example_func2highres.mat',
        warp='refin/reg/highres2standard_warp.nii.gz'
    )
    assert filecmp.cmp(outfile, 'refout/melodic_IC_thr_MNI2mm.nii.gz', shallow=False)
    shutil.rmtree(outdir)


def test_feature_time_series():
    max_rp_correl = aroma.feature_time_series(
        np.loadtxt('refout/melodic.ica/melodic_mix'),
        rparams=np.loadtxt('refin/mc/prefiltered_func_data_mcf.par'),
        seed=31415926
    )   
    scores = np.loadtxt('refout/feature_scores.txt').T
    assert ((max_rp_correl - scores[0])**2).sum() < 1e-6 


def test_feature_frequency():
    hfc = aroma.feature_frequency(
        np.loadtxt('refout/melodic.ica/melodic_FTmix'),
        t_r=2.0
    )
    scores = np.loadtxt('refout/feature_scores.txt').T
    assert ((hfc - scores[2])**2).sum() < 1e-6    


def test_feature_spatial():
    edge_fractions, csf_fractions = aroma.feature_spatial(
        'refout/melodic_IC_thr_MNI2mm.nii.gz',
        aroma_dir='..'
    )
    assert len(edge_fractions) == len(csf_fractions) == 45
    scores = np.loadtxt('refout/feature_scores.txt').T
    assert ((edge_fractions - scores[1])**2).sum() < 1e-6
    assert ((csf_fractions - scores[3])**2).sum() < 1e-6


def test_classification():
    scores = np.loadtxt('refout/feature_scores.txt').T
    max_rp_correl, edge_fraction, hfc, csf_fraction = scores
    indices = list(aroma.classification(max_rp_correl, edge_fraction, hfc, csf_fraction))
    # written out as base 1, so
    ref_indices = list(np.loadtxt('refout/classified_motion_ICs.txt', delimiter=',').astype(int) - 1)
    assert indices == ref_indices


def test_save_classification():
    outdir = mkdtemp(prefix='test_save_classification')
    ref_scores = np.loadtxt('refout/feature_scores.txt')
    time_series, edge_fractions, hfc, csf_fractions = ref_scores.T
    denoise_indices = list(np.loadtxt('refout/classified_motion_ICs.txt', delimiter=',').astype(int) - 1)
    aroma.save_classification(
        outdir=outdir,
        max_rp_correl=time_series,
        edge_fraction=edge_fractions,
        hfc=hfc,
        csf_fraction=csf_fractions,
        motion_ic_indices=denoise_indices
    )

    assert np.allclose(np.loadtxt(join(outdir, 'feature_scores.txt')), ref_scores)
    assert list(np.loadtxt(join(outdir, 'classified_motion_ICs.txt'), delimiter=',').astype(int) - 1) == denoise_indices
    assert filecmp.cmp(join(outdir, 'classification_overview.txt'), 'refout/classification_overview.txt')
    shutil.rmtree(outdir)


def test_denoising():
    outdir = mkdtemp(prefix='test_denoising')
    outfile = join(outdir, 'nonaggr_test.nii.gz')
    denoise_indices = list(np.loadtxt('refout/classified_motion_ICs.txt', delimiter=',').astype(int) - 1)
    aroma.denoising(
        infile='refin/filtered_func_data.nii.gz',
        outfile=outfile,
        mix=np.loadtxt('out/melodic.ica/melodic_mix'),
        denoise_indices=denoise_indices,
        aggressive=False,
    )
    assert filecmp.cmp(outfile, 'refout/denoised_func_data_nonaggr.nii.gz')
    shutil.rmtree(outdir)


def test_parse_cmdline():
    pass


def test_feat_args():
    pass


def test_nonfeat_arg():
    pass


def test_common_args():
    pass


def test_create_mask():
    outdir = mkdtemp(prefix='test_create_mask')
    outfile = join(outdir, 'mask.nii.gz')
    aroma.create_mask(infile='refin/filtered_func_data.nii.gz', outfile=outfile)

    assert filecmp.cmp(outfile, 'refout/mask.nii.gz')
    shutil.rmtree(outdir)


def test_run_aroma():
    outdir = mkdtemp(prefix='test_run_aroma')
    aroma.run_aroma(
        infile='refin/filtered_func_data.nii.gz',
        outdir=outdir,
        mask='refout/mask.nii.gz',
        dim=0,
        t_r=2.0 ,
        melodic_dir=None,
        affmat='refin/reg/example_func2highres.mat',
        warp='refin/reg/highres2standard_warp.nii.gz',
        mc='refin/mc/prefiltered_func_data_mcf.par',
        denoise_type='nonaggr',
        seed=31415926,
        verbose=True)


    files_to_check_exact = [
        'denoised_func_data_nonaggr.nii.gz',
        'classification_overview.txt',
    ]
    for f in files_to_check_exact:
        assert isfile(join(outdir, f)), 'Missing file %s' % f
        assert filecmp.cmp(join(outdir, f), join('refout', f)), 'File %s mismatch' % f

    # There are some trivial white space differences in these which are OK
    f = 'classified_motion_ICs.txt'
    assert np.allclose(
        np.loadtxt(join(outdir, f), delimiter=','),
        np.loadtxt(join('refout', f), delimiter=',')
    ), 'File %s numerical mismatch' % f

    f = 'feature_scores.txt'
    assert np.allclose(
        np.loadtxt(join(outdir, f)),
        np.loadtxt(join('refout', f))
    ), 'File %s numerical mismatch' % f

    shutil.rmtree(outdir)


