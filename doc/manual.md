# Manual for ICA-AROMA

## 1 Introduction

ICA-AROMA (for "ICA-based Automatic Removal Of Motion Artifacts") attempts to identify and remove motion artefacts from fMRI data.
To that end it exploits Independent Component Analysis (ICA) to decompose the data into a set of independent components.
Based on the results of a [FSL-MELODIC](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC) ICA decomposition, ICA-AROMA automatically identifies which of these components are related to head motion, by using four robust and standardized features.

The identified components are then removed from the data through linear regression in similar way to [fsl_regfilt](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC#fsl_regfilt_command-line_program).

Within the typical fMRI preprocessing pipeline ICA-AROMA has to be applied *after* spatial smoothing, but *prior to* temporal filtering.

Two manuscripts provide a detailed description and evaluation of ICA-AROMA:

1. Pruim, R.H.R., Mennes, M., van Rooij, D., Llera, A., Buitelaar, J.K., Beckmann, C.F.,
   2015, ICA-AROMA: A robust ICA-based strategy for removing motion artifacts from
   fMRI data. *NeuroImage* [doi:10.1016/j.neuroimage.2015.02.064](http://dx.doi.org/10.1016/j.neuroimage.2015.02.064)
2. Pruim, R.H.R., Mennes, M., Buitelaar, J.K., Beckmann, C.F., 2015. Evaluation of
   ICA-AROMA and alternative strategies for motion artifact removal in resting-state fMRI.
   *NeuroImage* [doi:10.1016/j.neuroimage.2015.02.063](http://dx.doi.org/10.1016/j.neuroimage.2015.02.063)


## 2 General info

The `icaaroma` package consists of essentially a single python script: `aroma.py` that can be installed within the package `icaaroma` or run standalone by making it executable and running directly. The `icaaroma` package furthermore contains three spatial maps (CSF, edge & out-of-brain masks) which are required to derive the spatial features used by ICA-AROMA. These need to be installed in a location accessible to the script, either within the package or in a system location such as `/usr/local/share/aroma`. There is an included `Makefile` for a standalone installation.

## 3 Requirements

- [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
- [Python](https://www.python.org/) (either version 2.7 or 3.X)
- [Numpy](http://www.numpy.org/)
- [Nibabel](http://nipy.org/nibabel/)

The system python/numpy may normally be used. Otherwise, the [Anaconda](https://docs.continuum.io/anaconda) distribution is both easy to install and well suited to scientific programming. Nibabel may be installed within anaconda using [pip](https://pypi.python.org/pypi).

## 4 Installation
From a local clone of the repository on [github](https://github.com/rtrhd/ICA-AROMA/tree/rhdclean):
```
$ git clone https://github.com/rtrhd/ICA-AROMA.git
$ cd ICA-AROMA
$ git checkout rhdclean
```
either install as a package `icaaroma` to the system python
```
$ sudo python setup.py install
```
or as a standalone script `aroma`
```
$ sudo make install
```
Either way an executable python script called `aroma` will be installed to `/usr/local/bin`.

## 5 Run ICA-AROMA - generic

In normal use, the `aroma` requires the following five command line arguments:

|                   |                                                                |
|-------------------|----------------------------------------------------------------|
|-i, --in           | Input file name of fMRI data (.nii.gz)                         |
|-o, --out          | Output directory name                                          |
|-a, --affmat       | File name of the mat-file describing the affine registration   |
|                   | (e.g. FSL FLIRT) of the functional data to structural space    |
|                   | (.mat file)                                                    |
|-w, --warp         | File name of the warp-file describing the non-linear           |
|                   | registration (e.g. FSL FNIRT) of the structural data to MNI152 |
|                   | space (.nii.gz)                                                |
|-p, --motionparams | File name of the text file containing the six (column-wise)    |
|                   | realignment parameters time-courses derived from               |
|                   | volume-realignment (e.g. MCFLIRT)                              |


Example:

```no-highlight
$ aroma \
    --in func_smoothed.nii.gz --out ICA_AROMA \
    --affmat reg/func2highres.mat
    --warp reg/highres2standard_warp.nii.gz --motionparams mc/rest_mcf.par
```

The program needs to be able to find the registration files required to transform the obtained ICA components to the MNI152 2mm template in order to derive standardized spatial feature scores. All ouput is the the specified directory. The input fMRI nifti image will not be modified.


### 5.1 Masking

Either the input fMRI data should be masked (i.e. brain-extracted) or a specific mask should be specified (-m, --mask) on the command line.

Example:
```no-highlight
$ aroma \
    --in func_smoothed.nii.gz --out ICA_AROMA \
    --motionparams mc/rest_mcf.par \
    --affmat reg/example_func2highres.mat \
    --warp reg/highres2standard_warp.nii.gz \
    --mask mask_aroma.nii.gz
```

We recommend not using the mask determined by [FSL-FEAT](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT). This mask is optimized to be used for first-level analysis, as has been dilated to ensure that all *active* voxels are included.
Instead, we recommend creating a mask using the [FSL-BET](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET) Brain Extraction Tool (using a fractional intensity of 0.3), on a non-brain-extracted example or a mean functional image (e.g. example_func within the FEAT directory).

Example of creating an appropriate mask:
```no-highlight
$ bet <input> <output> -f 0.3 -n -m -R
```

Note that the specified mask will only be used at the first stage (ICA) of ICA-AROMA. The
output fMRI data-file is not masked.


## 6 Run ICA-AROMA - after FEAT

ICA-AROMA is optimized for usage after preprocessing fMRI data with FSL FEAT, assuming
the directory meets the standardized folder/file-structure, no temporal filtering has been applied
and it was run including registration to the MNI152 template.
In this case, only the FEAT directory has to be specified (-f, -feat) next to an output
directory. ICA-AROMA will automatically define the appropriate files, create an appropriate
mask (see section 4.1) and use the `melodic.ica` directory if available (in case `MELODIC ICA
data exploration` was checked in FEAT). We don't recommend running MELODIC within FEAT
such that MELODIC will be run within ICA-AROMA using the appropriate mask. Moreover,
this option in FEAT is meant for data exploration after full data pre-processing. As such it can be
applied after ICA-AROMA, temporal high-pass filtering etc.

Example:
```no-highlight
$ aroma --feat rest.feat --out rest.feat/ICA_AROMA/
```

## 7 Output

|                             |                                                                |
|-----------------------------|----------------------------------------------------------------|
|denoised_func_data           | Denoised fMRI data, suffixed with "_nonaggr.nii.gz" or         |
|                             | "_aggr.nii.gz" depending on the requested type of denoising (8)|
|classification_overview.txt  | Complete overview of the classification results.               |
|classified_motion_ICs.txt    | List of indices of the components classified as motion/noise.  |
|feature_scores.txt           | File containing the four feature scores of all components.     |
|melodic_IC_thr_MNI2mm.nii.gz | Spatial maps resulting from MELODIC, after mixture modelling   |
|                             | thresholding and registered to the MNI152 2mm template.        |
|mask.nii,gz                  | Mask used for MELODIC.                                         |
|melodic.ica                  | MELODIC output directory.                                      | 

## 8 Additional options

### 8.1 Optional settings

|                   |                                                             |
|-------------------|-------------------------------------------------------------|
|--tr               | TR in seconds. If this is not specified the TR will be      |
|                   | extracted from the header of the fMRI nifti file. |
|-D, --dimreduction | Dimensionality reduction when running MELODIC |
|                   |  (default is automatic estimation)   |
|-t, --denoisetype  | Type of denoising strategy:            |
|                   |    - no: only classification, no denoising                    |
|                   |    - nonaggr (default): non-aggresssive denoising, i.e. partial component regression |
|                   |    - aggr: aggressive denoising, i.e. full component regression |
|                   |    - both: both aggressive and non-aggressive denoising (two outputs) |
|-s, --seed         | Use a fixed seed for RNG to ensure reproducible results |
|-L, --log          | Logging level:  CRITICAL, ERROR, WARNING, INFO (default) or DEBUG |

### 8.2 MELODIC

When you have already run MELODIC you can specify the melodic directory as additional input
(`-M`, `--melodicdir`; see example below) to avoid running MELODIC again. Note that MELODIC
should have been run on fMRI data prior to temporal filtering and after spatial smoothing.
Preferably, it has been run with the recommended mask (see section 4.1). Unless you have a
good reason for doing otherwise, we advise to run MELODIC as part of ICA-AROMA so that it
runs with optimal settings.

Example:
```no-highlight
$ aroma \
    --in filtered_func_data.nii.gz \
    --out ICA_AROMA --motionparams mc/rest_mcf.par --mask mask_aroma.nii.gz \
    --affmat reg/func2highres.mat \
    --warp reg/highres2standard_warp.nii.gz \
    --melodicdir filtered_func_data.ica
```

### 8.3 Registration

ICA-AROMA is designed (and validated) to run on data in native space, hence the requested
`affmat` and `warp` files. However, ICA-AROMA can also be applied on data within structural
or standard space. In these cases, just do not specify the `affmat` and/or `warp` files. Moreover, if
you applied linear instead of non-linear registration of the functional data to standard space, you
only have to specify the `affmat` file (e.g. `example_func2standard.mat`). In other words,
depending on the which registration files you specify, ICA-AROMA assumes the data to be in
native, structural or standard space and will run the specified registration. When you do not
have a `affmat` and/or `warp` file available (e.g. fMRI performed with another software package
then FSL), please create these files using [FSL-FLIRT](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide) and
[FSL-FNIRT](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FNIRT/UserGuide) respectively.

Example (for data in MNI152 space):
```no-highlight
$ aroma \
    --in filtered_func_data2standard.nii.gz \
    --out ICA_AROMA --motionparams mc/rest_mcf.par \
    --mask mask_aroma.nii.gz
```

Example (in case linear registration to MNI152 space should be applied):
```no-highlight
$ aroma \
    --in func_smoothed.nii.gz --out ICA_AROMA  \
    --motionparams mc/rest_mcf.par \
    --affmat reg/func2standard.mat --mask mask_aroma.nii.gz
```
