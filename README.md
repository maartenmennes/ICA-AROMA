# ICA-AROMA
ICA-AROMA (*ICA-based Automatic Removal Of Motion Artefacts*) is a data-driven method to identify and remove motion-related independent components from fMRI data. To that end it exploits a small, but robust set of theoretically motivated features, avoiding the need for classifier re-training and thus providing direct and easy applicability. This version requires python (version 2.7 or 3.5 upwards) and an installation of FSL. Please see the [manual](doc/manual.md) for a description of how to run ICA-AROMA.



## Change Log

Note that some earlier versions of the original ICA-AROMA scripts (v0.1-beta and v0.2-beta) contained a mistake at the denoising stage of the method that led to incorrect output. The issue was resolved in version v0.3-beta (27th of April 2015) onwards.

### v0.3-beta to v0.4.0
- port to python 2/3
- general refactor
- speed up and simplify feature calculations
- add test harness (nose)
- remove restrictions due to irregular path handling
- use nibabel for accessing nifti file info
- move documentation to markdown
- replace `fsl_regfilt` with an internal routine
- allow installation as package via pip or directly as standalone script
- gnu standard short/long forms for command line arguments

### 0.2-beta to v0.3-beta
- Correct off by one error in the definition of the string of indices of the components to be removed by *fsl_regfilt*
- Take the maximum of the *absolute* value of the correlation between the component time-course and set of realignment parameters
- Correct for the fact that the defined frequency-range used for the high-frequency content feature, in few cases, did not include the final Nyquist frequency due to limited numerical precision

## Dependencies
This is tested with python versions 2.7 and 3.5, specifically those distributed in the [anaconda](https://docs.continuum.io/anaconda/) python distribution. It should also work with the system python (generally 2.7) on recent versions of Linux. In addition to a python installation the following is required:

 - The [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) software suite.
 - The [numpy](http://www.numpy.org/) python package
 - The [nibabel](http://nipy.org/nibabel/) python package.

 These are available on ubuntu and debian systems via the [neurodebian](http://neuro.debian.net/) repository or may be installed explicitly via
 downloads and/or `pip install`.
