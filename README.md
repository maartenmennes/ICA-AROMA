# ICA-AROMA
ICA-AROMA (*ICA-based Automatic Removal Of Motion Artifacts*) is a data-driven method to identify and remove motion-related independent components from fMRI data. To that end it exploits a small, but robust set of theoretically motivated features, avoiding the need for classifier re-training and thus providing direct and easy applicability. This version requires python (version 2.7 or 3.5 upwards) and an installation of FSL. Please read the [manual](doc/manual.md) for a description of how to run ICA-AROMA.

**NOTE**: Previous versions of the ICA-AROMA scripts (v0.1-beta and v0.2-beta) contained a crucial mistake at the denoising stage of the method. Unfortunately, this means that the output of these versions of the script is incorrect. The issue is resolved in version v0.3-beta (27th of April 2015) onwards.

## Change Log

### v0.3-beta to v0.4
1) port to python 3
2) general refactor
3) speed up and simplify feature calculations
3) add test harness (nose)
4) remove restrictions due to irregular path handling
5) use nibabel for accessing nifti file info
6) move documentation to markdown
7) replace `fsl_regfilt` with an internal routine

### 0.2-beta to v0.3-beta
1) Correct for incorrect definition of the string of indices of the components to be removed by *fsl_regfilt*:
```
	changed   denIdxStr = np.char.mod('%i',denIdx)
	to        denIdxStr = np.char.mod('%i',(denIdx+1))
```
2) Now take the maximum of the 'absolute' value of the correlation between the component time-course and set of realignment parameters: 
```
	changed   maxTC[i,:] = corMatrix.max(axis=1)
	to        corMatrixAbs = np.abs(corMatrix)
              maxTC[i,:] = corMatrixAbs.max(axis=1)
```
3) Correct for the fact that the defined frequency-range, used for the high-frequency content feature, in few cases did not include the final Nyquist frequency due to limited numerical precision:
```
	changed   step = Ny / FT.shape[0]
	          f = np.arange(step,Ny,step)
	to        f = Ny*(np.array(range(1,FT.shape[0]+1)))/(FT.shape[0])
```

## Dependencies
This is tested with python versions 2.7 and 3.5, specifically those distributed in the [anaconda](https://docs.continuum.io/anaconda/) python distribution. It should also work with the system python (generally 2.7) on recent versions of Linux. In addition to a python installation the following is required:

 - The [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) software suite.
 - The [numpy](http://www.numpy.org/) python package
 - The [nibabel](http://nipy.org/nibabel/) python package.

 These are available on ubuntu and debian systems via the [neurodebian](http://neuro.debian.net/) repository or may be installed explicitly via
 downloads and/or `pip install`.
