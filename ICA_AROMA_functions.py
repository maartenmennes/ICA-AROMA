#!/usr/bin/env python

# Functions for ICA-AROMA v0.3 beta

from __future__ import division
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from past.utils import old_div
def runICA(fslDir, inFile, outDir, melDirIn, mask, dim, TR):
	""" This function runs MELODIC and merges the mixture modeled thresholded ICs into a single 4D nifti file

	Parameters
	---------------------------------------------------------------------------------
	fslDir:		Full path of the bin-directory of FSL
	inFile:		Full path to the fMRI data file (nii.gz) on which MELODIC should be run
	outDir:		Full path of the output directory
	melDirIn:	Full path of the MELODIC directory in case it has been run before, otherwise define empty string
	mask:		Full path of the mask to be applied during MELODIC
	dim:		Dimensionality of ICA
	TR:		TR (in seconds) of the fMRI data
	
	Output (within the requested output directory)
	---------------------------------------------------------------------------------
	melodic.ica		MELODIC directory
	melodic_IC_thr.nii.gz	merged file containing the mixture modeling thresholded Z-statistical maps located in melodic.ica/stats/ """

	# Import needed modules
	import os
	import subprocess

	# Define the 'new' MELODIC directory and predefine some associated files
	melDir = os.path.join(outDir,'melodic.ica')
	melIC = os.path.join(melDir,'melodic_IC.nii.gz')
	melICmix = os.path.join(melDir,'melodic_mix')
	melICthr = os.path.join(outDir,'melodic_IC_thr.nii.gz')

	# When a MELODIC directory is specified, check wheter all needed files are present. Otherwise... run MELODIC again
	if (len(melDir) != 0) and os.path.isfile(os.path.join(melDirIn,'melodic_IC.nii.gz')) and os.path.isfile(os.path.join(melDirIn,'melodic_FTmix')) and os.path.isfile(os.path.join(melDirIn,'melodic_mix')):

		print('  - The existing/specified MELODIC directory will be used.')

		# If a 'stats' directory is present (contains thresholded spatial maps) create a symbolic link to the MELODIC directory. Otherwise create specific links and run mixture modeling to obtain thresholded maps.
		if os.path.isdir(os.path.join(melDirIn,'stats')):
			os.symlink(melDirIn,melDir)
		else:
			print('  - The MELODIC directory does not contain the required \'stats\' folder. Mixture modeling on the Z-statistical maps will be run.')
			
			# Create symbolic links to the items in the specified melodic directory
			os.makedirs(melDir)
			for item in os.listdir(melDirIn):
				os.symlink(os.path.join(melDirIn,item),os.path.join(melDir,item))

			# Run mixture modeling
			os.system(' '.join([os.path.join(fslDir,'melodic'),
				'--in=' + melIC,
				'--ICs=' + melIC,
				'--mix=' + melICmix,
				'--outdir=' + melDir,
				'--Ostats --mmthresh=0.5']))
			
	else:
		# If a melodic directory was specified, display that it did not contain all files needed for ICA-AROMA (or that the directory does not exist at all)
		if len(melDirIn) != 0 :
			if not os.path.isdir(melDirIn):
				print('  - The specified MELODIC directory does not exist. MELODIC will be run seperately.')
			else:
				print('  - The specified MELODIC directory does not contain the required files to run ICA-AROMA. MELODIC will be run seperately.')
		
		# Run MELODIC
		os.system(' '.join([os.path.join(fslDir,'melodic'),
			'--in=' + inFile, 
			'--outdir=' + melDir, 
			'--mask=' + mask, 
			'--dim=' + str(dim),
			'--Ostats --nobet --mmthresh=0.5 --report',
			'--tr=' + str(TR)]))

	# Get number of components
	cmd = ' '.join([os.path.join(fslDir,'fslinfo'),
		melIC,
		'| grep dim4 | head -n1 | awk \'{print $2}\''])
	nrICs=int(float(subprocess.getoutput(cmd)))

	# Merge mixture modeled thresholded spatial maps. Note! In case that mixture modeling did not converge, the file will contain two spatial maps. The latter being the results from a simple null hypothesis test. In that case, this map will have to be used (first one will be empty).
	for i in range(1,nrICs+1):
		# Define thresholded zstat-map file
		zTemp = os.path.join(melDir,'stats','thresh_zstat' + str(i) + '.nii.gz')
		cmd = ' '.join([os.path.join(fslDir,'fslinfo'),
			zTemp,
			'| grep dim4 | head -n1 | awk \'{print $2}\''])
		lenIC=int(float(subprocess.getoutput(cmd)))

		# Define zeropad for this IC-number and new zstat file
		cmd = ' '.join([os.path.join(fslDir,'zeropad'),
			str(i),
			'4'])
		ICnum=subprocess.getoutput(cmd)	
		zstat = os.path.join(outDir,'thr_zstat' + ICnum)		

		# Extract last spatial map within the thresh_zstat file
		os.system(' '.join([os.path.join(fslDir,'fslroi'),
			zTemp,		# input
			zstat,		# output
			str(lenIC-1),	# first frame
			'1']))		# number of frames

	# Merge and subsequently remove all mixture modeled Z-maps within the output directory
	os.system(' '.join([os.path.join(fslDir,'fslmerge'),
		'-t',						# concatenate in time
		melICthr,					# output
		os.path.join(outDir,'thr_zstat????.nii.gz')]))	# inputs

	os.system('rm ' + os.path.join(outDir,'thr_zstat????.nii.gz'))

	# Apply the mask to the merged file (in case a melodic-directory was predefined and run with a different mask)
	os.system(' '.join([os.path.join(fslDir,'fslmaths'),
		melICthr,
		'-mas ' + mask,
		melICthr]))

def register2MNI(fslDir, inFile, outFile, affmat, warp):
	""" This function registers an image (or time-series of images) to MNI152 T1 2mm. If no affmat is defined, it only warps (i.e. it assumes that the data has been registerd to the structural scan associated with the warp-file already). If no warp is defined either, it only resamples the data to 2mm isotropic if needed (i.e. it assumes that the data has been registered to a MNI152 template). In case only an affmat file is defined, it assumes that the data has to be linearly registered to MNI152 (i.e. the user has a reason not to use non-linear registration on the data).

	Parameters
	---------------------------------------------------------------------------------
	fslDir:		Full path of the bin-directory of FSL
	inFile:		Full path to the data file (nii.gz) which has to be registerd to MNI152 T1 2mm
	outFile:	Full path of the output file
	affmat:		Full path of the mat file describing the linear registration (if data is still in native space)
	warp:		Full path of the warp file describing the non-linear registration (if data has not been registered to MNI152 space yet)

	Output (within the requested output directory)
	---------------------------------------------------------------------------------
	melodic_IC_mm_MNI2mm.nii.gz	merged file containing the mixture modeling thresholded Z-statistical maps registered to MNI152 2mm """


	# Import needed modules
	import os
	import subprocess

	# Define the MNI152 T1 2mm template
	fslnobin = fslDir.rsplit('/',2)[0] 
	ref = os.path.join(fslnobin,'data','standard','MNI152_T1_2mm_brain.nii.gz')

	# If the no affmat- or warp-file has been specified, assume that the data is already in MNI152 space. In that case only check if resampling to 2mm is needed
	if (len(affmat) == 0) and (len(warp) == 0):
		# Get 3D voxel size
		pixdim1=float(subprocess.getoutput('%sfslinfo %s | grep pixdim1 | awk \'{print $2}\'' % (fslDir,inFile) ))
		pixdim2=float(subprocess.getoutput('%sfslinfo %s | grep pixdim2 | awk \'{print $2}\'' % (fslDir,inFile) ))
		pixdim3=float(subprocess.getoutput('%sfslinfo %s | grep pixdim3 | awk \'{print $2}\'' % (fslDir,inFile) ))
	
		# If voxel size is not 2mm isotropic, resample the data, otherwise copy the file
		if (pixdim1 != 2) or (pixdim2 != 2) or (pixdim3 !=2 ):
			os.system(' '.join([os.path.join(fslDir,'flirt'),
				' -ref ' + ref,
				' -in ' + inFile,
				' -out ' + outFile,
				' -applyisoxfm 2 -interp trilinear']))
		else:
			os.system('cp ' + inFile + ' ' + outFile)
	
	# If only a warp-file has been specified, assume that the data has already been registered to the structural scan. In that case apply the warping without a affmat
	elif (len(affmat) == 0) and (len(warp) != 0):
		# Apply warp
		os.system(' '.join([os.path.join(fslDir,'applywarp'),
			'--ref=' + ref,
			'--in=' + inFile,
			'--out=' + outFile,
			'--warp=' + warp,
			'--interp=trilinear']))

	# If only a affmat-file has been specified perform affine registration to MNI
	elif (len(affmat) != 0) and (len(warp) == 0):
		os.system(' '.join([os.path.join(fslDir,'flirt'),
			'-ref ' + ref,
			'-in ' + inFile,
			'-out ' + outFile,
			'-applyxfm -init ' + affmat,
			'-interp trilinear']))

	# If both a affmat- and warp-file have been defined, apply the warping accordingly
	else:
		os.system(' '.join([os.path.join(fslDir,'applywarp'),
			'--ref=' + ref,
			'--in=' + inFile,
			'--out=' + outFile,
			'--warp=' + warp,
			'--premat=' + affmat,
			'--interp=trilinear']))


def feature_time_series(melmix, mc):
	""" This function extracts the maximum RP correlation feature scores. It determines the maximum robust correlation of each component time-series with a model of 72 realigment parameters.

	Parameters
	---------------------------------------------------------------------------------
	melmix:		Full path of the melodic_mix text file
	mc:		Full path of the text file containing the realignment parameters
	
	Returns
	---------------------------------------------------------------------------------
	maxRPcorr:	Array of the maximum RP correlation feature scores for the components of the melodic_mix file"""

	# Import required modules
	import numpy as np
	import random

	# Read melodic mix file (IC time-series), subsequently define a set of squared time-series
	mix = np.loadtxt(melmix)
	mixsq = np.power(mix,2)

	# Read motion parameter file
	RP6 = np.loadtxt(mc)

	# Determine the derivatives of the RPs (add zeros at time-point zero)
	RP6_der = np.array(RP6[list(range(1,RP6.shape[0])),:] - RP6[list(range(0,RP6.shape[0]-1)),:])
	RP6_der = np.concatenate((np.zeros((1,6)),RP6_der),axis=0)

	# Create an RP-model including the RPs and its derivatives
	RP12 = np.concatenate((RP6,RP6_der),axis=1)

	# Add the squared RP-terms to the model
	RP24 = np.concatenate((RP12,np.power(RP12,2)),axis=1)

	# Derive shifted versions of the RP_model (1 frame for and backwards)
	RP24_1fw = np.concatenate((np.zeros((1,24)),np.array(RP24[list(range(0,RP24.shape[0]-1)),:])),axis=0)
	RP24_1bw = np.concatenate((np.array(RP24[list(range(1,RP24.shape[0])),:]),np.zeros((1,24))),axis=0)

	# Combine the original and shifted mot_pars into a single model
	RP_model = np.concatenate((RP24,RP24_1fw,RP24_1bw),axis=1)

	# Define the column indices of respectively the squared or non-squared terms
	idx_nonsq = np.array(np.concatenate((list(range(0,12)), list(range(24,36)), list(range(48,60))),axis=0))
	idx_sq = np.array(np.concatenate((list(range(12,24)), list(range(36,48)), list(range(60,72))),axis=0))

	# Determine the maximum correlation between RPs and IC time-series
	nSplits=int(1000)
	maxTC = np.zeros((nSplits,mix.shape[1]))
	for i in range(0,nSplits):
		# Get a random set of 90% of the dataset and get associated RP model and IC time-series matrices
		idx = np.array(random.sample(list(range(0,mix.shape[0])),int(round(0.9*mix.shape[0]))))
		RP_model_temp = RP_model[idx,:]
		mix_temp = mix[idx,:]
		mixsq_temp = mixsq[idx,:]

		# Calculate correlation between non-squared RP/IC time-series
		RP_model_nonsq = RP_model_temp[:,idx_nonsq]
		cor_nonsq = np.array(np.zeros((mix_temp.shape[1],RP_model_nonsq.shape[1])))
		for j in range(0,mix_temp.shape[1]):
			for k in range(0,RP_model_nonsq.shape[1]):
				cor_temp = np.corrcoef(mix_temp[:,j],RP_model_nonsq[:,k])
				cor_nonsq[j,k] = cor_temp[0,1]

		# Calculate correlation between squared RP/IC time-series
		RP_model_sq = RP_model_temp[:,idx_sq]
		cor_sq = np.array(np.zeros((mix_temp.shape[1],RP_model_sq.shape[1])))
		for j in range(0,mixsq_temp.shape[1]):
			for k in range(0,RP_model_sq.shape[1]):
				cor_temp = np.corrcoef(mixsq_temp[:,j],RP_model_sq[:,k])
				cor_sq[j,k] = cor_temp[0,1]

		# Combine the squared an non-squared correlation matrices
		corMatrix = np.concatenate((cor_sq,cor_nonsq),axis=1)

		# Get maximum absolute temporal correlation for every IC
		corMatrixAbs = np.abs(corMatrix)
		maxTC[i,:] = corMatrixAbs.max(axis=1)

	# Get the mean maximum correlation over all random splits
	maxRPcorr = maxTC.mean(axis=0)

	# Return the feature score
	return maxRPcorr

def feature_frequency(melFTmix, TR):
	""" This function extracts the high-frequency content feature scores. It determines the frequency, as fraction of the Nyquist frequency, at which the higher and lower frequencies explain half of the total power between 0.01Hz and Nyquist. 
	
	Parameters
	---------------------------------------------------------------------------------
	melFTmix:	Full path of the melodic_FTmix text file
	TR:		TR (in seconds) of the fMRI data (float)
	
	Returns
	---------------------------------------------------------------------------------
	HFC:		Array of the HFC ('High-frequency content') feature scores for the components of the melodic_FTmix file"""

	# Import required modules
	import numpy as np

	# Determine sample frequency
	Fs = old_div(1,TR)

	# Determine Nyquist-frequency
	Ny = old_div(Fs,2)
		
	# Load melodic_FTmix file
	FT=np.loadtxt(melFTmix)

	# Determine which frequencies are associated with every row in the melodic_FTmix file  (assuming the rows range from 0Hz to Nyquist)
	f = Ny*(np.array(list(range(1,FT.shape[0]+1))))/(FT.shape[0])

	# Only include frequencies higher than 0.01Hz
	fincl = np.squeeze(np.array(np.where( f > 0.01 )))
	FT=FT[fincl,:]
	f=f[fincl]

	# Set frequency range to [0-1]
	f_norm = old_div((f-0.01),(Ny-0.01))

	# For every IC; get the cumulative sum as a fraction of the total sum
	fcumsum_fract = old_div(np.cumsum(FT,axis=0), np.sum(FT,axis=0))

	# Determine the index of the frequency with the fractional cumulative sum closest to 0.5
	idx_cutoff=np.argmin(np.abs(fcumsum_fract-0.5),axis=0)

	# Now get the fractions associated with those indices index, these are the final feature scores
	HFC = f_norm[idx_cutoff]
		 
	# Return feature score
	return HFC

def feature_spatial(fslDir, tempDir, aromaDir, melIC):
	""" This function extracts the spatial feature scores. For each IC it determines the fraction of the mixture modeled thresholded Z-maps respecitvely located within the CSF or at the brain edges, using predefined standardized masks.

	Parameters
	---------------------------------------------------------------------------------
	fslDir:		Full path of the bin-directory of FSL
	tempDir:	Full path of a directory where temporary files can be stored (called 'temp_IC.nii.gz')
	aromaDir:	Full path of the ICA-AROMA directory, containing the mask-files (mask_edge.nii.gz, mask_csf.nii.gz & mask_out.nii.gz) 
	melIC:		Full path of the nii.gz file containing mixture-modeled threholded (p>0.5) Z-maps, registered to the MNI152 2mm template
	
	Returns
	---------------------------------------------------------------------------------
	edgeFract:	Array of the edge fraction feature scores for the components of the melIC file
	csfFract:	Array of the CSF fraction feature scores for the components of the melIC file"""

	# Import required modules
	import numpy as np
	import os
	import subprocess

	# Get the number of ICs
	numICs = int(subprocess.getoutput('%sfslinfo %s | grep dim4 | head -n1 | awk \'{print $2}\'' % (fslDir, melIC) ))

	# Loop over ICs
	edgeFract=np.zeros(numICs)
	csfFract=np.zeros(numICs)
	for i in range(0,numICs):
		# Define temporary IC-file
		tempIC = os.path.join(tempDir,'temp_IC.nii.gz')

		# Extract IC from the merged melodic_IC_thr2MNI2mm file
		os.system(' '.join([os.path.join(fslDir,'fslroi'),
			melIC,
			tempIC,
			str(i),
			'1']))

		# Change to absolute Z-values
		os.system(' '.join([os.path.join(fslDir,'fslmaths'),
			tempIC,
			'-abs',
			tempIC]))
		
		# Get sum of Z-values within the total Z-map (calculate via the mean and number of non-zero voxels)
		totVox = int(subprocess.getoutput(' '.join([os.path.join(fslDir,'fslstats'),
							tempIC,
							'-V | awk \'{print $1}\''])))
		
		if not (totVox == 0):
			totMean = float(subprocess.getoutput(' '.join([os.path.join(fslDir,'fslstats'),
							tempIC,
							'-M'])))
		else:
			print('     - The spatial map of component ' + str(i+1) + ' is empty. Please check!')
			totMean = 0

		totSum = totMean * totVox
		
		# Get sum of Z-values of the voxels located within the CSF (calculate via the mean and number of non-zero voxels)
		csfVox = int(subprocess.getoutput(' '.join([os.path.join(fslDir,'fslstats'),
							tempIC,
							'-k mask_csf.nii.gz',
							'-V | awk \'{print $1}\''])))

		if not (csfVox == 0):
			csfMean = float(subprocess.getoutput(' '.join([os.path.join(fslDir,'fslstats'),
							tempIC,
							'-k mask_csf.nii.gz',
							'-M'])))
		else:
			csfMean = 0

		csfSum = csfMean * csfVox	

		# Get sum of Z-values of the voxels located within the Edge (calculate via the mean and number of non-zero voxels)
		edgeVox = int(subprocess.getoutput(' '.join([os.path.join(fslDir,'fslstats'),
							tempIC,
							'-k mask_edge.nii.gz',
							'-V | awk \'{print $1}\''])))
		if not (edgeVox == 0):
			edgeMean = float(subprocess.getoutput(' '.join([os.path.join(fslDir,'fslstats'),
							tempIC,
							'-k mask_edge.nii.gz',
							'-M'])))
		else:
			edgeMean = 0
		
		edgeSum = edgeMean * edgeVox

		# Get sum of Z-values of the voxels located outside the brain (calculate via the mean and number of non-zero voxels)
		outVox = int(subprocess.getoutput(' '.join([os.path.join(fslDir,'fslstats'),
							tempIC,
							'-k mask_out.nii.gz',
							'-V | awk \'{print $1}\''])))
		if not (outVox == 0):
			outMean = float(subprocess.getoutput(' '.join([os.path.join(fslDir,'fslstats'),
							tempIC,
							'-k mask_out.nii.gz',
							'-M'])))
		else:
			outMean = 0
		
		outSum = outMean * outVox

		# Determine edge and CSF fraction
		if not (totSum == 0):
			edgeFract[i] = old_div((outSum + edgeSum),(totSum - csfSum))
			csfFract[i] = old_div(csfSum, totSum)
		else:
			edgeFract[i]=0
			csfFract[i]=0

	# Remove the temporary IC-file
	os.remove(tempIC)

	# Return feature scores
	return edgeFract, csfFract

def classification(outDir, maxRPcorr, edgeFract, HFC, csfFract):
	""" This function classifies a set of components into motion and non-motion components based on four features; maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

	Parameters
	---------------------------------------------------------------------------------
	outDir:		Full path of the output directory
	maxRPcorr:	Array of the 'maximum RP correlation' feature scores of the components
	edgeFract:	Array of the 'edge fraction' feature scores of the components
	HFC:		Array of the 'high-frequency content' feature scores of the components
	csfFract:	Array of the 'CSF fraction' feature scores of the components

	Return
	---------------------------------------------------------------------------------
	motionICs	Array containing the indices of the components identified as motion components

	Output (within the requested output directory)
	---------------------------------------------------------------------------------
	classified_motion_ICs.txt	A text file containing the indices of the components identified as motion components """

	# Import required modules
	import numpy as np
	import os
	import subprocess

	# Classify the ICs as motion or non-motion

	# Define criteria needed for classification (thresholds and hyperplane-parameters)
	thr_csf = 0.10
	thr_HFC = 0.35
	hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])
	
	# Project edge & maxRPcorr feature scores to new 1D space
	x = np.array([maxRPcorr, edgeFract])
	proj = hyp[0] + np.dot(x.T,hyp[1:])

	# Classify the ICs
	motionICs = np.squeeze(np.array(np.where((proj > 0) + (csfFract > thr_csf) + (HFC > thr_HFC))))

	# Put the feature scores in a text file
	np.savetxt(os.path.join(outDir,'feature_scores.txt'),np.vstack((maxRPcorr,edgeFract,HFC,csfFract)).T)

	# Put the indices of motion-classified ICs in a text file
	txt = open(os.path.join(outDir,'classified_motion_ICs.txt'),'w')
	if len(motionICs) != 0:
		txt.write(','.join(['%.0f' % num for num in (motionICs+1)]))
	txt.close()

	# Create a summary overview of the classification
	txt = open(os.path.join(outDir,'classification_overview.txt'),'w')
	txt.write('IC' + '\t' +  'Motion/noise' + '\t' +  'maximum RP correlation' + '\t' +  'Edge-fraction' + '\t\t' +  'High-frequency content' + '\t' + 'CSF-fraction')
	txt.write('\n')
	for i in range(0,len(csfFract)):
		if (proj[i] > 0) or (csfFract[i] > thr_csf) or (HFC[i] > thr_HFC):
			classif="True"
		else:
			classif="False"
		txt.write('%.0f\t%s\t\t%.2f\t\t\t%.2f\t\t\t%.2f\t\t\t%.2f\n' % (i+1, classif, maxRPcorr[i], edgeFract[i], HFC[i], csfFract[i]))
	txt.close()

	return motionICs

def denoising(fslDir, inFile, outDir, melmix, denType, denIdx):
	""" This function classifies the ICs based on the four features; maximum RP correlation, high-frequency content, edge-fraction and CSF-fraction

	Parameters
	---------------------------------------------------------------------------------
	fslDir:		Full path of the bin-directory of FSL
	inFile:		Full path to the data file (nii.gz) which has to be denoised
	outDir:		Full path of the output directory
	melmix:		Full path of the melodic_mix text file
	denType:	Type of requested denoising ('aggr': aggressive, 'nonaggr': non-aggressive, 'both': both aggressive and non-aggressive 
	denIdx:		Indices of the components that should be regressed out

	Output (within the requested output directory)
	---------------------------------------------------------------------------------
	denoised_func_data_<denType>.nii.gz:		A nii.gz file of the denoised fMRI data"""

	# Import required modules
	import os
	import numpy as np

	# Check if denoising is needed (i.e. are there components classified as motion)
	check = len(denIdx) > 0

	if check==1:
		# Put IC indices into a char array
		denIdxStr = np.char.mod('%i',(denIdx+1))

		# Non-aggressive denoising of the data using fsl_regfilt (partial regression), if requested
		if (denType == 'nonaggr') or (denType == 'both'):		
			os.system(' '.join([os.path.join(fslDir,'fsl_regfilt'),
				'--in=' + inFile,
				'--design=' + melmix,
				'--filter="' + ','.join(denIdxStr) + '"',
				'--out=' + os.path.join(outDir,'denoised_func_data_nonaggr.nii.gz')]))

		# Aggressive denoising of the data using fsl_regfilt (full regression)
		if (denType == 'aggr') or (denType == 'both'):
			os.system(' '.join([os.path.join(fslDir,'fsl_regfilt'),
				'--in=' + inFile,
				'--design=' + melmix,
				'--filter="' + ','.join(denIdxStr) + '"',
				'--out=' + os.path.join(outDir,'denoised_func_data_aggr.nii.gz'),
				'-a']))
	else:
		print("  - None of the components was classified as motion, so no denoising is applied (a symbolic link to the input file will be created).")
		if (denType == 'nonaggr') or (denType == 'both'):
			os.symlink(inFile,os.path.join(outDir,'denoised_func_data_nonaggr.nii.gz'))
		if (denType == 'aggr') or (denType == 'both'):
			os.symlink(inFile,os.path.join(outDir,'denoised_func_data_aggr.nii.gz'))
