from distutils.core import setup
from os.path import join

setup(
    name='aroma',
    description='ICA-AROMA',
    version='0.4',
    packages=['icaaroma'],
    scripts=[join('bin', 'aroma')],
    package_data={'icaaroma': ['mask_*.nii.gz']}
)
