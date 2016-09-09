from distutils.core import setup
from os.path import join

setup(
    name='aroma',
    description='ICA-AROMA',
    version='0.4',    
    scripts = [join('bin', 'aroma')]
)
