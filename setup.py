from setuptools import setup
from os.path import join, dirname

def readme(fname):
    with open(join(dirname(__file__), fname)) as f:
        text = f.read()
    return text

setup(
    name='aroma',
    description='ICA-AROMA',
    long_description=readme('README.md'),
    author="R. Pruim, Ronald Hartley-Davies",
    author_email="r.pruim@donders.ru.nl, rtrhd@bristol.ac.uk",
    version='0.4',
    license='Apache Software License',
    url="http://github.com/rtrhd/icaaroma",
    keywords="ics fmri motion",
    packages=['icaaroma'],
    entry_points={'console_scripts':['aroma = icaaroma.aroma:main']},
    package_data={'icaaroma': ['data/mask_*.nii.gz']},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ]
)
