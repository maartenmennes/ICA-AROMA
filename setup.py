from setuptools import setup
from os.path import join, dirname

def readme(fname):
    with open(join(dirname(__file__), fname)) as f:
        text = f.read()
    return text


setup(
    name='icaaroma',
    description='ICA-based Automatic Removal Of Motion Artefacts',
    long_description=readme('README.md'),
    author="Maarten Mennes",
    author_email="m.mennes@donders.ru.nl",
    version='0.4.0',
    license='Apache Software License',
    url="http://github.com/rtrhd/icaaroma",
    keywords="ics fmri motion",
    packages=['ica_aroma'],
    install_requires=['nibabel>=1.3.0', 'matplotlib>=2.2', 'numpy>=1.14',
                      'pandas>=0.23', 'seaborn>=0.9.0'],
    tests_require=['pytest'],
    entry_points={'console_scripts': ['ica_aroma = ica_aroma.ica_aroma:ica_aroma_cli']},
    package_data={'icaaroma': ['data/mask_*.nii.gz']},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ]
)
