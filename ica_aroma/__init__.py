import os
import ica_aroma

__version__ = "0.3-beta"


def get_data_path():
    """find directory location of
     mask_csf.nii.gz, mask_edge.nii.gz, and mask_out.nii.gz.

    will allow env var AROMADIR to overwrite package location
    @return path to data/ (string)
    """
    datadir = os.environ.get("AROMADIR", None) or \
        os.path.join(os.path.dirname(ica_aroma.__loader__.path), "data")

    assert os.path.isdir(datadir)
    assert os.path.isfile(os.path.join(datadir, 'mask_csf.nii.gz'))

    return datadir
