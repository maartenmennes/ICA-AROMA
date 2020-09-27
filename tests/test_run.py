import ica_aroma
from ica_aroma.ica_aroma import ica_aroma_cli
import pytest
import os
import re

REFIN = os.path.join(os.path.dirname(__file__), "refin")

def setup_function():
    """clear environ every test"""
    os.environ["AROMADIR"] = ""


def test_find_data():
    """get_data_path has an assert that will error if path DNE"""
    ica_aroma.get_data_path()


@pytest.mark.xfail(reason="bad *_mask.nii.gz path")
def test_bad_data_path():
    """intentionally point to a bad path"""
    os.environ["AROMADIR"] = "/path/to/no/where"
    ica_aroma.get_data_path()


def test_no_arg_usage(capsys):
    """does main even run"""
    try:
        ica_aroma_cli()
    except SystemExit as exit_code:
        assert exit_code.code == 2
    else:
        pytest.fail("no exit code on ica_aroma without args!?")

    _, help_text = capsys.readouterr()
    assert re.match(r"usage: ", help_text)


@pytest.mark.skipif(not os.path.isdir(REFIN), reason="need test-data {REFIN}")
def test_run():
    """does main even run"""

    # https://github.com/rtrhd/test-data/raw/master/icaaroma/0.4.0/refin.tar.bz2
    args = [
        "-o", "/tmp/icaaroma",
        "-i", f"{REFIN}/filtered_func_data.nii.gz",
        "-m", f"{REFIN}/mask.nii.gz",
        "-mc", f"{REFIN}/mc/prefiltered_func_data_mcf.par",
        "-tr", "2.0",
        "-overwrite",
    ]

    ica_aroma_cli(["ica_aroma", *args])
