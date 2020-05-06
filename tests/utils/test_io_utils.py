import tempfile
from pathlib import Path
import pytest
from yaml.scanner import ScannerError

from src.utils.io_utils import load_config, get_googly_filepath, GOOGLY_DIR


def test_load_config_fail():
    # Given an illegal yaml file on a temporary directory
    data = "illegal_yml: very : illegal"
    filename = "config.yml"
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / filename
        p.write_text(data)
        # When we attempt to load the yaml file, then we get an exception (ScannerError)
        with pytest.raises(ScannerError):
            load_config(tmp_dir)


def test_load_config_missing():
    # Given a temporary directory without a config file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # When we attempt to load the yaml file, then we get an exception (FileNotFoundError)
        with pytest.raises(FileNotFoundError):
            load_config(tmp_dir)


def test_load_config_success():
    # Given a fake yaml file
    filename = "config.yml"
    data = "test: 0"
    # When we attempt to load the yaml file
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / filename
        p.write_text(data)
        result = load_config(tmp_dir)
        # Then we obtain the correct dictionary
        assert result == {"test": 0}


def test_get_googly_filepath_simple(mocker):
    # Given a filename
    filename = "picture.png"
    mocker.patch('os.makedirs')
    # When we get the filepath for the googlified picture
    googly_filepath = get_googly_filepath(filename)
    # Then we get the expected filepath
    assert googly_filepath == GOOGLY_DIR + "/picture_googlified.png"


def test_get_googly_filepath_new_directory(mocker):
    # Given a filename and a specific directory to upload
    filename, config_dir = "picture.png", "extra"
    mock_mkdir = mocker.patch('os.makedirs')
    # When we get the filepath for the googlified picture
    googly_filepath = get_googly_filepath(filename, config_dir)
    # Then we get the expected filepath and the directory's created
    assert googly_filepath == config_dir + "/picture_googlified.png"
    mock_mkdir.assert_called_once()
