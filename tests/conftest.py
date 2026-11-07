import pytest
from synthetic import write_raw_dir


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    """A small synthetic raw-data folder: 8 records from 4 vehicles."""
    return write_raw_dir(tmp_path_factory.mktemp("raw"))
