import pytest
from pathlib import Path
from datetime import datetime
import logging
import sys

@pytest.fixture(scope="session")
def test_session_folder() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    test_output_folder = Path(__file__).parents[1] / "tests_outputs" / timestamp

    test_output_folder.mkdir(parents=True, exist_ok=True)

    logfile = test_output_folder / "pytest.log"
    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[file_handler],
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
        force=True
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info(f"Logging to {logfile}")
    return test_output_folder