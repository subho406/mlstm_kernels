import pytest
from pathlib import Path
from datetime import datetime

@pytest.fixture(scope="session")
def test_session_folder() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return Path(__file__).parents[1] / "tests_outputs" / timestamp