import pytest

from .template_test_parallel_interface import template_test_parallel_interface


@pytest.fixture
def mlstm_parallel_interface_test() -> callable:
    return template_test_parallel_interface
