#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from mlstm_kernels.utils.test.fixtures import test_session_folder  # noqa
from mlstm_kernels.utils.test.fixtures import test_output_folder  # noqa


import pytest

combinations_long = {
    "S": [256],
    "B": [1],
    "NH": [2],
    "DHQK": [64],
    "DHHV": [128],
}
combinations_long_list = [values for values in zip(*combinations_long.values())]

final_combinations = combinations_long_list

combinations_other = {
    "S":    [256, 256, 256, 256, 256, 256, 256],
    "B":    [4, 2, 4, 1, 2, 2, 1],
    "NH":   [2, 4, 8, 2, 4, 2, 2],
    "DHQK": [64, 32, 16, 48, 256, 24, 256],
    "DHHV": [128, 64, 32, 96, 512, 48, 256],
}
combinations_other_list = [values for values in zip(*combinations_other.values())]


pytest.short_test = False

