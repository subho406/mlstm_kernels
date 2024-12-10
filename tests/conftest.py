#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from mlstm_kernels.utils.test.fixtures import test_session_folder  # noqa
from mlstm_kernels.utils.test.fixtures import test_output_folder  # noqa


import pytest

combinations_long = {
    "S": [256],  # [8192],
    "B": [1],  # [2, 2, 2, 2],
    "NH": [2],  # [3, 3, 3, 3],
    "DHQK": [64],  # [5, 5, 5, 5],
    "DHHV": [128],  # [5, 5, 5, 5],
}
combinations_long_list = [values for values in zip(*combinations_long.values())]

final_combinations = combinations_long_list
