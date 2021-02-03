# Licensed under a 3-clause BSD style license - see LICENSE.rst
import ska_helpers

from .timbre import load_model_specs, get_full_dtype, get_local_model, c_to_f, f_to_c, setup_model, run_profile
from .timbre import calc_binary_schedule, create_opt_fun, find_second_dwell, run_state_pairs
from .timbre import *

__version__ = ska_helpers.get_version(__package__)


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)
