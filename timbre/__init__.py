# Licensed under a 3-clause BSD style license - see LICENSE.rst
import ska_helpers

from .timbre import load_model_specs, get_local_model, c_to_f, f_to_c, setup_model, run_profile  # noqa
from .timbre import calc_binary_schedule, create_opt_fun, find_second_dwell, run_state_pairs  # noqa
from .timbre import load_github_model_specs, get_github_chandra_models_version_info # noqa
from .balance import Balance, Balance1DPAMZT, Balance1DEAMZT, Balance1PDEAAT, BalanceFPTEMP_11, BalanceAACCCDPT  # noqa
from .balance import Balance4RT700T, BalancePFTANK2T, BalancePM1THV2T, BalancePM2THV1T, BalancePLINE04T  # noqa
from .balance import BalancePLINE03T, Composite, get_limited_results, get_offset_results   # noqa
from .run_tools import add_inputs, run_instance, run_all_permutations, process_queue
__version__ = ska_helpers.get_version(__package__)


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)
