# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import numpy as np

import xija
import timbre


model_init = {'aacccdpt': {'aacccdpt': -7., 'aca0': -7., 'eclipse': False},
              'pftank2t': {'pftank2t': timbre.f_to_c(95.), 'pf0tank2t': timbre.f_to_c(95.), 'eclipse': False},
              'tcylaft6': {'tcylaft6': timbre.f_to_c(120.), 'cc0': timbre.f_to_c(120.), 'eclipse': False},
              '4rt700t': {'4rt700t': timbre.f_to_c(95.), 'oba0': timbre.f_to_c(95.), 'eclipse': False},
              '1dpamzt': {'1dpamzt': 35., 'dpa0': 35., 'eclipse': False, 'vid_board': True, 'clocking': True,
                          'dpa_power': 0.0, 'sim_z': 100000},
              '1deamzt': {'1deamzt': 35., 'dea0': 35., 'eclipse': False, 'vid_board': True, 'clocking': True,
                          'dpa_power': 0.0, 'sim_z': 100000}}

root = Path(__file__).parents[0]
aca_model_spec, aca_md5 = timbre.get_local_model(Path(root, 'data', 'aca_spec.json'))


def test_c_to_f():
    """ Test Celsius to Fahrenheit function.
    """
    f = timbre.c_to_f(100)
    assert f == 212.0


def test_f_to_c():
    """ Test Celsius to Fahrenheit function.
    """
    c = timbre.f_to_c(212)
    assert c == 100.0


# def test_get_full_dtype():
#     """ Test boilerplate dtype generation.
#     """
#
#     d = timbre.get_full_dtype({})
#
#     assert isinstance(np.dtype(d), np.dtype)


def test_setup_model():
    """ Test the model setup/initialization method `setup_model`.
    """
    model = timbre.setup_model('aacccdpt', '2030:001:00:00:00', '2031:001:00:00:00', aca_model_spec,
                               model_init['aacccdpt'])

    assert isinstance(model, xija.XijaModel)


def test_find_second_dwell():
    """ Test the method, `find_second_dwell`, which is used to find the dwell time needed to balance t_dwell1.
    """
    date = '2021:001:00:00:00'
    t_dwell1 = 20000.
    msid = 'aacccdpt'
    limit = -7.1
    dwell1_state = {'pitch': 90.2}
    dwell2_state = {'pitch': 148.95}

    answer = 58648.0

    results = timbre.find_second_dwell(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, aca_model_spec,
                                       model_init[msid], limit_type='max')
    assert isinstance(results, dict)
    assert np.isclose(results['dwell_2_time'], answer, atol=1.0e3)


def test_find_second_dwell_with_bounds():
    """ Test the method, `find_second_dwell`, which is used to find the dwell time needed to balance t_dwell1.
    """
    date = '2021:001:00:00:00'
    t_dwell1 = 20000.
    msid = 'aacccdpt'
    limit = -7.1
    dwell1_state = {'pitch': 90.2}
    dwell2_state = {'pitch': 148.95}

    answer = 58648.0

    results = timbre.find_second_dwell(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, aca_model_spec,
                                       model_init[msid], min_dwell=30000, max_dwell=80000, limit_type='max')
    assert np.isclose(results['dwell_2_time'], answer, atol=1.0e3)

    results_fail = timbre.find_second_dwell(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, aca_model_spec,
                                            model_init[msid], min_dwell=10000, max_dwell=30000, limit_type='max')
    assert results_fail['unconverged_hot'] is True

    results_fail = timbre.find_second_dwell(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, aca_model_spec,
                                            model_init[msid], min_dwell=90000, max_dwell=100000, limit_type='max')
    assert results_fail['unconverged_cold'] is True


def test_run_state_pairs():
    """ Test the method, `run_state_pairs`, which is used as a helper function to run a number of cases.
    """
    date = '2021:001:00:00:00'
    t_dwell1 = 20000.
    msid = 'aacccdpt'
    limit = -7.1

    state_pairs = (({'pitch': 144.2}, {'pitch': 154.95}),
                   ({'pitch': 90.2}, {'pitch': 148.95}),
                   ({'pitch': 50}, {'pitch': 140}),
                   ({'pitch': 90}, {'pitch': 100}),
                   ({'pitch': 75}, {'pitch': 130}),
                   ({'pitch': 170}, {'pitch': 90}),
                   ({'pitch': 90}, {'pitch': 170}))

    results = timbre.run_state_pairs(msid, aca_model_spec, model_init[msid], limit, date, t_dwell1, state_pairs)

    assert isinstance(results, np.ndarray)


def test_generate_balanced_pitch_dwells():

    # msid = 'aacccdpt'
    date = '2021:001:00:00:00'
    limit = -7.1
    aca_constant_conditions = {}
    anchor_offset_pitch = 160
    anchor_limited_pitch = 90
    anchor_offset_time = 20000
    pitch_range = np.arange(45, 181, 5)

    aca = timbre.BalanceAACCCDPT(date,
                                 aca_model_spec,
                                 limit,
                                 aca_constant_conditions)

    aca.find_anchor_condition(anchor_offset_pitch, anchor_limited_pitch, anchor_offset_time, limit)
    aca.results = aca.generate_balanced_pitch_dwells(aca.datesecs,
                                                     anchor_limited_pitch,
                                                     aca.anchor_limited_time,
                                                     anchor_offset_pitch,
                                                     anchor_offset_time,
                                                     aca.limit,
                                                     pitch_range)

    median_limited_dwell = np.median(timbre.get_limited_results(aca.results, anchor_offset_pitch)['t_dwell2'])

    assert median_limited_dwell > 20000
    assert median_limited_dwell < 50000
