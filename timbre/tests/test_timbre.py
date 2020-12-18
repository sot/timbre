# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

import numpy as np

import xija
import timbre

root = os.path.dirname(__file__)

model_init = {'aacccdpt': {'aacccdpt': -7., 'aca0': -7., 'eclipse': False},
              'pftank2t': {'pftank2t': timbre.f_to_c(95.), 'pf0tank2t': timbre.f_to_c(95.), 'eclipse': False},
              'tcylaft6': {'tcylaft6': timbre.f_to_c(120.), 'cc0': timbre.f_to_c(120.), 'eclipse': False},
              '4rt700t': {'4rt700t': timbre.f_to_c(95.), 'oba0': timbre.f_to_c(95.), 'eclipse': False},
              '1dpamzt': {'1dpamzt': 35., 'dpa0': 35., 'eclipse': False, 'vid_board': True, 'clocking': True,
                          'dpa_power': 0.0, 'sim_z': 100000},
              '1deamzt': {'1deamzt': 35., 'dea0': 35., 'eclipse': False, 'vid_board': True, 'clocking': True,
                          'dpa_power': 0.0, 'sim_z': 100000}}

aca_model_spec_filename = os.path.join(root, 'data', 'aca_spec.json')
aca_model_spec, aca_md5 = timbre.get_local_model(aca_model_spec_filename)


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


def test_get_full_dtype():
    """ Test boilerplate dtype generation.
    """

    d = timbre.get_full_dtype({})

    assert isinstance(np.dtype(d), np.dtype)


def test_setup_model():
    model = timbre.setup_model('aacccdpt', '2030:001:00:00:00', '2031:001:00:00:00', aca_model_spec,
                               model_init['aacccdpt'])

    assert isinstance(model, xija.XijaModel)


def test_find_second_dwell():

    date = '2021:001:00:00:00'
    t_dwell1 = 20000.
    msid = 'aacccdpt'
    limit = -7.1
    dwell1_state = {'pitch': 90.2}
    dwell2_state = {'pitch': 148.95}

    results = timbre.find_second_dwell(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, aca_model_spec,
                                       model_init[msid], limit_type='max')
    assert isinstance(results, dict)
    assert np.isclose(results['dwell_2_time'], 65723.0, atol=1.0e3)


def test_run_state_pairs():

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

    state_pair_dtype = {'pitch': np.float64}

    results_dtype = timbre.get_full_dtype(state_pair_dtype)

    results = timbre.run_state_pairs(msid, aca_model_spec, model_init[msid], limit, date, t_dwell1, state_pairs,
                                     results_dtype)

    assert isinstance(results, np.ndarray)
