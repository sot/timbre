# Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy
import numpy as np
import pandas as pd
from os import path

from cxotime import CxoTime

from .timbre import run_state_pairs, find_second_dwell, load_model_specs


DEFAULT_ANCHORS = {
    '1dpamzt': {'anchor_limited_pitch': 155, 'anchor_offset_pitch': 70},
    '1deamzt': {'anchor_limited_pitch': 155, 'anchor_offset_pitch': 70},
    'fptemp_11': {'anchor_limited_pitch': 170, 'anchor_offset_pitch': 70},
    '1pdeaat': {'anchor_limited_pitch': 45, 'anchor_offset_pitch': 160},
    'aacccdpt': {'anchor_limited_pitch': 90, 'anchor_offset_pitch': 160},
    'pm1thv2t': {'anchor_limited_pitch': 60, 'anchor_offset_pitch': 160},
    'pm2thv1t': {'anchor_limited_pitch': 60, 'anchor_offset_pitch': 160},
    '4rt700t': {'anchor_limited_pitch': 90, 'anchor_offset_pitch': 160},
    'pftank2t': {'anchor_limited_pitch': 60, 'anchor_offset_pitch': 160},
    'pline03t': {'anchor_limited_pitch': 175, 'anchor_offset_pitch': 70},
    'pline04t': {'anchor_limited_pitch': 175, 'anchor_offset_pitch': 70},
    '2ceahvpt': {'anchor_limited_pitch': 150, 'anchor_offset_pitch': 90},
}


def get_limited_results(results, anchor_offset_pitch):
    """ Extract the limited dwell time results.

    :param results: Results calculated by `generate_balanced_pitch_dwells`
    :type results: numpy.ndarray
    :param anchor_offset_pitch: offset pitch used to anchor the limited dwell curve
    :type anchor_offset_pitch: int or float
    :returns: Numpy array of limited dwell time results
    :rtype: numpy.ndarray

    Note: The returned array is a structured numpy array with column names defined by Timbre and the input conditions.

    """

    if results is not None:
        ind = results['converged'] & (results['pitch1'] == anchor_offset_pitch) & \
              ~np.isnan(results['t_dwell2'])
        return results[ind]
    else:
        return None


def get_offset_results(results, anchor_limited_pitch):
    """ Extract the offset dwell time results.

    :param results: Results calculated by `generate_balanced_pitch_dwells`
    :type results: numpy.ndarray
    :param anchor_limited_pitch: limited pitch used to anchor the offset dwell curve
    :type anchor_limited_pitch: int or float
    :returns: Numpy array of offset dwell time results
    :rtype: numpy.ndarray

    Note: The returned array is a structured numpy array with column names defined by Timbre and the input conditions.

    """
    if results is not None:
        ind = results['converged'] & (results['pitch1'] == anchor_limited_pitch) & \
              ~np.isnan(results['t_dwell2'])
        return results[ind]
    else:
        return None


class Balance(object):
    """ Base class for defining individual thermal balance calculation objects

    """

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor, offset_conditions=None,
                 limited_conditions=None, maneuvers=False):
        """ Run a Xija model for a given time and state profile.

        :param date: Date used for the dwell balance analysis
        :type date: str
        :param model_spec: Dictionary of model parameters for a single model
        :type model_spec: dict
        :param limit: Thermal Limit
        :type limit: float
        :param constant_conditions: Dictionary of conditions that remain constant for balance analysis, any required
            parameters not included as constant_conditions must be included in the dwell state 1 and state 2 inputs.
            This does not necessarily include initial conditions for primary and pseudo nodes.
        :type constant_conditions: dict
        :param margin_factor: Knockdown/safety factor to reduce predicted available limited dwell time, intended to add
            some conservatism to the ACIS maximum dwell time predictions, which are used to determine the available
            cooling time for models that heat at forward and normal sun attitudes
        :type margin_factor: float
        :param offset_conditions: Dictionary of conditions that only occur during offset dwells
        :type offset_conditions: dict or None, optional
        :param limited_conditions: Dictionary of conditions that only occur during limiting dwells
        :type limited_conditions: dict or None, optional
        :param maneuvers: Boolean indicating whether to consider maneuver time
        :type maneuvers: bool, optional

        """

        if offset_conditions is None:
            offset_conditions = {}

        if limited_conditions is None:
            limited_conditions = {}

        self.date = CxoTime(date).date
        self.datesecs = CxoTime(date).secs
        self.model_spec = model_spec
        self.limit = limit
        self.constant_conditions = constant_conditions
        self.margin_factor = margin_factor
        self.anchor_offset_time = np.nan
        self.anchor_limited_time = np.nan
        self.offset_conditions = offset_conditions
        self.limited_conditions = limited_conditions
        self.maneuvers = maneuvers
        self.results = None

    def find_anchor_condition(self, p_offset, p_limited, anchor_offset_time, limit):
        """ Given a known offset duration, determine the maximum dwell time for the anchor limited pitch.

        :param p_offset: Anchor offset pitch
        :type p_offset: float
        :param p_limited: Anchor limited pitch
        :type p_limited: float
        :param anchor_offset_time: Anchor offset dwell duration (seconds)
        :type anchor_offset_time: float
        :param limit: Thermal limit
        :type limit: float

        """

        # Define dwell states that are used to determine max cooling dwell time
        #
        # Note: self.constant_conditions does not include initial conditions for the primary and pseudo nodes.
        dwell1_state = {**{'pitch': p_offset}, **self.constant_conditions, **self.offset_conditions}
        dwell2_state = {**{'pitch': p_limited}, **self.constant_conditions, **self.limited_conditions}

        # Find maximum hot time (including margin safety factor)
        dwell_results = find_second_dwell(self.date, dwell1_state, dwell2_state, anchor_offset_time, self.msid,
                                          limit, self.model_spec, self.model_init, limit_type=self.limit_type,
                                          n_dwells=30, maneuvers=self.maneuvers)

        self.anchor_offset_time = anchor_offset_time
        self.anchor_limited_time = dwell_results['dwell_2_time'] * self.margin_factor

    def generate_balanced_pitch_dwells(self, datesecs, anchor_limited_pitch, t_1, anchor_offset_pitch, t_2_orig, limit,
                                       pitch_range):
        """ Calculate the balanced heating and cooling dwell curves seeded by `anchor_limited_pitch` and
        `anchor_offset_pitch`.

        :param datesecs: Date used for simulation (seconds)
        :type datesecs: float
        :param anchor_limited_pitch: Anchor limited pitch
        :type anchor_limited_pitch: float
        :param t_1: Anchor limited dwell duration (seconds)
        :type t_1: float
        :param anchor_offset_pitch: Anchor offset pitch
        :type anchor_offset_pitch: float
        :param t_2_orig: Originally calculated anchor offset dwell duration (seconds)
        :type t_2_orig: float
        :param limit: Thermal limit
        :type limit: float
        :param pitch_range: Pitch values used to define dwell limits
        :type pitch_range: iterable (list, numpy.ndarray)

        A balanced set of dwell curves are calculated in two steps:
            1) Calculate the offset dwell curve yielded by `anchor_limited_pitch` at `t_1`.
            2) Use the time calculated in step 1 for `anchor_offset_pitch` with this pitch to determine the limited
               dwell curve.

        """

        # Expand dwell capability curve yielded by time #1 at pitch #1 (e.g. anchor hot time)
        state_pairs = list(({**{'pitch': anchor_limited_pitch}, **self.constant_conditions, **self.limited_conditions},
                            {**{'pitch': p2}, **self.constant_conditions, **self.offset_conditions})
                           for p2 in pitch_range)

        if np.isnan(t_1):
            msg1 = f'{self.msid} is not limited at a pitch of {anchor_limited_pitch} degrees near ' \
                   f'{CxoTime(datesecs).date}, with the following constant conditions:\n{self.constant_conditions},\n' \
                   f'with the following limited conditions:\n{self.limited_conditions}\n' \
                   f'with the following offset conditions:\n{self.offset_conditions}\n' \
                   f'a limit of {limit}, an anchor limited duration of {t_1} seconds, and an anchor offset duration' \
                   f' of {t_2_orig} seconds.\n'
            print(msg1)
            return None

        args = (self.msid, self.model_spec, self.model_init, limit, datesecs, t_1, state_pairs)
        kwargs = {'limit_type': self.limit_type, 'print_progress': False, 'n_dwells': 30, 'maneuvers':self.maneuvers}
        results1 = run_state_pairs(*args, **kwargs)

        # This deals with a weird case where the anchor limited time@pitch does not reproduce a converged solution at
        # the originally calculated  offset time at the offset pitch. I encountered this issue with the pftank2t model
        # with an edge case that initially had some long dwell times before refining the composite dwell curve (as a
        # part of the algorithm).
        ind = results1['pitch2'] == anchor_offset_pitch
        if np.isnan(results1[ind]['t_dwell2']):
            results1['t_dwell2'] = t_2_orig

        # Expand dwell capability curve yielded by pitch #2 at the associated time calculated above.
        # This will include but expand upon the original time #1 at pitch #1 passed to this method.
        t_2_ind = results1['pitch2'] == anchor_offset_pitch
        t_2 = results1['t_dwell2'][t_2_ind].item()
        state_pairs = list(({**{'pitch': anchor_offset_pitch}, **self.constant_conditions, **self.offset_conditions},
                            {**{'pitch': p2}, **self.constant_conditions, **self.limited_conditions})
                           for p2 in pitch_range)

        args = (self.msid, self.model_spec, self.model_init, limit, datesecs, t_2, state_pairs)
        results2 = run_state_pairs(*args, **kwargs)

        results = np.hstack((results1, results2))

        not_nan = np.where(~np.isnan(results['t_dwell2']))[0]
        if any(not_nan):
            inds = np.where(results[not_nan]['t_dwell2'] < 0)[0]
            if any(inds):
                for ind in inds:
                    ind = not_nan[ind].item()
                    results[ind]['t_dwell2'] = np.nan

        not_nan = np.where(~np.isnan(results['t_dwell2']))[0]
        if any(not_nan):
            inds = np.where(results[not_nan]['t_dwell2'] > 300000)[0]
            if any(inds):
                for ind in inds:
                    ind = not_nan[ind].item()
                    results[ind]['t_dwell2'] = np.nan

        return results


class Balance2CEAHVPT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=0.95, custom_offset_conditions=None,
                 custom_limited_conditions=None, imaging_detector=True, spectroscopy_detector=False, maneuvers=False):

        assert ~(imaging_detector and spectroscopy_detector), 'Imaging and spectroscopy detectors cannot both be True'

        if custom_offset_conditions is None:
            custom_offset_conditions = {}

        if custom_limited_conditions is None:
            custom_limited_conditions = {}

        self.msid = '2ceahvpt'
        self.model_init = {'2ceahvpt': limit, 'cea0': limit, 'cea1': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.limit_type = 'max'

        limited_conditions = {
            '2ps5aon_on': True,
            '2ps5bon_on': False,
            '2imonst_on': imaging_detector,
            '2sponst_on': spectroscopy_detector,
            '2s2onst_on': True,
            '224pcast_off': False,
            '215pcast_off': True,
            'ccd_count': 2,
            'fep_count': 2,
            'clocking': True,
            'vid_board': True,
            'sim_z': -99616
        }
        limited_conditions.update(custom_limited_conditions)

        # FEP and CCD counts are updated with the custom_offset_conditions dictionary.
        offset_conditions = {
            '2ps5aon_on': True,
            '2ps5bon_on': False,
            '2imonst_on': False,
            '2sponst_on': False,
            '2s2onst_on': False,
            '224pcast_off': False,
            '215pcast_off': False,
            'ccd_count': 4,
            'fep_count': 4,
            'clocking': True,
            'vid_board': True,
            'sim_z': 100000
        }
        offset_conditions.update(custom_offset_conditions)

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor,
                         offset_conditions=offset_conditions, limited_conditions=limited_conditions,
                         maneuvers=maneuvers)


class Balance1DPAMZT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=0.95, maneuvers=False):
        self.msid = '1dpamzt'
        self.model_init = {'1dpamzt': limit, 'dpa0': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class Balance1DEAMZT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=0.95, maneuvers=False):
        self.msid = '1deamzt'
        self.model_init = {'1deamzt': limit, 'dea0': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class Balance1PDEAAT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0, maneuvers=False):
        self.msid = '1pdeaat'
        self.model_init = {'1pdeaat': limit, 'pin1at': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class BalanceFPTEMP_11(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=0.95, maneuvers=False):
        self.msid = 'fptemp'
        self.model_init = {'fptemp': limit, '1cbat': -55.0, 'sim_px': 110.0, 'eclipse': False, 'dpa_power': 0.0,
                           'orbitephem0_x': 125000e3, 'orbitephem0_y': 125000e3, 'orbitephem0_z': 125000e3,
                           'solarephem0_x': 2.6e10, 'solarephem0_y': -1.3e11, 'solarephem0_z': -5.7e10,
                           'aoattqt1': 0.0, 'aoattqt2': 0.0, 'aoattqt3': 0.0, 'aoattqt4': 1.0, 'dh_heater': False,
                           '215pcast_off': False}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class BalanceAACCCDPT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0, maneuvers=False):
        self.msid = 'aacccdpt'
        self.model_init = {'aacccdpt': limit, 'aca0': limit, 'eclipse': False}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class Balance4RT700T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0, maneuvers=False):
        self.msid = '4rt700t'
        self.model_init = {'4rt700t': limit, 'oba0': limit, 'eclipse': False}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class BalancePFTANK2T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0, maneuvers=False):
        self.msid = 'pftank2t'
        self.model_init = {'pftank2t': limit, 'pf0tank2t': limit, 'eclipse': False, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class BalancePM1THV2T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0, maneuvers=False):
        self.msid = 'pm1thv2t'
        self.model_init = {'pm1thv2t': limit, 'mups0': limit, 'eclipse': False, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class BalancePM2THV1T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0, maneuvers=False):
        self.msid = 'pm2thv1t'
        self.model_init = {'pm2thv1t': limit, 'mups0': limit, 'mups1': limit, 'eclipse': False, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class BalancePLINE04T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0, maneuvers=False):
        self.msid = 'pline04t'
        self.model_init = {'pline04t': limit, 'pline04t0': limit, 'eclipse': False, }
        self.limit_type = 'min'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class BalancePLINE03T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0, maneuvers=False):
        self.msid = 'pline03t'
        self.model_init = {'pline03t': limit, 'pline03t0': limit, 'eclipse': False, }
        self.limit_type = 'min'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor, maneuvers=maneuvers)


class Composite(object):
    """ Generate a composite maximum dwell estimate using all thermal models.

    This class can be used to automatically calculate the maximum best-case dwell duration possible for each model at
    each limited pitch (often heating) by considering the impact each model has on available cooling for all other
    models. This class also produces the associated offset (often cooling) dwell durations that can support the limited
    dwell durations produced for each model.

    Example::

        from timbre import Composite, f_to_c

        limits = {
            "fptemp_11": -112.0,
            "1dpamzt": 37.5,
            "1deamzt": 37.5,
            "1pdeaat": 52.5,
            "aacccdpt": -6.5,
            "4rt700t": f_to_c(100.0),
            "pftank2t": f_to_c(110.0),
            "pm1thv2t": f_to_c(210.0),
            "pm2thv1t": f_to_c(210.0),
            "pline03t": f_to_c(50.0),
            "pline04t": f_to_c(50.0),
        }

        date = "2022:001:00:00:00"
        chips = 4
        roll = 0

        timbre_object = Composite(date, chips, roll, limits, max_dwell=200000, pitch_step=5)

    """

    def __init__(self, date, chips, roll, limits, max_dwell=100000, pitch_step=1, model_specs=None, anchors=None,
                 maneuvers=False, execute=True):
        """ Run a Xija model for a given time and state profile.

        :param date: Date used for the dwell balance analysis
        :type date: str
        :param chips: Number of ACIS chips used, this is used to set both the number of FEPs and CCDs
        :type chips: int
        :param roll: Body roll
        :type roll: float or int
        :param limits: Dictionary of MSID limits
        :type limits: dict
        :param max_dwell: Initial guess for maximum available cooling
        :type max_dwell: int or float
        :param model_specs: Dictionary of model parameters, if None then the `timbre.load_model_specs` method will be
            used to load all model specs.
        :type model_specs: dict or None, optional
        :param anchors: Dictionary of limited and offset anchor pitch values for each MSID.
        :type anchors" dict or None, optional
        :param maneuvers: Boolean indicating whether to consider maneuver time
        :type maneuvers: bool, optional
        :param execute: Boolean indicating whether to run the composite balance calculations upon object instantiation
        :type execute: bool, optional

        """

        self.date = CxoTime(date).date
        self.datesecs = CxoTime(date).secs
        self.chips = chips
        self.roll = roll
        self.limits = limits
        self.max_dwell = max_dwell

        if model_specs is None:
            model_specs = load_model_specs()
        self.model_specs = model_specs

        if anchors is None:
            self.anchors = DEFAULT_ANCHORS
        else:
            self.anchors = anchors

        # Start with only the pitch values used to map out dwell capability (from the "anchors" defined above).
        self.pitch_range = self._get_required_pitch_values()
        tail_pitches = self._get_required_pitch_values()
        self.tail_pitches = tail_pitches[tail_pitches > 130]
        self.non_tail_pitches = tail_pitches[tail_pitches <= 130]

        if not set(self.pitch_range).issubset(set(range(45, 181, pitch_step))):
            raise ValueError(f'Pitch step must be defined to include all anchor pitch values: {self.pitch_range}')

        self.msids = list(self.anchors.keys())
        self.limited_results = pd.DataFrame(index=range(45, 181, pitch_step), columns=self.msids)
        self.offset_results = pd.DataFrame(index=range(45, 181, pitch_step), columns=self.msids)
        self.dwell_limits = pd.Series(index=range(45, 181, pitch_step))
        self.dwell_limits.loc[:] = self.max_dwell

        acis_on_const = {'roll': roll, 'fep_count': chips, 'ccd_count': chips, 'clocking': True, 'vid_board': True,
                         'sim_z': 100000}
        acis_off_stowed_const = {'roll': roll, 'fep_count': 3, 'ccd_count': 0, 'clocking': False, 'vid_board': False,
                                 'sim_z': -99616}
        psmc_on_const = {'roll': roll, 'fep_count': chips, 'ccd_count': chips, 'clocking': True, 'vid_board': True,
                         'sim_z': 100000, 'dh_heater': False}
        psmc_off_stowed_const = {'roll': roll, 'fep_count': 3, 'ccd_count': 0, 'clocking': False, 'vid_board': False,
                                 'sim_z': -99616, 'dh_heater': False}

        sc_const = {'roll': roll}

        if self.limits['fptemp_11'] < -99.0:
            self.dpa = Balance1DPAMZT(self.date, self.model_specs['1dpamzt'], self.limits['1dpamzt'], acis_on_const,
                                      maneuvers=maneuvers)
            self.dea = Balance1DEAMZT(self.date, self.model_specs['1deamzt'], self.limits['1deamzt'], acis_on_const,
                                      maneuvers=maneuvers)
            self.acisfp = BalanceFPTEMP_11(self.date, self.model_specs['fptemp_11'], self.limits['fptemp_11'],
                                           acis_on_const, maneuvers=maneuvers)
            self.psmc = Balance1PDEAAT(self.date, self.model_specs['1pdeaat'], self.limits['1pdeaat'], psmc_on_const,
                                       maneuvers=maneuvers)

        else:
            self.dpa = Balance1DPAMZT(self.date, self.model_specs['1dpamzt'], self.limits['1dpamzt'],
                                      acis_off_stowed_const, maneuvers=maneuvers)
            self.dea = Balance1DEAMZT(self.date, self.model_specs['1deamzt'], self.limits['1deamzt'],
                                      acis_off_stowed_const, maneuvers=maneuvers)
            self.acisfp = BalanceFPTEMP_11(self.date, self.model_specs['fptemp_11'], self.limits['fptemp_11'],
                                           acis_off_stowed_const, maneuvers=maneuvers)
            self.psmc = Balance1PDEAAT(self.date, self.model_specs['1pdeaat'], self.limits['1pdeaat'],
                                       psmc_off_stowed_const, maneuvers=maneuvers)

        self.aca = BalanceAACCCDPT(self.date, self.model_specs['aacccdpt'], self.limits['aacccdpt'], {},
                                   maneuvers=maneuvers)
        self.oba = Balance4RT700T(self.date, self.model_specs['4rt700t'], self.limits['4rt700t'], {},
                                  maneuvers=maneuvers)
        self.mups1b = BalancePM1THV2T(self.date, self.model_specs['pm1thv2t'], self.limits['pm1thv2t'], sc_const,
                                      maneuvers=maneuvers)
        self.mups2a = BalancePM2THV1T(self.date, self.model_specs['pm2thv1t'], self.limits['pm2thv1t'], sc_const,
                                      maneuvers=maneuvers)
        self.tank = BalancePFTANK2T(self.date, self.model_specs['pftank2t'], self.limits['pftank2t'], sc_const,
                                    maneuvers=maneuvers)
        self.pline03t = BalancePLINE03T(self.date, self.model_specs['pline03t'], self.limits['pline03t'], sc_const,
                                        maneuvers=maneuvers)
        self.pline04t = BalancePLINE04T(self.date, self.model_specs['pline04t'], self.limits['pline04t'], sc_const,
                                        maneuvers=maneuvers)

        if execute is True:
            dashes = ''.join(["-", ] * 120)
            print(f'{dashes}\nMap Dwell Capability\n{dashes}')
            self.map_composite()

            print(f'{dashes}\nFill In Dwell Capability\n{dashes}\n')
            self.fill_composite()

    def _get_required_pitch_values(self):
        """ Get the set of all anchor limited and offset pitch values.
        """
        return np.unique([p for msid, pitches in self.anchors.items() for p in list(pitches.values())])

    def balance_model(self, msid, model):
        """ Calculate the balanced limited and offset dwell curves for one model.
        """
        anchor_offset_pitch = self.anchors[msid]['anchor_offset_pitch']
        anchor_limited_pitch = self.anchors[msid]['anchor_limited_pitch']

        anchor_offset_time = self.dwell_limits.loc[self.anchors[msid]['anchor_offset_pitch']]

        model.find_anchor_condition(anchor_offset_pitch, anchor_limited_pitch, anchor_offset_time, self.limits[msid])
        model.results = model.generate_balanced_pitch_dwells(model.datesecs,
                                                             anchor_limited_pitch,
                                                             model.anchor_limited_time,
                                                             anchor_offset_pitch,
                                                             anchor_offset_time,
                                                             self.limits[msid],
                                                             self.pitch_range)

        limited_results = get_limited_results(model.results, anchor_offset_pitch)
        offset_results = get_offset_results(model.results, anchor_limited_pitch)

        if limited_results is not None:
            pitch = limited_results['pitch2']
            t_dwell2 = limited_results['t_dwell2']
            self.limited_results.loc[pitch, msid] = t_dwell2
            self.dwell_limits.loc[pitch] = np.minimum(t_dwell2, self.dwell_limits.loc[pitch])

        if offset_results is not None:
            pitch = offset_results['pitch2']
            t_dwell2 = offset_results['t_dwell2']
            self.offset_results.loc[pitch, msid] = t_dwell2

    def map_composite(self):
        """ Use the anchor pitch values to map the fundamental dwell balance data.
        """

        dashes = ''.join(["-", ] * 40)
        print(f'\n{dashes}\nStart of Iteration:\n')

        # Note the initial time used for cooling ACIS models sensitive to tail sun heating
        initial_dwell_limits = copy.copy(self.dwell_limits.loc[self.pitch_range])

        self.balance_model('1dpamzt', self.dpa)
        self.balance_model('1deamzt', self.dea)
        self.balance_model('fptemp_11', self.acisfp)

        # Note the tail sun time being used for cooling non ACIS models (except PLINES)
        tail_sun_dwell_limits = copy.copy(self.dwell_limits.loc[self.pitch_range])

        self.balance_model('1pdeaat', self.psmc)
        self.balance_model('aacccdpt', self.aca)
        self.balance_model('4rt700t', self.oba)
        self.balance_model('pm1thv2t', self.mups1b)
        self.balance_model('pm2thv1t', self.mups2a)
        self.balance_model('pftank2t', self.tank)
        self.balance_model('pline03t', self.pline03t)
        self.balance_model('pline04t', self.pline04t)

        s = ''.join([f'    {p:>3}    {d:6.2f}\n' for p, d in self.dwell_limits.loc[self.pitch_range].items()])
        print(f'Approximate dwell limits calculated by this iteration: \n  Pitch    Duration\n{s}')

        final_dwell_limits = copy.copy(self.dwell_limits.loc[self.pitch_range])
        rerun_tail = False
        # Check to see if the pline dwell balances reduced tail sun cooling time for normal and fwd sun heating models.
        for p in self.tail_pitches:
            final_tail_duration = final_dwell_limits.loc[p]
            initial_tail_duration = tail_sun_dwell_limits.loc[p]

            if final_tail_duration < initial_tail_duration:
                # self.dwell_limits.loc[p] = final_tail_duration * 0.99  # Avoid a potentially infinite loop
                rerun_tail = True

        rerun_fwd = False
        # Check to see if the initial assumed dwell time for ACIS models exceeds final available dwell time.
        for p in self.non_tail_pitches:
            final_non_tail_duration = final_dwell_limits[p]
            initial_non_tail_duration = initial_dwell_limits.loc[p]

            if final_non_tail_duration < initial_non_tail_duration:
                rerun_fwd = True

        if rerun_tail is True or rerun_fwd is True:
            if rerun_tail is True:
                print('Tail sun time available for cooling is less than originally assumed at the start of this'
                      ' iteration, start new iteration.')
            if rerun_fwd is True:
                print('Forward sun time available for cooling is less than originally assumed at the start of this'
                      ' iteration, start new iteration.')
            self.map_composite()

    def fill_composite(self):
        """ Fill in the limited and offset dwell curves after mapping the fundamental dwell balance data.
        """
        self.pitch_range = np.array(self.dwell_limits.index)

        self.balance_model('1dpamzt', self.dpa)
        self.balance_model('1deamzt', self.dea)
        self.balance_model('fptemp_11', self.acisfp)
        self.balance_model('1pdeaat', self.psmc)
        self.balance_model('aacccdpt', self.aca)
        self.balance_model('4rt700t', self.oba)
        self.balance_model('pm1thv2t', self.mups1b)
        self.balance_model('pm2thv1t', self.mups2a)
        self.balance_model('pftank2t', self.tank)
        self.balance_model('pline03t', self.pline03t)
        self.balance_model('pline04t', self.pline04t)

        dashes = ''.join(["-", ] * 120)
        s = ''.join([f'    {p:>3}    {d:6.2f}\n' for p, d in self.dwell_limits.loc[self.pitch_range].items()])
        print(f'{dashes}\nFinal Dwell Limits: \n  Pitch    Duration\n{s}\n')



def get_constant_condition_dictionaries(roll, chips):
    """Generate condition dictionaries for different ACIS and spacecraft states.
    
    :param roll: spacecraft roll angle
    :type roll: float
    :param chips: number of ACIS chips
    :type chips: int
    :returns: Dictionary containing all condition dictionaries
    :rtype: dict
    """
    conditions = {
        'acis_on': {
            'roll': roll,
            'fep_count': chips,
            'ccd_count': chips,
            'clocking': True,
            'vid_board': True,
            'sim_z': 100000
        },
        'acis_off_stowed': {
            'roll': roll,
            'fep_count': 3,
            'ccd_count': 0,
            'clocking': False,
            'vid_board': False,
            'sim_z': -99616
        },
        'psmc_on': {
            'roll': roll,
            'fep_count': chips,
            'ccd_count': chips,
            'clocking': True,
            'vid_board': True,
            'sim_z': 100000,
            'dh_heater': False
        },
        'psmc_off_stowed': {
            'roll': roll,
            'fep_count': 3,
            'ccd_count': 0,
            'clocking': False,
            'vid_board': False,
            'sim_z': -99616,
            'dh_heater': False
        },
        'sc': {
            'roll': roll
        },
        'empty': {}
    }
    
    return conditions


def get_msid_constant_conditions(conditions, fptemp_limit):
    """Map MSIDs to their appropriate condition dictionaries based on FPTEMP limit.
    
    :param conditions: Dictionary of condition dictionaries from get_condition_dictionaries()
    :type conditions: dict
    :param fptemp_limit: FPTEMP limit value
    :type fptemp_limit: float
    :returns: Dictionary mapping MSIDs to their condition dictionaries
    :rtype: dict
    """
    if fptemp_limit < -99.0:
        msid_conditions = {
            '1dpamzt': conditions['acis_on'],
            '1deamzt': conditions['acis_on'],
            'fptemp_11': conditions['acis_on'],
            '1pdeaat': conditions['psmc_on']
        }
    else:
        msid_conditions = {
            '1dpamzt': conditions['acis_off_stowed'],
            '1deamzt': conditions['acis_off_stowed'], 
            'fptemp_11': conditions['acis_off_stowed'],
            '1pdeaat': conditions['psmc_off_stowed']
        }

    # Add spacecraft MSIDs
    msid_conditions.update({
        'aacccdpt': conditions['empty'],
        '4rt700t': conditions['empty'],
        'pm1thv2t': conditions['sc'],
        'pm2thv1t': conditions['sc'],
        'pftank2t': conditions['sc'],
        'pline03t': conditions['sc'],
        'pline04t': conditions['sc']
    })

    return msid_conditions


# def scale_dwells_single_process(results_file, anchors=DEFAULT_ANCHORS, pitch_range=None, model_specs=None):
#     """ Scale dwell durations to determine appropriate ratio of limited to offset dwell time for shorter dwells.
    
#     :param results_file: Path to results CSV file containing dwell time data
#     :type results_file: str
#     :returns: DataFrame containing scaled results for each unique case and MSID
#     :rtype: pandas.DataFrame
#     """

#     if pitch_range is None:
#         pitch_range = np.arange(45, 181, 1)

#     if model_specs is None:
#         home = path.expanduser('~')
#         model_specs = load_model_specs(local_repository_location=home + '/chandra_models/xija/')

#     # Read the results file 
#     df = pd.read_csv(results_file)
#         # Columns that define unique cases
#     case_cols = ['fptemp_11_limit', '1dpamzt_limit', '1deamzt_limit', '1pdeaat_limit',
#                 'aacccdpt_limit', '4rt700t_limit', 'pftank2t_limit', 'pm1thv2t_limit',
#                 'pm2thv1t_limit', 'pline03t_limit', 'pline04t_limit', 'date', 'datesecs',
#                 'roll', 'chips']
    
#     index_cols = case_cols + ['pitch', 'dwell_type']

#     msids = ['1dpamzt', '1deamzt', 'fptemp_11', '1pdeaat', 'aacccdpt', 'pm1thv2t', 'pm2thv1t', '4rt700t', 'pftank2t', 
#              'pline03t', 'pline04t']


#     # Set index for input results DataFrame
#     indexed_df = df.set_index(index_cols)
    
#     # Map of MSID to Balance class
#     balance_classes = {
#         '1dpamzt': Balance1DPAMZT,
#         '1deamzt': Balance1DEAMZT,
#         'fptemp_11': BalanceFPTEMP_11,
#         '1pdeaat': Balance1PDEAAT,
#         'aacccdpt': BalanceAACCCDPT,
#         'pm1thv2t': BalancePM1THV2T,
#         'pm2thv1t': BalancePM2THV1T,
#         '4rt700t': Balance4RT700T,
#         'pftank2t': BalancePFTANK2T,
#         'pline03t': BalancePLINE03T,
#         'pline04t': BalancePLINE04T
#     }

#     # Scale factors to test
#     scale_factors = [0.25, 0.5, 0.75, 1.0]

#     # Columns for results
#     results_cols = [f'scale_factor_results_{sf}' for sf in scale_factors]

#     # Create a new index that adds the msid to the index, and shifts the index so the last two columns are dwell_type and pitch
#     old_index = pd.MultiIndex.from_arrays([df[col] for col in index_cols])
#     new_tuples = [(*idx_tuple, m) for idx_tuple in old_index for m in msids]
#     new_index = pd.MultiIndex.from_tuples(new_tuples, names=index_cols + ['msid', ])

#     # Create a DataFrame with case columns as index and values columns for pitch, composite limit, and scale factors
#     results_df = pd.DataFrame(index=new_index, columns=results_cols)
#     results_df.sort_index(inplace=True)

#     truncated_index = results_df.index.droplevel(["pitch", "dwell_type", "msid"]).unique()

#     num_cases = len(truncated_index)

#     for n, truncated_case_index in enumerate(truncated_index):

#         case_dict = dict(zip(case_cols, truncated_case_index))

#         # if you want to create a function to outsource this code, you can do so by passing in the following arguments:
#         # truncated_index, case_dict, balance_classes, msids, scale_factors, anchors, indexed_df, 

#         if n % 100 == 0:
#             print(f"Running case: {n} of {num_cases}")

#         roll = case_dict['roll']
#         chips = case_dict['chips']
#         fptemp_11_limit = case_dict['fptemp_11_limit']
#         baseline_constant_conditions = get_constant_condition_dictionaries(roll, chips)
#         all_msid_constant_conditions = get_msid_constant_conditions(baseline_constant_conditions, fptemp_11_limit)

#         for msid in msids:
#             constant_conditions = all_msid_constant_conditions[msid]

#             limit = case_dict[f'{msid}_limit']
#             date = case_dict['date']

#             balance_class = balance_classes[msid]
#             balance_obj = balance_class(
#                 date=date,
#                 model_spec=model_specs[msid],
#                 limit=limit,
#                 constant_conditions=constant_conditions,
#                 margin_factor=1.0
#             )

#             anchor_limited_pitch = anchors[msid]['anchor_limited_pitch']
#             anchor_offset_pitch = anchors[msid]['anchor_offset_pitch']

#             # These are the composite dwell limits at the anchor pitches
#             offset_time = indexed_df.loc[truncated_case_index + (anchor_offset_pitch, 'offset')].min()
#             limited_time = indexed_df.loc[truncated_case_index + (anchor_limited_pitch, 'limit')].min()

#             if pd.isna(offset_time) or pd.isna(limited_time):
#                 continue

#             for scale in scale_factors:
#                 scaled_offset_anchor_time = offset_time * scale
#                 scaled_limited_anchor_time = limited_time * scale

#                 scale_factor_name = f'scale_factor_results_{scale}'

#                 results_df.loc[truncated_case_index + (anchor_offset_pitch, 'offset', msid), scale_factor_name] = scaled_offset_anchor_time

#                 if scale == 1.0:
#                     msid_limited_times = indexed_df.loc[truncated_case_index + (slice(None), 'limit'), msid]
#                     results_df.loc[truncated_case_index + (slice(None), 'limit', msid), scale_factor_name] = msid_limited_times

#                     msid_offset_times = indexed_df.loc[truncated_case_index + (slice(None), 'offset'), msid]
#                     results_df.loc[truncated_case_index + (slice(None), 'offset', msid), scale_factor_name] = msid_offset_times

#                     continue

#                 # Recalculate limited dwell time using scaled offset time
#                 dwell1_state = {**{'pitch': anchor_offset_pitch}, **constant_conditions, 
#                                 **balance_obj.offset_conditions}
#                 dwell2_state = {**{'pitch': anchor_limited_pitch}, **constant_conditions, 
#                                 **balance_obj.limited_conditions}
                
#                 dwell_results = find_second_dwell(
#                     balance_obj.date, 
#                     dwell1_state, 
#                     dwell2_state, 
#                     scaled_offset_anchor_time,
#                     balance_obj.msid,
#                     balance_obj.limit,
#                     balance_obj.model_spec,
#                     balance_obj.model_init,
#                     limit_type=balance_obj.limit_type,
#                     n_dwells=30,
#                     maneuvers=balance_obj.maneuvers
#                 )

#                 if dwell_results is not None:
#                     # Fill in the dwell times for the rest of the pitch range

#                     results_df.loc[truncated_case_index + (anchor_limited_pitch, 'limit', msid), scale_factor_name] = dwell_results['dwell_2_time']

#                     # Fill in the rest of the pitch range for limited dwells
#                     limited_fill_results, p = fill_pitch_range_dwells(
#                         balance_obj,
#                         pitch_range,
#                         anchor_offset_pitch,
#                         scaled_offset_anchor_time,
#                         fill_dwell_type='limit'
#                     )
#                     results_df.loc[truncated_case_index + (p, 'limit', msid), scale_factor_name] = limited_fill_results['t_dwell2']

#                     # Fill in the rest of the pitch range for offset dwells
#                     offset_fill_results, p = fill_pitch_range_dwells(
#                         balance_obj,
#                         pitch_range,
#                         anchor_limited_pitch,
#                         scaled_limited_anchor_time,
#                         fill_dwell_type='offset'
#                     )
#                     results_df.loc[truncated_case_index + (p, 'offset', msid), scale_factor_name] = offset_fill_results['t_dwell2']   

#     return results_df


def fill_pitch_range_dwells(balance_obj, pitch_range, anchor_pitch, anchor_time, fill_dwell_type='limit'):
    """Fill in dwell times for remaining pitch values in the range.
    
    :param balance_obj: Balance object instance containing model parameters
    :type balance_obj: Balance
    :param pitch_range: Array of pitch values to fill
    :type pitch_range: numpy.ndarray
    :param anchor_pitch: Pitch value used as anchor point
    :type anchor_pitch: float
    :param anchor_time: Dwell time at anchor pitch
    :type anchor_time: float
    :param fill_dwell_type: Type of dwell to fill, either 'limit' or 'offset'
    :type fill_dwell_type: str
    :returns: Array of dwell time results for each pitch
    :rtype: numpy.ndarray
    """
    # Remove anchor pitch from range to fill
    p = pitch_range # [pitch_range != anchor_pitch]
    
    # Set up state pairs based on whether we're filling limited or offset dwells
    # The only reason these are separate is because the offset conditions could be different from the limited conditions,
    # depending on the MSID.
    if fill_dwell_type == 'limit':
        state_pairs = list(
            (
                {**{'pitch': anchor_pitch}, **balance_obj.constant_conditions, **balance_obj.offset_conditions},
                {**{'pitch': p2}, **balance_obj.constant_conditions, **balance_obj.limited_conditions}
            )
            for p2 in p)
    else:
        state_pairs = list(
            (
                {**{'pitch': anchor_pitch}, **balance_obj.constant_conditions, **balance_obj.limited_conditions},
                {**{'pitch': p2}, **balance_obj.constant_conditions, **balance_obj.offset_conditions}
            )
            for p2 in p)
    
    # Run the model to get dwell times
    fill_results = run_state_pairs(
        balance_obj.msid,
        balance_obj.model_spec,
        balance_obj.model_init,
        balance_obj.limit,
        balance_obj.date,
        anchor_time,
        state_pairs,
        limit_type=balance_obj.limit_type,
        n_dwells=30,
        maneuvers=balance_obj.maneuvers,
        print_progress=False
    )
    
    return fill_results, p


def stack_inputs_for_scale_dwells_mp(
    results_file,
    anchors=DEFAULT_ANCHORS,
    pitch_range=None,
    model_specs=None,
    scale_factors=None,
    overrides=None,
    msids=None,
):
    """
    Load the data for scale_dwells from the results file.

    :param results_file: Path to results CSV file containing dwell time data
    :type results_file: str
    :param anchors: Dictionary of anchor pitch values for each MSID
    :type anchors: dict
    :param pitch_range: Array of pitch values to fill
    :type pitch_range: numpy.ndarray
    :param model_specs: Dictionary of model specifications for each MSID
    :type model_specs: dict
    :param scale_factors: List of scale factors to test
    :type scale_factors: list
    :param overrides: Dictionary of dwell time overrides for the default anchors
    :type overrides: dict
    :returns: List of dictionaries containing the inputs for each case
    :rtype: list

    """

    if pitch_range is None:
        pitch_range = np.arange(45, 181, 1)

    if model_specs is None:
        home = path.expanduser("~")
        model_specs = load_model_specs(
            local_repository_location=home
            + "/AXAFLIB/chandra_models/"
        )

    if scale_factors is None:
        scale_factors = [0.25, 0.5, 0.75, 1.0]

    if overrides is None:
        overrides = {}

    # Read the results file
    df = pd.read_csv(results_file)

    df.columns = df.columns.str.strip().str.lower()

    # Columns that define unique cases
    case_cols = [
        "fptemp_11_limit",
        "1dpamzt_limit",
        "1deamzt_limit",
        "1pdeaat_limit",
        "aacccdpt_limit",
        "4rt700t_limit",
        "pftank2t_limit",
        "pm1thv2t_limit",
        "pm2thv1t_limit",
        "pline03t_limit",
        "pline04t_limit",
        "date",
        "datesecs",
        "roll",
        "chips",
    ]

    index_cols = case_cols + ["pitch", "dwell_type"]

    if msids is None:   
        msids = [
            "1dpamzt",
            "1deamzt",
            "fptemp_11",
            "1pdeaat",
            "aacccdpt",
            "pm1thv2t",
            "pm2thv1t",
            "4rt700t",
            "pftank2t",
            "pline03t",
            "pline04t",
            ]

    # Set index for input results DataFrame
    indexed_df = df.set_index(index_cols)
    indexed_df.sort_index(inplace=True)

    # Map of MSID to Balance class
    balance_classes = {
        "1dpamzt": Balance1DPAMZT,
        "1deamzt": Balance1DEAMZT,
        "fptemp_11": BalanceFPTEMP_11,
        "1pdeaat": Balance1PDEAAT,
        "aacccdpt": BalanceAACCCDPT,
        "pm1thv2t": BalancePM1THV2T,
        "pm2thv1t": BalancePM2THV1T,
        "4rt700t": Balance4RT700T,
        "pftank2t": BalancePFTANK2T,
        "pline03t": BalancePLINE03T,
        "pline04t": BalancePLINE04T,
    }

    # Columns for results
    results_cols = [f"scale_factor_results_{sf}" for sf in scale_factors]

    # Create a new index that adds the msid to the index, and shifts the index so the last two columns are dwell_type and pitch
    old_index = pd.MultiIndex.from_arrays([df[col] for col in index_cols])
    new_tuples = [(*idx_tuple, m) for idx_tuple in old_index for m in msids]
    new_index = pd.MultiIndex.from_tuples(
        new_tuples,
        names=index_cols
        + [
            "msid",
        ],
    )

    # Create a DataFrame with case columns as index and values columns for pitch, composite limit, and scale factors
    #
    # For the multiprocessing version, this is only needed to provide the index for the results Dataframe, it should
    # probably be replaced with a more direct approach to creating the index for the results Dataframe.
    results_df = pd.DataFrame(index=new_index, columns=results_cols)
    results_df.sort_index(inplace=True)

    truncated_index = results_df.index.droplevel(
        ["pitch", "dwell_type", "msid"]
    ).unique()
    case_results_index = results_df.index.droplevel(truncated_index.names).unique()

    multiprocessing_inputs = []

    for n, truncated_case_index in enumerate(truncated_index):
        case_dict = dict(zip(case_cols, truncated_case_index))

        dwell_balance_inputs = {
            "case_dict": case_dict,
            "input_case_results": indexed_df.loc[truncated_case_index],
            "anchors": anchors,
            "pitch_range": pitch_range,
            "model_specs": model_specs,
            "scale_factors": scale_factors,
            "balance_classes": balance_classes,
            "msids": msids,
            "case_results_index": case_results_index,
            "case_results_cols": results_cols,
            "overrides": overrides,
            "id": n,
        }

        multiprocessing_inputs.append(dwell_balance_inputs)

    return multiprocessing_inputs


# create a separate version of scale_dwells that can be used in a multiprocessing pool
def scale_dwells_mp(inputs):
    case_dict = inputs["case_dict"]
    input_case_results = inputs["input_case_results"]
    anchors = inputs["anchors"]
    pitch_range = inputs["pitch_range"]
    model_specs = inputs["model_specs"]
    scale_factors = inputs["scale_factors"]
    balance_classes = inputs["balance_classes"]
    msids = inputs["msids"]
    case_results_index = inputs["case_results_index"]
    case_results_cols = inputs["case_results_cols"]
    overrides = inputs["overrides"]
    case_results = pd.DataFrame(index=case_results_index, columns=case_results_cols)

    roll = case_dict["roll"]
    chips = case_dict["chips"]
    date = case_dict["date"]
    fptemp_11_limit = case_dict["fptemp_11_limit"]
    baseline_constant_conditions = get_constant_condition_dictionaries(roll, chips)
    all_msid_constant_conditions = get_msid_constant_conditions(
        baseline_constant_conditions, fptemp_11_limit
    )

    for msid in msids:
        constant_conditions = all_msid_constant_conditions[msid]
        limit = case_dict[f"{msid}_limit"]
        balance_class = balance_classes[msid]
        balance_obj = balance_class(
            date=date,
            model_spec=model_specs[msid],
            limit=limit,
            constant_conditions=constant_conditions,
            margin_factor=1.0,
        )

        anchor_limited_pitch = anchors[msid]["anchor_limited_pitch"]
        anchor_offset_pitch = anchors[msid]["anchor_offset_pitch"]

        # These are the composite dwell limits at the anchor pitches, unless overridden
        offset_time = input_case_results.loc[(anchor_offset_pitch, "limit")].min() # shouldn't max offset time be limited by the limited time?
        limited_time = input_case_results.loc[(anchor_limited_pitch, "limit")].min()
        if msid in overrides:
            if overrides[msid]["offset_time"] is not None and overrides[msid]["limited_time"] is not None:
                raise ValueError(f"Both offset and limited time cannot be overridden for {msid}")
                continue

            if overrides[msid]["offset_time"] is not None:
                offset_time = overrides[msid]["offset_time"]

            elif overrides[msid]["limited_time"] is not None:
                limited_time = overrides[msid]["limited_time"]

        if pd.isna(offset_time) or pd.isna(limited_time):
            continue

        for scale in scale_factors:
            scaled_offset_anchor_time = offset_time * scale
            scaled_limited_anchor_time = limited_time * scale

            scale_factor_name = f"scale_factor_results_{scale}"

            start_condition = "offset"
            if msid in overrides and overrides[msid]["limited_time"] is not None:
                start_condition = "limit"

            if start_condition == "offset":
                case_results.loc[
                    (anchor_offset_pitch, "offset", msid), scale_factor_name
                ] = scaled_offset_anchor_time
            else:
                # else start with limited time
                case_results.loc[
                    (anchor_limited_pitch, "limit", msid), scale_factor_name
                ] = scaled_limited_anchor_time

            # If the scale factor is 1.0 and there are no overrides, then we can just use the input case results
            if scale == 1.0 and msid not in overrides:
                msid_limited_times = input_case_results.loc[
                    (slice(None), "limit"), msid
                ]
                case_results.loc[
                    (slice(None), "limit", msid), scale_factor_name
                ] = msid_limited_times

                msid_offset_times = input_case_results.loc[
                    (slice(None), "offset"), msid
                ]
                case_results.loc[
                    (slice(None), "offset", msid), scale_factor_name
                ] = msid_offset_times

                continue

            # Recalculate limited dwell time using scaled offset time
            dwell1_state = {
                **{"pitch": anchor_offset_pitch},
                **constant_conditions,
                **balance_obj.offset_conditions,
            }
            dwell2_state = {
                **{"pitch": anchor_limited_pitch},
                **constant_conditions,
                **balance_obj.limited_conditions,
            }
            if start_condition == "limited":
                dwell1_state, dwell2_state = dwell2_state, dwell1_state


            dwell_results = find_second_dwell(
                balance_obj.date,
                dwell1_state,
                dwell2_state,
                scaled_offset_anchor_time,
                balance_obj.msid,
                balance_obj.limit,
                balance_obj.model_spec,
                balance_obj.model_init,
                limit_type=balance_obj.limit_type,
                n_dwells=30,
                maneuvers=balance_obj.maneuvers,
            )

            if dwell_results is not None:
                # Fill in the dwell times for the rest of the pitch range

                # If the start condition is offset, then the dwell2 time is the limited dwell time
                if start_condition == "offset":
                    case_results.loc[
                        (anchor_limited_pitch, "limit", msid), scale_factor_name
                    ] = dwell_results["dwell_2_time"]

                    # Fill in the rest of the pitch range for limited dwells
                    limited_fill_results, p = fill_pitch_range_dwells(
                        balance_obj,
                        pitch_range,
                        anchor_offset_pitch,
                        scaled_offset_anchor_time,
                        fill_dwell_type="limit",
                    )
                    case_results.loc[
                        (p, "limit", msid), scale_factor_name
                    ] = limited_fill_results["t_dwell2"]

                    # The scaled_limited_anchor_time is recalculated in the fill_pitch_range_dwells function using the scaled_offset_anchor_time.
                    scaled_limited_anchor_time = case_results.loc[
                        (anchor_limited_pitch, "limit", msid), scale_factor_name
                    ]       

                    # Fill in the rest of the pitch range for offset dwells
                    offset_fill_results, p = fill_pitch_range_dwells(
                        balance_obj,
                        pitch_range,
                        anchor_limited_pitch,
                        scaled_limited_anchor_time,
                        fill_dwell_type="offset",
                    )
                    case_results.loc[
                        (p, "offset", msid), scale_factor_name
                    ] = offset_fill_results["t_dwell2"]

                # If the start condition is limit, then the dwell2 time is the offset dwell time
                else:
                    case_results.loc[
                        (anchor_offset_pitch, "offset", msid), scale_factor_name
                    ] = dwell_results["dwell_2_time"]

                    # Because the scaled_xxxxxx_anchor_time is defined above within the start_condition conditional check, 
                    # we need to perform the next two fill operations in the same order as the scaled times were defined.

                    # Fill in the rest of the pitch range for offset dwells
                    offset_fill_results, p = fill_pitch_range_dwells(
                        balance_obj,
                        pitch_range,
                        anchor_limited_pitch,
                        scaled_limited_anchor_time,
                        fill_dwell_type="offset",
                    )
                    case_results.loc[
                        (p, "offset", msid), scale_factor_name
                    ] = offset_fill_results["t_dwell2"]

                    # The scaled_offset_anchor_time is recalculated in the fill_pitch_range_dwells function using the scaled_limited_anchor_time
                    scaled_offset_anchor_time = case_results.loc[
                        (anchor_offset_pitch, "offset", msid), scale_factor_name
                    ]   

                    # Fill in the rest of the pitch range for limited dwells
                    limited_fill_results, p = fill_pitch_range_dwells(
                        balance_obj,
                        pitch_range,
                        anchor_offset_pitch,
                        scaled_offset_anchor_time,
                        fill_dwell_type="limit",
                    )
                    case_results.loc[
                        (p, "limit", msid), scale_factor_name
                    ] = limited_fill_results["t_dwell2"]

    return case_results
