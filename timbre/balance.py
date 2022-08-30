# Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy

import numpy as np
import pandas as pd

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

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor):
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


        """

        self.date = CxoTime(date).date
        self.datesecs = CxoTime(date).secs
        self.model_spec = model_spec
        self.limit = limit
        self.constant_conditions = constant_conditions
        self.margin_factor = margin_factor
        self.anchor_offset_time = np.nan
        self.anchor_limited_time = np.nan
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
        dwell1_state = {**{'pitch': p_offset}, **self.constant_conditions}
        dwell2_state = {**{'pitch': p_limited}, **self.constant_conditions}

        # Find maximum hot time (including margin safety factor)
        dwell_results = find_second_dwell(self.date, dwell1_state, dwell2_state, anchor_offset_time, self.msid,
                                          limit, self.model_spec, self.model_init, limit_type=self.limit_type,
                                          n_dwells=30)

        self.anchor_offset_time = anchor_offset_time
        self.anchor_limited_time = dwell_results['dwell_2_time'] * self.margin_factor

    def generate_balanced_pitch_dwells(self, datesecs, pitch_1, t_1, pitch_2, t_2_orig, limit, pitch_range):
        """ Calculate the balanced heating and cooling dwell curves seeded by `pitch_1` and `pitch_2`.

        :param datesecs: Date used for simulation (seconds)
        :type datesecs: float
        :param pitch_1: Anchor limited pitch
        :type pitch_1: float
        :param t_1: Anchor limited dwell duration (seconds)
        :type t_1: float
        :param pitch_2: Anchor offset pitch
        :type pitch_2: float
        :param t_2_orig: Originally calculated anchor offset dwell duration (seconds)
        :type t_2_orig: float
        :param limit: Thermal limit
        :type limit: float
        :param pitch_range: Pitch values used to define dwell limits
        :type pitch_range: iterable (list, numpy.ndarray)

        A balanced set of dwell curves are calculated in two steps:
            1) Calculate the offset dwell curve yielded by `pitch_1` at `t_1`.
            2) Use the time calculated in step 1 for `pitch_2` with this pitch to determine the limited dwell curve.

        """

        # Expand dwell capability curve yielded by time #1 at pitch #1 (e.g. anchor hot time)
        state_pairs = list(({**{'pitch': pitch_1}, **self.constant_conditions},
                            {**{'pitch': p2}, **self.constant_conditions})
                           for p2 in pitch_range)

        if np.isnan(t_1):
            msg1 = f'{self.msid} is not limited at a pitch of {pitch_1} degrees near {CxoTime(datesecs).date},' \
                   f' with the following constant conditions:\n{self.constant_conditions}.\n'
            print(msg1)
            return None

        args = (self.msid, self.model_spec, self.model_init, limit, datesecs, t_1, state_pairs)
        kwargs = {'limit_type': self.limit_type, 'print_progress': False, 'n_dwells': 30}
        results1 = run_state_pairs(*args, **kwargs)

        # This deals with a weird case where the anchor limited time@pitch does not reproduce a converged solution at
        # the originally calculated  offset time at the offset pitch. I encountered this issue with the pftank2t model
        # with an edge case that initially had some long dwell times before refining the composite dwell curve (as a
        # part of the algorithm).
        ind = results1['pitch2'] == pitch_2
        if np.isnan(results1[ind]['t_dwell2']):
            results1['t_dwell2'] = t_2_orig

        # Expand dwell capability curve yielded by pitch #2 at the associated time calculated above.
        # This will include but expand upon the original time #1 at pitch #1 passed to this method.
        t_2_ind = results1['pitch2'] == pitch_2
        t_2 = results1['t_dwell2'][t_2_ind].item()
        state_pairs = list(({**{'pitch': pitch_2}, **self.constant_conditions},
                            {**{'pitch': p2}, **self.constant_conditions})
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


class Balance1DPAMZT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=0.95):
        self.msid = '1dpamzt'
        self.model_init = {'1dpamzt': limit, 'dpa0': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class Balance1DEAMZT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=0.95):
        self.msid = '1deamzt'
        self.model_init = {'1deamzt': limit, 'dea0': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class Balance1PDEAAT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0):
        self.msid = '1pdeaat'
        self.model_init = {'1pdeaat': limit, 'pin1at': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class BalanceFPTEMP_11(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=0.95):
        self.msid = 'fptemp'
        self.model_init = {'fptemp': limit, '1cbat': -55.0, 'sim_px': 110.0, 'eclipse': False, 'dpa_power': 0.0,
                           'orbitephem0_x': 125000e3, 'orbitephem0_y': 125000e3, 'orbitephem0_z': 125000e3,
                           'solarephem0_x': 2.6e10, 'solarephem0_y': -1.3e11, 'solarephem0_z': -5.7e10,
                           'aoattqt1': 0.0, 'aoattqt2': 0.0, 'aoattqt3': 0.0, 'aoattqt4': 1.0, 'dh_heater': False}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class BalanceAACCCDPT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0):
        self.msid = 'aacccdpt'
        self.model_init = {'aacccdpt': limit, 'aca0': limit, 'eclipse': False}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class Balance4RT700T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0):
        self.msid = '4rt700t'
        self.model_init = {'4rt700t': limit, 'oba0': limit, 'eclipse': False}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class BalancePFTANK2T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0):
        self.msid = 'pftank2t'
        self.model_init = {'pftank2t': limit, 'pf0tank2t': limit, 'eclipse': False, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class BalancePM1THV2T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0):
        self.msid = 'pm1thv2t'
        self.model_init = {'pm1thv2t': limit, 'mups0': limit, 'mups1': limit, 'eclipse': False, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class BalancePM2THV1T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0):
        self.msid = 'pm2thv1t'
        self.model_init = {'pm2thv1t': limit, 'mups0': limit * 10, 'eclipse': False, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class BalancePLINE04T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0):
        self.msid = 'pline04t'
        self.model_init = {'pline04t': limit, 'pline04t0': limit, 'eclipse': False, }
        self.limit_type = 'min'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


class BalancePLINE03T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, margin_factor=1.0):
        self.msid = 'pline03t'
        self.model_init = {'pline03t': limit, 'pline03t0': limit, 'eclipse': False, }
        self.limit_type = 'min'

        super().__init__(date, model_spec, limit, constant_conditions, margin_factor)


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

    def __init__(self, date, chips, roll, limits, max_dwell=100000, pitch_step=1, model_specs=None, anchors=None):
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
        acis_off_stowed_const = {'roll': roll, 'fep_count': 0, 'ccd_count': 0, 'clocking': False, 'vid_board': False,
                                 'sim_z': -99616}
        psmc_on_const = {'roll': roll, 'fep_count': chips, 'ccd_count': chips, 'clocking': True, 'vid_board': True,
                         'sim_z': 100000, 'dh_heater': False}
        psmc_off_stowed_const = {'roll': roll, 'fep_count': 0, 'ccd_count': 0, 'clocking': False, 'vid_board': False,
                                 'sim_z': -99616, 'dh_heater': False}

        sc_const = {'roll': roll}

        if chips > 0:
            self.dpa = Balance1DPAMZT(self.date, self.model_specs['1dpamzt'], self.limits['1dpamzt'], acis_on_const)
            self.dea = Balance1DEAMZT(self.date, self.model_specs['1deamzt'], self.limits['1deamzt'], acis_on_const)
            self.acisfp = BalanceFPTEMP_11(self.date, self.model_specs['fptemp_11'], self.limits['fptemp_11'],
                                           acis_on_const)
            self.psmc = Balance1PDEAAT(self.date, self.model_specs['1pdeaat'], self.limits['1pdeaat'], psmc_on_const)

        else:
            self.dpa = Balance1DPAMZT(self.date, self.model_specs['1dpamzt'], self.limits['1dpamzt'],
                                      acis_off_stowed_const)
            self.dea = Balance1DEAMZT(self.date, self.model_specs['1deamzt'], self.limits['1deamzt'],
                                      acis_off_stowed_const)
            self.acisfp = BalanceFPTEMP_11(self.date, self.model_specs['fptemp_11'], self.limits['fptemp_11'],
                                           acis_off_stowed_const)
            self.psmc = Balance1PDEAAT(self.date, self.model_specs['1pdeaat'], self.limits['1pdeaat'],
                                       psmc_off_stowed_const)

        self.aca = BalanceAACCCDPT(self.date, self.model_specs['aacccdpt'], self.limits['aacccdpt'], {})
        self.oba = Balance4RT700T(self.date, self.model_specs['4rt700t'], self.limits['4rt700t'], {})
        self.mups1b = BalancePM1THV2T(self.date, self.model_specs['pm1thv2t'], self.limits['pm1thv2t'], sc_const)
        self.mups2a = BalancePM2THV1T(self.date, self.model_specs['pm2thv1t'], self.limits['pm2thv1t'], sc_const)
        self.tank = BalancePFTANK2T(self.date, self.model_specs['pftank2t'], self.limits['pftank2t'], sc_const)
        self.pline03t = BalancePLINE03T(self.date, self.model_specs['pline03t'], self.limits['pline03t'], sc_const)
        self.pline04t = BalancePLINE04T(self.date, self.model_specs['pline04t'], self.limits['pline04t'], sc_const)

        dashes = ''.join(["-", ] * 120)
        print(f'{dashes}\nMap Dwell Capability\n{dashes}')
        self.composite = self.map_composite()

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

        s = ''.join([f'    {p:>3}    {d:6.2f}\n' for p, d in self.dwell_limits.loc[self.pitch_range].iteritems()])
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
        s = ''.join([f'    {p:>3}    {d:6.2f}\n' for p, d in self.dwell_limits.loc[self.pitch_range].iteritems()])
        print(f'{dashes}\nFinal Dwell Limits: \n  Pitch    Duration\n{s}\n')
