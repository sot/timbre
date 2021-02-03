# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from cxotime import CxoTime
from .timbre import run_state_pairs, find_second_dwell


class Balance(object):
    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                 anchor_limited_pitch, anchor_offset_time):
        self.date = CxoTime(date).date
        self.datesecs = CxoTime(date).secs
        self.model_spec = model_spec
        self.limit = limit
        self.pitch_step = pitch_step
        self.constant_conditions = constant_conditions
        self.anchor_offset_pitch = anchor_offset_pitch
        self.anchor_offset_time = anchor_offset_time
        self.anchor_limited_pitch = anchor_limited_pitch
        self.margin_factor = 1.0
        self.limit_type = 'max'
        self.anchor_limited_time = self.find_anchor_condition(self.anchor_offset_pitch,
                                                              self.anchor_limited_pitch,
                                                              self.anchor_offset_time,
                                                              self.limit)

    def find_anchor_condition(self, p_offset, p_limited, time_offset_steady_state, limit):

        # Define dwell states that are used to determine max cooling dwell time
        dwell1_state = {**{'pitch': p_offset}, **self.constant_conditions}
        dwell2_state = {**{'pitch': p_limited}, **self.constant_conditions}

        # Find maximum hot time (including margin safety factor)
        dwell_results = find_second_dwell(self.date, dwell1_state, dwell2_state, time_offset_steady_state, self.msid,
                                          limit,
                                          self.model_spec, self.model_init, limit_type=self.limit_type)
        return dwell_results['dwell_2_time'] * self.margin_factor

    def generate_balanced_pitch_dwells(self, datesecs, pitch_1, t_1, pitch_2, limit):

        # Expand dwell capability curve yielded by time #1 at pitch #1 (e.g. anchor hot time)
        state_pairs = list(({**{'pitch': pitch_1}, **self.constant_conditions},
                            {**{'pitch': p2}, **self.constant_conditions})
                           for p2 in range(45, 181, self.pitch_step))

        args = (self.msid, self.model_spec, self.model_init, limit, datesecs, t_1, state_pairs, self.state_pair_dtype)
        kwargs = {'limit_type': self.limit_type}
        results1 = run_state_pairs(*args, **kwargs)

        # Expand dwell capability curve yielded by pitch #2 at the associated time calculated above.
        # This will include but expand upon the original time #1 at pitch #1 passed to this method.
        t_2_ind = results1['pitch2'] == pitch_2
        t_2 = results1['t_dwell2'][t_2_ind].item()
        state_pairs = list(({**{'pitch': pitch_2}, **self.constant_conditions},
                            {**{'pitch': p2}, **self.constant_conditions})
                           for p2 in range(45, 181, self.pitch_step))

        args = (self.msid, self.model_spec, self.model_init, limit, datesecs, t_2, state_pairs, self.state_pair_dtype)
        results2 = run_state_pairs(*args, **kwargs)

        results = np.hstack((results1, results2))

        not_nan = np.where(~np.isnan(results['t_dwell2']))[0]
        if any(not_nan):
            ind = np.where(results[not_nan]['t_dwell2'] < 0)[0]
            if any(ind):
                inds = not_nan[ind].item()
                for ind in inds:
                    results[ind]['t_dwell2'] = np.nan

        not_nan = np.where(~np.isnan(results['t_dwell2']))[0]
        if any(not_nan):
            inds = np.where(results[not_nan]['t_dwell2'] > 300000)[0]
            if any(ind):
                for ind in inds:
                    ind = not_nan[ind].item()
                    results[ind]['t_dwell2'] = np.nan

        return results


class Balance1DPAMZT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=60,
                 anchor_limited_pitch=170, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = '1dpamzt'
        self.model_init = {'1dpamzt': limit, 'dpa0': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'fep_count': np.int8,
                                 'ccd_count': np.int8, 'clocking': np.bool, 'vid_board': np.bool, 'sim_z': np.int32}
        self.margin_factor = 0.95  # margin on maximum limited time at anchor pitch


class Balance1DEAMZT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=60,
                 anchor_limited_pitch=170, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = '1deamzt'
        self.model_init = {'1deamzt': limit, 'dea0': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'fep_count': np.int8,
                                 'ccd_count': np.int8, 'clocking': np.bool, 'vid_board': np.bool, 'sim_z': np.int32}
        self.margin_factor = 0.95  # margin on maximum limited time at anchor pitch


class Balance1PDEAAT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=45, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = '1pdeaat'
        self.model_init = {'1pdeaat': limit, 'pin1at': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'fep_count': np.int8,
                                 'ccd_count': np.int8, 'clocking': np.bool, 'vid_board': np.bool, 'sim_z': np.int32}


class BalanceFPTEMP_11(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=60,
                 anchor_limited_pitch=170, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = 'fptemp'
        self.model_init = {'fptemp': limit, '1cbat': -55.0, 'sim_px': 110.0, 'eclipse': False, 'dpa_power': 0.0,
                           'orbitephem0_x': 25000e3, 'orbitephem0_y': 25000e3, 'orbitephem0_z': 25000e3,
                           'aoattqt1': 0.0, 'aoattqt2': 0.0, 'aoattqt3': 0.0, 'aoattqt4': 1.0, 'dh_heater': False}
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'fep_count': np.int8,
                                 'ccd_count': np.int8, 'clocking': np.bool, 'vid_board': np.bool, 'sim_z': np.int32}
        self.margin_factor = 0.95  # margin on maximum limited time at anchor pitch


class BalanceAACCCDPT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=90, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = 'aacccdpt'
        self.model_init = {'aacccdpt': limit, 'aca0': limit, 'eclipse': False}
        self.state_pair_dtype = {'pitch': np.float64, }


class Balance4RT700T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=90, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = '4rt700t'
        self.model_init = {'4rt700t': limit, 'oba0': limit, 'eclipse': False}
        self.state_pair_dtype = {'pitch': np.float64, }


class BalancePFTANK2T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=50, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = 'pftank2t'
        self.model_init = {'pftank2t': limit, 'pf0tank2t': limit, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }


class BalancePM1THV2T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=60, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = 'pm1thv2t'
        self.model_init = {'pm1thv2t': limit, 'mups0': limit, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }


class BalancePM2THV1T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=60, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = 'pm2thv1t'
        self.model_init = {'pm2thv1t': limit, 'mups0': limit * 10, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }


class BalancePLINE03T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=60, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = 'pline03t'
        self.model_init = {'pline03t': limit, 'pline03t0': limit, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }
        self.limit_type = 'min'


class BalancePLINE04T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=60, anchor_offset_time=20000):
        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time)

        self.msid = 'pline04t'
        self.model_init = {'pline4t': limit, 'pline04t0': limit, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }
        self.limit_type = 'min'

