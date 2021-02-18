# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pandas as pd
from cxotime import CxoTime
from .timbre import run_state_pairs, find_second_dwell, load_model_specs


class Balance(object):
    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                 anchor_limited_pitch, anchor_offset_time, margin_factor):
        self.date = CxoTime(date).date
        self.datesecs = CxoTime(date).secs
        self.model_spec = model_spec
        self.limit = limit
        self.pitch_step = pitch_step
        self.pitch_range = np.arange(45, 181, pitch_step).astype(np.int)
        self.constant_conditions = constant_conditions
        self.anchor_offset_pitch = anchor_offset_pitch
        self.anchor_offset_time = anchor_offset_time
        self.anchor_limited_pitch = anchor_limited_pitch
        self.margin_factor = margin_factor
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
                                          limit, self.model_spec, self.model_init, limit_type=self.limit_type)
        return dwell_results['dwell_2_time'] * self.margin_factor

    def generate_balanced_pitch_dwells(self, datesecs, pitch_1, t_1, pitch_2, limit):

        # Expand dwell capability curve yielded by time #1 at pitch #1 (e.g. anchor hot time)
        state_pairs = list(({**{'pitch': pitch_1}, **self.constant_conditions},
                            {**{'pitch': p2}, **self.constant_conditions})
                           for p2 in self.pitch_range)

        if np.isnan(t_1):
            msg1 = f'Either {self.msid} is not limited at a pitch of {pitch_1} degrees, '
            msg2 = 'or there was an error passing the associated dwell duration.'
            print(msg1 + msg2)
            return None

        args = (self.msid, self.model_spec, self.model_init, limit, datesecs, t_1, state_pairs, self.state_pair_dtype)
        kwargs = {'limit_type': self.limit_type, 'print_progress': False}
        results1 = run_state_pairs(*args, **kwargs)

        # Expand dwell capability curve yielded by pitch #2 at the associated time calculated above.
        # This will include but expand upon the original time #1 at pitch #1 passed to this method.
        t_2_ind = results1['pitch2'] == pitch_2
        t_2 = results1['t_dwell2'][t_2_ind].item()
        state_pairs = list(({**{'pitch': pitch_2}, **self.constant_conditions},
                            {**{'pitch': p2}, **self.constant_conditions})
                           for p2 in self.pitch_range)

        args = (self.msid, self.model_spec, self.model_init, limit, datesecs, t_2, state_pairs, self.state_pair_dtype)
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

    def get_limited_results(self, results):
        if results is not None:
            ind = results['converged'] & (results['pitch1'] == self.anchor_offset_pitch) & \
                  ~np.isnan(results['t_dwell2'])
            return results[ind]
        else:
            return None

    def get_offset_results(self, results):
        if results is not None:
            ind = results['converged'] & (results['pitch1'] == self.anchor_limited_pitch) & ~np.isnan(results['t_dwell2'])
            return results[ind]
        else:
            return None


class Balance1DPAMZT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=60,
                 anchor_limited_pitch=170, anchor_offset_time=20000, margin_factor=0.95):
        self.msid = '1dpamzt'
        self.model_init = {'1dpamzt': limit, 'dpa0': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'fep_count': np.int8,
                                 'ccd_count': np.int8, 'clocking': np.bool, 'vid_board': np.bool, 'sim_z': np.int32}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class Balance1DEAMZT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=60,
                 anchor_limited_pitch=170, anchor_offset_time=20000, margin_factor=0.95):
        self.msid = '1deamzt'
        self.model_init = {'1deamzt': limit, 'dea0': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'fep_count': np.int8,
                                 'ccd_count': np.int8, 'clocking': np.bool, 'vid_board': np.bool, 'sim_z': np.int32}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class Balance1PDEAAT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=45, anchor_offset_time=20000, margin_factor=1.0):
        self.msid = '1pdeaat'
        self.model_init = {'1pdeaat': limit, 'pin1at': limit, 'eclipse': False, 'dpa_power': 0.0}
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'fep_count': np.int8, 'ccd_count': np.int8,
                                 'clocking': np.bool, 'vid_board': np.bool, 'sim_z': np.int32, 'dh_heater': np.bool}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class BalanceFPTEMP_11(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=60,
                 anchor_limited_pitch=170, anchor_offset_time=20000, margin_factor=0.95):
        self.msid = 'fptemp'
        self.model_init = {'fptemp': limit, '1cbat': -55.0, 'sim_px': 110.0, 'eclipse': False, 'dpa_power': 0.0,
                           'orbitephem0_x': 25000e3, 'orbitephem0_y': 25000e3, 'orbitephem0_z': 25000e3,
                           'aoattqt1': 0.0, 'aoattqt2': 0.0, 'aoattqt3': 0.0, 'aoattqt4': 1.0, 'dh_heater': False}
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'fep_count': np.int8,
                                 'ccd_count': np.int8, 'clocking': np.bool, 'vid_board': np.bool, 'sim_z': np.int32}
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class BalanceAACCCDPT(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=90, anchor_offset_time=20000, margin_factor=1.0):
        self.msid = 'aacccdpt'
        self.model_init = {'aacccdpt': limit, 'aca0': limit, 'eclipse': False}
        self.state_pair_dtype = {'pitch': np.float64, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class Balance4RT700T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=90, anchor_offset_time=20000, margin_factor=1.0):
        self.msid = '4rt700t'
        self.model_init = {'4rt700t': limit, 'oba0': limit, 'eclipse': False}
        self.state_pair_dtype = {'pitch': np.float64, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class BalancePFTANK2T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=50, anchor_offset_time=20000, margin_factor=1.0):
        self.msid = 'pftank2t'
        self.model_init = {'pftank2t': limit, 'pf0tank2t': limit, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class BalancePM1THV2T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=60, anchor_offset_time=20000, margin_factor=1.0):
        self.msid = 'pm1thv2t'
        self.model_init = {'pm1thv2t': limit, 'mups0': limit, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class BalancePM2THV1T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=170,
                 anchor_limited_pitch=60, anchor_offset_time=20000, margin_factor=1.0):
        self.msid = 'pm2thv1t'
        self.model_init = {'pm2thv1t': limit, 'mups0': limit * 10, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }
        self.limit_type = 'max'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class BalancePLINE04T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=60,
                 anchor_limited_pitch=170, anchor_offset_time=20000, margin_factor=1.0):
        self.msid = 'pline04t'
        self.model_init = {'pline04t': limit, 'pline04t0': limit, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }
        self.limit_type = 'min'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


class BalancePLINE03T(Balance):

    def __init__(self, date, model_spec, limit, constant_conditions, pitch_step=1, anchor_offset_pitch=60,
                 anchor_limited_pitch=175, anchor_offset_time=20000, margin_factor=1.0):
        self.msid = 'pline03t'
        self.model_init = {'pline03t': limit, 'pline03t0': limit, 'eclipse': False, }
        self.state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, }
        self.limit_type = 'min'

        super().__init__(date, model_spec, limit, constant_conditions, pitch_step, anchor_offset_pitch,
                         anchor_limited_pitch, anchor_offset_time, margin_factor)


def generate_balanced_dwells(date, chips, roll, limits, model_specs=None):

    # Calculate limiting acis factor for non acis models, acisfp may not always be the best to use.


    if model_specs == None:
        model_specs = load_model_specs()

    msid = '1dpamzt'
    constant_conditions = {'roll': roll, 'fep_count': chips, 'ccd_count': chips, 'clocking': True, 'vid_board': True,
                           'sim_z': 100000}
    dpa = Balance1DPAMZT(date, model_specs[msid], limits[msid], constant_conditions)
    dpa.results = dpa.generate_balanced_pitch_dwells(dpa.datesecs, dpa.anchor_limited_pitch, dpa.anchor_limited_time,
                                                     dpa.anchor_offset_pitch, dpa.limit)

    msid = '1deamzt'
    constant_conditions = {'roll': roll, 'fep_count': chips, 'ccd_count': chips, 'clocking': True, 'vid_board': True,
                           'sim_z': 100000}
    dea = Balance1DEAMZT(date, model_specs[msid], limits[msid], constant_conditions)
    dea.results = dea.generate_balanced_pitch_dwells(dea.datesecs, dea.anchor_limited_pitch, dea.anchor_limited_time,
                                                     dea.anchor_offset_pitch, dea.limit)

    msid = 'fptemp_11'
    acisfp_constant_conditions = {'roll': roll, 'fep_count': chips, 'ccd_count': chips, 'clocking': True,
                                  'vid_board': True,
                                  'sim_z': 100000}
    acisfp = BalanceFPTEMP_11(date, model_specs[msid], limits[msid], acisfp_constant_conditions)
    acisfp.results = acisfp.generate_balanced_pitch_dwells(acisfp.datesecs, acisfp.anchor_limited_pitch,
                                                           acisfp.anchor_limited_time, acisfp.anchor_offset_pitch,
                                                           acisfp.limit)

    msid = '1pdeaat'
    constant_conditions = {'roll': roll, 'fep_count': chips, 'ccd_count': chips, 'clocking': True, 'vid_board': True,
                           'sim_z': 100000, 'dh_heater': False}
    psmc = Balance1PDEAAT(date, model_specs[msid], limits[msid], constant_conditions,
                          anchor_offset_time=acisfp.anchor_limited_time)
    psmc.results = psmc.generate_balanced_pitch_dwells(psmc.datesecs, psmc.anchor_limited_pitch,
                                                       psmc.anchor_limited_time,
                                                       psmc.anchor_offset_pitch, psmc.limit)

    msid = 'aacccdpt'
    aca_constant_conditions = {}
    aca = BalanceAACCCDPT(date, model_specs[msid], limits[msid], aca_constant_conditions,
                          anchor_offset_time=acisfp.anchor_limited_time)
    aca.results = aca.generate_balanced_pitch_dwells(aca.datesecs, aca.anchor_limited_pitch, aca.anchor_limited_time,
                                                     aca.anchor_offset_pitch, aca.limit)

    msid = '4rt700t'
    oba_constant_conditions = {}
    oba = Balance4RT700T(date, model_specs[msid], limits[msid], oba_constant_conditions,
                         anchor_offset_time=acisfp.anchor_limited_time)
    oba.results = oba.generate_balanced_pitch_dwells(oba.datesecs, oba.anchor_limited_pitch, oba.anchor_limited_time,
                                                     oba.anchor_offset_pitch, oba.limit)

    msid = 'pm1thv2t'
    mups1b_constant_conditions = {'roll': roll}
    mups1b = BalancePM1THV2T(date, model_specs[msid], limits[msid], mups1b_constant_conditions,
                             anchor_offset_time=acisfp.anchor_limited_time)
    mups1b.results = mups1b.generate_balanced_pitch_dwells(mups1b.datesecs, mups1b.anchor_limited_pitch,
                                                           mups1b.anchor_limited_time, mups1b.anchor_offset_pitch,
                                                           mups1b.limit)

    msid = 'pm2thv1t'
    mups2a_constant_conditions = {'roll': roll}
    mups2a = BalancePM2THV1T(date, model_specs[msid], limits[msid], mups2a_constant_conditions,
                             anchor_offset_time=acisfp.anchor_limited_time)
    mups2a.results = mups2a.generate_balanced_pitch_dwells(mups2a.datesecs, mups2a.anchor_limited_pitch,
                                                           mups2a.anchor_limited_time, mups2a.anchor_offset_pitch,
                                                           mups2a.limit)

    msid = 'pftank2t'
    tank_constant_conditions = {'roll': roll}
    tank = BalancePFTANK2T(date, model_specs[msid], limits[msid], tank_constant_conditions,
                           anchor_offset_time=acisfp.anchor_limited_time)
    tank.results = tank.generate_balanced_pitch_dwells(tank.datesecs, tank.anchor_limited_pitch,
                                                       tank.anchor_limited_time,
                                                       tank.anchor_offset_pitch, tank.limit)

    msid = 'pline03t'
    pline03t_constant_conditions = {'roll': roll}
    pline03t = BalancePLINE03T(date, model_specs[msid], limits[msid], pline03t_constant_conditions,
                               anchor_offset_time=mups2a.anchor_limited_time)
    pline03t.results = pline03t.generate_balanced_pitch_dwells(pline03t.datesecs, pline03t.anchor_limited_pitch,
                                                               pline03t.anchor_limited_time,
                                                               pline03t.anchor_offset_pitch,
                                                               pline03t.limit)

    msid = 'pline04t'
    pline04t_constant_conditions = {'roll': roll}
    pline04t = BalancePLINE04T(date, model_specs[msid], limits[msid], pline04t_constant_conditions,
                               anchor_offset_time=mups2a.anchor_limited_time)
    pline04t.results = pline04t.generate_balanced_pitch_dwells(pline04t.datesecs, pline04t.anchor_limited_pitch,
                                                               pline04t.anchor_limited_time,
                                                               pline04t.anchor_offset_pitch,
                                                               pline04t.limit)

    limited_results_dict = {
        'fptemp_11': acisfp.get_limited_results(acisfp.results),
        '1dpamzt': dpa.get_limited_results(dpa.results),
        '1deamzt': dea.get_limited_results(dea.results),
        '1pdeaat': psmc.get_limited_results(psmc.results),
        'aacccdpt': aca.get_limited_results(aca.results),
        '4rt700t': oba.get_limited_results(oba.results),
        'pftank2t': tank.get_limited_results(tank.results),
        'pm1thv2t': mups1b.get_limited_results(mups1b.results),
        'pm2thv1t': mups2a.get_limited_results(mups2a.results),
        'pline03t': pline03t.get_limited_results(pline03t.results),
        'pline04t': pline04t.get_limited_results(pline04t.results)
    }

    offset_results_dict = {
        'fptemp_11': acisfp.get_offset_results(acisfp.results),
        '1dpamzt': dpa.get_offset_results(dpa.results),
        '1deamzt': dea.get_offset_results(dea.results),
        '1pdeaat': psmc.get_offset_results(psmc.results),
        'aacccdpt': aca.get_offset_results(aca.results),
        '4rt700t': oba.get_offset_results(oba.results),
        'pftank2t': tank.get_offset_results(tank.results),
        'pm1thv2t': mups1b.get_offset_results(mups1b.results),
        'pm2thv1t': mups2a.get_offset_results(mups2a.results),
        'pline03t': pline03t.get_offset_results(pline03t.results),
        'pline04t': pline04t.get_offset_results(pline04t.results)
    }

    msids = list(limited_results_dict.keys())
    pitch_range = np.arange(45, 181, 1)

    limited_results = pd.DataFrame(index=pitch_range, columns=msids, dtype=np.float64)
    for msid, res in limited_results_dict.items():
        if not isinstance(res, type(None)):
            limited_results[msid][res['pitch2']] = res['t_dwell2']

    offset_results = pd.DataFrame(index=pitch_range, columns=msids, dtype=np.float64)
    for msid, res in offset_results_dict.items():
        if not isinstance(res, type(None)):
            offset_results[msid][res['pitch2']] = res['t_dwell2']

    return limited_results, offset_results
