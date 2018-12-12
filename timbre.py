
import numpy as np
import urllib
import json
from scipy.optimize import fminbound

from os.path import expanduser

from Chandra.Time import DateTime
import xija

home = expanduser("~")


pseudo_names = dict(
    zip(['aacccdpt', 'pftank2t', '1dpamzt', '4rt700t', '1deamzt'], ['aca0', 'pf0tank2t', 'dpa0', 'oba0', None]))


def load_model_specs():

    branch = 'https://raw.githubusercontent.com/sot/chandra_models/master/'

    model_specs = {}

    try:
        model_spec_url = branch + 'chandra_models/xija/dea/dea_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
        f = response.decode('utf-8')
    except:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/dea/dea_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['1deamzt'] = json.loads(f)

    try:
        model_spec_url = branch + 'chandra_models/xija/dpa/dpa_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    except:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/dpa/dpa_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['1dpamzt'] = json.loads(f)

    try:
        model_spec_url = branch + 'chandra_models/xija/acisfp/acisfp_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    except:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/acisfp/acisfp_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['fptemp'] = json.loads(f)

    try:
        model_spec_url = branch + 'chandra_models/xija/psmc/psmc_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    except:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/psmc/psmc_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['1pdeaat'] = json.loads(f)

    try:
        model_spec_url = branch + 'chandra_models/xija/pftank2t/pftank2t_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    except:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/pftank2t/pftank2t_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['pftank2t'] = json.loads(f)

    try:
        model_spec_url = branch + 'chandra_models/xija/tcylaft6/tcylaft6_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    except:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/tcylaft6/tcylaft6_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['tcylaft6'] = json.loads(f)

    try:
        model_spec_url = branch + 'chandra_models/xija/fwdblkhd/4rt700t_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    except:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/fwdblkhd/4rt700t_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['4rt700t'] = json.loads(f)

    try:
        model_spec_url = branch + 'chandra_models/xija/aca/aca_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    except:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/aca/aca_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['aacccdpt'] = json.loads(f)

    return model_specs


def CtoF(cs):
    try:
        return [c * 1.8 + 32 for c in cs]
    except TypeError:
        return cs * 1.8 + 32


def FtoC(cs):
    try:
        return [(c - 32) / 1.8 for c in cs]
    except TypeError:
        return (cs - 32) / 1.8


def setup_model(msid, t0, t1, model_spec, init):

    model = xija.ThermalModel(msid, start=t0, stop=t1, model_spec=model_spec)
    for key, value in init.items():
        if type(value) == type({}):
            model.comp[key].set_data(value['data'], value['times'])
        else:
            model.comp[key].set_data(value)

    return model


def run_profile(times, pitch, msid, model_spec, init):
    """ Assign initial values and run model.

    """

    model = setup_model(msid, times[0], times[-1], model_spec, init)
    model.comp['pitch'].set_data(pitch, times=times)
    model.make()
    model.calc()
    tmsid = model.get_comp(msid)

    # if pseudo_names[msid] is not None:
    #     tpseudo = model.get_comp(pseudo_names[msid])
    # else:
    #     tpseudo = None

    return {msid: tmsid}


def calc_binary_schedule(datesecs, p1, p2, t_dwell1, t_dwell2, msid, model_spec, init, duration=2592000,
                         t_backoff=864000):
    num = np.int(duration / (t_dwell1 + t_dwell2))
    times = np.cumsum([1, t_dwell1 - 1, 1, t_dwell2 - 1] * num)
    times = list(times)
    times.insert(0, 0)
    times = np.array(times) - times[0] + datesecs - t_backoff

    pitch = [p1, p1, p2, p2] * num
    pitch.insert(0, p1)
    pitch = np.array(pitch)

    model_results = run_profile(times, pitch, msid, model_spec, init)

    return model_results


def find_second_dwell(date, dwell1_pitch, dwell2_pitch, t_dwell1, dwell1_type, msid, limit, model_spec, init,
                      learning_rate=10000, max_iters=10000, max_dwell=1e6, precision=0.01, duration=2592000,
                      t_backoff=648000, debug=False):

    datesecs = DateTime(date).secs
    delta_temp = 1e6
    iters = 0
    current_t_dwell2 = t_dwell1
    results = {'converged': False, 'max_iteration': iters, 'iteration_limit': max_iters, 'unconverged_hot': False,
               'unconverged_cold': False, 'max_temp': 1e6, 'temperature_limit': limit, 'dwell_2_time': current_t_dwell2,
               'dwell_2_time_limit': max_dwell}

    # model = setup_model(msid, datesecs, datesecs + duration, model_spec, init)

    mult = 1.  # If 'hot' in dwell1_type
    if 'cool' in dwell1_type.lower():
        mult = -1.

    while delta_temp > precision and iters < max_iters and 0 < current_t_dwell2 < max_dwell:

        model_results = calc_binary_schedule(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, current_t_dwell2, msid,
                                             model_spec, init, t_backoff=t_backoff, duration=duration)
        model_temps = model_results[msid].mvals
        model_times = model_results[msid].times

        ind = model_times > (model_times[-1] - t_backoff * 2)
        max_temp = np.max(model_temps[ind])

        if debug:
            print('Current Dwell 2 Time: {}, Limit: {}, Max_Temp: {}, Delta Dwell 2 Time: {}'.format(current_t_dwell2, limit,
                                                                                       max_temp,
                                                                                       learning_rate *
                                                                                       (limit - max_temp)))

        current_t_dwell2 = current_t_dwell2 - mult * learning_rate * (limit - max_temp)

        delta_temp = np.abs(limit - max_temp)

        iters = iters + 1

    results['max_iteration'] = iters
    results['max_temp'] = max_temp
    results['dwell_2_time'] = current_t_dwell2

    if current_t_dwell2 >= max_dwell or current_t_dwell2 < 0:
        if max_temp >= limit:
            results['converged'] = False
            results['unconverged_hot'] = True

        elif max_temp < limit:
            results['converged'] = False
            results['unconverged_cold'] = True

    elif iters == max_iters:
        results['converged'] = False

    else:
        results['converged'] = True

    return results, model_times, model_temps


def create_opt_fun(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, msid, limit, model_spec, init, t_backoff, duration):
    def opt_binary_schedule(t):
        model_results = calc_binary_schedule(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, t, msid, model_spec,
                                             init, duration=duration, t_backoff=t_backoff)

        model_temps = model_results[msid].mvals
        model_times = model_results[msid].times
        ind = model_times > (model_times[-1] - t_backoff * 2)
        delta = np.abs(limit - np.max(model_temps[ind]))

        return delta

    return opt_binary_schedule


def find_second_dwell_sp(date, dwell1_pitch, dwell2_pitch, t_dwell1, msid, limit, model_spec, init,
                         max_iters=200, max_dwell=1e6, precision=0.01, duration=2592000, t_backoff=648000, debug=False):

    disp_num = 1
    if debug:
        disp_num = 3

    datesecs = DateTime(date).secs

    results = {'converged': False, 'max_iteration': 0, 'iteration_limit': max_iters, 'unconverged_hot': False,
               'unconverged_cold': False, 'max_temp': np.nan, 'temperature_limit': limit,
               'dwell_2_time': np.nan, 'dwell_2_time_limit': max_dwell}

    opt_fun = create_opt_fun(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, msid, limit, model_spec, init, t_backoff,
                             duration)

    res = fminbound(opt_fun, 0, 1e6, xtol=precision / 10., maxfun=max_iters, disp=disp_num, full_output=True)

    results['dwell_2_time'] = res[0]
    results['max_iteration'] = res[-1]

    model_results = calc_binary_schedule(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, results['dwell_2_time'], msid,
                                         model_spec, init, t_backoff=t_backoff, duration=duration)

    model_temps = model_results[msid].mvals
    model_times = model_results[msid].times

    ind = model_times > (model_times[-1] - t_backoff * 2)
    max_temp = np.max(model_temps[ind])

    results['max_temp'] = max_temp

    # I am using precision for two purposes, one is for convergence (above, one order of magnitude less), the other is
    # the minimum difference between the max temperature and the limit (here).
    if max_temp > (limit + precision):
        results['converged'] = False
        results['unconverged_hot'] = True

    elif max_temp < (limit - precision):
        results['converged'] = False
        results['unconverged_cold'] = True

    else:
        results['converged'] = True

    return results, model_times, model_temps


def char_model(msid, model_spec, init, limit, date, t_dwell1, pitch_step=1, dwell1_mask=None, dwell2_mask=None):
    pitch1 = np.arange(45, 171, pitch_step)
    pitch2 = np.arange(45, 171, pitch_step)

    if dwell1_mask is not None:
        for mask in dwell1_mask:
            ind = (pitch1 <= mask[0]) | (pitch1 >= mask[1])
            pitch1 = pitch1[ind]

    if dwell2_mask is not None:
        for mask in dwell2_mask:
            ind = (pitch2 <= mask[0]) | (pitch2 >= mask[1])
            pitch2 = pitch2[ind]

    converged = np.zeros((len(pitch1), len(pitch2))) == 1
    t_dwell2 = np.zeros((len(pitch1), len(pitch2)))

    for n, p1 in enumerate(pitch1):
        print("Running simulations for pitch: {}".format(p1))
        for m, p2 in enumerate(pitch2):
            results, _, _ = find_second_dwell_sp(date, p1, p2, t_dwell1, msid, limit, model_spec,
                                              init, debug=False)
            if results['converged']:
                converged[n, m] = True
                t_dwell2[n, m] = results['dwell_2_time']

    t_dwell2[~converged] = 0

    return t_dwell2, converged, pitch1, pitch2

if __name__ == '__main__':

    model_init = {'aacccdpt': {'aacccdpt': -9.5, 'aca0': -9.5, 'eclipse': False},
                  'pftank2t': {'pftank2t': FtoC(95), 'pf0tank2t': FtoC(95), 'eclipse': False, 'roll': 0},
                  '4rt700t': {'4rt700t': FtoC(95), 'oba0': FtoC(95), 'eclipse': False},
                  '1dpamzt': {'1dpamzt': 35, 'dpa0': 35, 'eclipse': False, 'roll': 0, 'vid_board': True,
                              'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000},
                  '1deamzt': {'1deamzt': 35, 'eclipse': False, 'roll': 0, 'vid_board': True, 'clocking': True,
                              'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}}

    model_specs = load_model_specs()

    date = '2019:001:00:00:00'
    t_dwell1 = 50000

    msid = 'aacccdpt'
    limit = -9.5
    hot_pitch = 135
    cool_pitch = 155
    dwell1_type = 'hot'

    # results, model_times, model_temps = find_second_dwell(date, hot_pitch, cool_pitch, t_dwell1, dwell1_type, msid,
    #                                                       limit, model_specs[msid], model_init[msid], max_iters=1000,
    #                                                       learning_rate=1000, debug=False)

    results, model_times, model_temps = find_second_dwell_sp(date, hot_pitch, cool_pitch, t_dwell1, msid,
                                                          limit, model_specs[msid], model_init[msid], debug=True)

    print(results)

