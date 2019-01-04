
import numpy as np
import urllib
import json
import scipy.optimize as opt
from pandas import DataFrame
from multiprocessing import Process, Manager

from os.path import expanduser

from Chandra.Time import DateTime
import xija

home = expanduser("~")


def load_model_specs():

    branch = 'https://raw.githubusercontent.com/sot/chandra_models/master/'

    model_specs = {}

    internet = True
    try:
        _ = urllib.request.urlopen('https://example.com/index.html')
    except urllib.error.URLError:
        internet = False

    if internet:
        model_spec_url = branch + 'chandra_models/xija/aca/aca_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    else:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/aca/aca_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['aacccdpt'] = json.loads(f)

    if internet:
        model_spec_url = branch + 'chandra_models/xija/dea/dea_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
        f = response.decode('utf-8')
    else:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/dea/dea_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['1deamzt'] = json.loads(f)

    if internet:
        model_spec_url = branch + 'chandra_models/xija/dpa/dpa_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    else:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/dpa/dpa_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['1dpamzt'] = json.loads(f)

    if internet:
        model_spec_url = branch + 'chandra_models/xija/acisfp/acisfp_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    else:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/acisfp/acisfp_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['fptemp'] = json.loads(f)

    if internet:
        model_spec_url = branch + 'chandra_models/xija/psmc/psmc_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    else:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/psmc/psmc_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['1pdeaat'] = json.loads(f)

    if internet:
        model_spec_url = branch + 'chandra_models/xija/pftank2t/pftank2t_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    else:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/pftank2t/pftank2t_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['pftank2t'] = json.loads(f)

    if internet:
        model_spec_url = branch + 'chandra_models/xija/tcylaft6/tcylaft6_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    else:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/tcylaft6/tcylaft6_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['tcylaft6'] = json.loads(f)

    if internet:
        model_spec_url = branch + 'chandra_models/xija/fwdblkhd/4rt700t_spec.json'
        with urllib.request.urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')
    else:
        with open(home + '/AXAFLIB/chandra_models/chandra_models/xija/fwdblkhd/4rt700t_spec.json', 'rb') as fid:
            f = fid.read()
    model_specs['4rt700t'] = json.loads(f)

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


def run_profile(times, pitch, msid, model_spec, init, pseudo=None):
    """ Assign initial values and run model.

    """

    model = setup_model(msid, times[0], times[-1], model_spec, init)
    model.comp['pitch'].set_data(pitch, times=times)
    model.make()
    model.calc()
    tmsid = model.get_comp(msid)
    results = {msid: tmsid}

    if pseudo is not None:
        results[pseudo] = model.get_comp(pseudo_names[msid])

    return results


def calc_binary_schedule(datesecs, p1, p2, t_dwell1, t_dwell2, msid, model_spec, init, duration=2592000,
                         t_backoff=864000, scale_factor=500., pseudo=None):

    t_dwell2 = t_dwell2[0] * scale_factor
    t_dwell1 = t_dwell1 * scale_factor

    if t_dwell2 < 0:
        t_dwell2 = 0

    if t_dwell2 > 1e6:
        t_dwell2 = 1e6

    # print('{}: {}'.format(t_dwell1, t_dwell2))

    num = np.int(duration / (t_dwell1 + t_dwell2))
    times = np.cumsum([1, t_dwell1 - 1, 1, t_dwell2 - 1] * num)
    times = list(times)
    times.insert(0, 0)
    times = np.array(times) - times[0] + datesecs - t_backoff

    pitch = [p1, p1, p2, p2] * num
    pitch.insert(0, p1)
    pitch = np.array(pitch)

    model_results = run_profile(times, pitch, msid, model_spec, init, pseudo=pseudo)

    return model_results


def create_opt_fun(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, msid, limit, model_spec, init, t_backoff, duration,
                   scale_factor=1.):
    def opt_binary_schedule(t):
        model_results = calc_binary_schedule(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, t, msid, model_spec,
                                             init, duration=duration, t_backoff=t_backoff, scale_factor=scale_factor)

        model_temps = model_results[msid].mvals
        model_times = model_results[msid].times
        ind = model_times > (model_times[-1] - t_backoff * 2)
        delta = np.abs(limit - np.max(model_temps[ind]))

        return delta

    return opt_binary_schedule


def find_second_dwell_sp(date, dwell1_pitch, dwell2_pitch, t_dwell1, msid, limit, model_spec, init, scale_factor=500.,
                         max_iters=200, max_dwell=1e6, temperature_precision=0.01, duration=2592000, t_backoff=648000,
                         debug=False, pseudo=None):

    datesecs = DateTime(date).secs

    results = {'converged': False, 'max_iteration': 0, 'iteration_limit': max_iters, 'unconverged_hot': False,
               'unconverged_cold': False, 'max_temp': np.nan, 'temperature_limit': limit,
               'dwell_2_time': np.nan, 'dwell_2_time_limit': max_dwell, 'min_temp':np.nan, 'max_pseudo':np.nan,
               'min_pseudo':np.nan}

    opt_fun = create_opt_fun(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, msid, limit, model_spec, init, t_backoff,
                             duration, scale_factor=scale_factor)

    res = opt.minimize(opt_fun, np.array([10,]), method='powell', options={'disp': debug}, tol=0.00001)

    t_dwell2 = res.x * scale_factor

    # opt.minimize will return what it tried to use, not what I forced it to use. Unfortunately the bounded minimization
    # options did not provide adequate convergence performance, or required a jacobian.
    if t_dwell2 > max_dwell:
        t_dwell2 = max_dwell

    results['dwell_2_time'] = t_dwell2
    results['max_iteration'] = res.nit

    model_results = calc_binary_schedule(datesecs, dwell1_pitch, dwell2_pitch, t_dwell1, [res.x,], msid,
                                         model_spec, init, t_backoff=t_backoff, duration=duration,
                                         scale_factor=scale_factor, pseudo=pseudo)

    model_temps = model_results[msid].mvals
    model_times = model_results[msid].times

    ind = model_times > (model_times[-1] - t_backoff * 2)
    max_temp = np.max(model_temps[ind])
    min_temp = np.min(model_temps[ind])
    results['max_temp'] = max_temp
    results['min_temp'] = min_temp

    if pseudo:
        pseudo_temps = model_results[pseudo].mvals
        max_pseudo = np.max(pseudo_temps[ind])
        min_pseudo = np.min(pseudo_temps[ind])
        results['max_pseudo'] = max_pseudo
        results['min_pseudo'] = min_pseudo

    if max_temp > (limit + temperature_precision):
        results['converged'] = False
        results['unconverged_hot'] = True

    elif max_temp < (limit - temperature_precision):
        results['converged'] = False
        results['unconverged_cold'] = True

    else:
        results['converged'] = True

    return results, model_results


def char_model(msid, model_spec, init, limit, date, t_dwell1, pitch_step=1, scale_factor=500., pseudo='aca0'):
    pitch1 = np.arange(45, 171, pitch_step)
    pitch2 = np.arange(45, 171, pitch_step)

    converged = np.zeros((len(pitch1), len(pitch2))) == 1
    t_dwell2 = np.zeros((len(pitch1), len(pitch2)))

    temperature_range = np.zeros((len(pitch1), len(pitch2), 2))
    if pseudo:
        pseudo_range = np.zeros((len(pitch1), len(pitch2), 2))
    else:
        pseudo_range = None

    for n, p1 in enumerate(pitch1):
        print("Running simulations for pitch: {}".format(p1))
        for m, p2 in enumerate(pitch2):
            results, model_results = find_second_dwell_sp(date, p1, p2, t_dwell1, msid, limit, model_spec,
                                              init, debug=False, scale_factor=scale_factor, pseudo='aca0')
            if results['converged']:
                converged[n, m] = True
                t_dwell2[n, m] = results['dwell_2_time']
                temperature_range[n, m, 0] = results['max_temp']
                temperature_range[n, m, 1] = results['min_temp']

                if pseudo:
                    pseudo_range[n, m, 0] = results['max_pseudo']
                    pseudo_range[n, m, 1] = results['min_pseudo']

    return t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range


def extract_data_from_file(filename):
    d = np.genfromtxt(filename, delimiter=',')
    pitch1 = d[1:, 0]
    pitch2 = d[0, 1:]
    t_dwell2 = d[1:, 1:]
    return t_dwell2, pitch1, pitch2


def save_characterization(pitch1, pitch2, t_dwell2, msid, t_dwell1, datestr, step, pad_text=''):

    tc = DataFrame(t_dwell2)
    r = dict(zip(np.arange(0, len(pitch1)), pitch1))
    c = dict(zip(np.arange(0, len(pitch2)), pitch2))
    tc = tc.rename(index=r, columns=c)
    filename = '{}_fixed_initial_dwell_{}ks_{}_{}deg_step{}.csv'.format(msid, int(t_dwell1), datestr, step, pad_text)
    tc.to_csv(filename)
    print('Saved to: {}'.format(filename))


def combine_characterizations(cases, len1, len2):
    t2 = np.reshape(np.vstack(cases), (len(cases), len1, len2))
    t2[t2 == 0] = np.nan
    t2 = np.nanmedian(t2, axis=0)
    t2[np.isnan(t2)] = 0
    return t2


def characterization(msid, limit, date, t_dwell1, step, init, model_spec, scale_factor, shared_data):

    date = DateTime(date).date
    datestr = date[:4] + date[5:8]

    msg = 'Running: {} with a fixed initial dwell of {}ks, for {}, with a {} degree pitch step and {} scale factor,' + \
          ' Notes: {}'
    print(msg.format(msid, int(t_dwell1), datestr, step, scale_factor, 'None'))

    t_dwell1_scaled = t_dwell1 / scale_factor
    t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range = char_model(msid, model_spec, init, limit,
                                                                                      date, t_dwell1_scaled,
                                                                                      pitch_step=step,
                                                                                      scale_factor=scale_factor)
    save_characterization(pitch1, pitch2, t_dwell2, msid, t_dwell1, datestr, step,
                          pad_text='_{}_scale_factor'.format(int(scale_factor)))
    save_characterization(pitch1, pitch2, temperature_range[:, :, 0], msid, t_dwell1, datestr, step,
                          pad_text='_{}_scale_factor_max_temp'.format(int(scale_factor)))
    save_characterization(pitch1, pitch2, temperature_range[:, :, 1], msid, t_dwell1, datestr, step,
                          pad_text='_{}_scale_factor_min_temp'.format(int(scale_factor)))
    save_characterization(pitch1, pitch2, pseudo_range[:, :, 0], msid, t_dwell1, datestr, step,
                          pad_text='_{}_scale_factor_max_pseudo'.format(int(scale_factor)))
    save_characterization(pitch1, pitch2, pseudo_range[:, :, 1], msid, t_dwell1, datestr, step,
                          pad_text='_{}_scale_factor_min_pseudo'.format(int(scale_factor)))

    if shared_data is not None:
        shared_data.append((scale_factor, t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range))
    else:
        return t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range


def run_characterization_all(msid, limit, date, t_dwell1, step, init, model_spec, scale_factors=(200, 500, 1000, 2000)):

    date = DateTime(date).date
    datestr = date[:4] + date[5:8]

    cases = []

    for scale_factor in scale_factors:
        t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range = \
            characterization(msid, limit, date, t_dwell1, step, init, model_spec, scale_factor, None)
        cases.append(t_dwell2)

    t_dwell2 = combine_characterizations(cases, len(pitch1), len(pitch2))
    save_characterization(pitch1, pitch2, t_dwell2, msid, t_dwell1, datestr, step, pad_text='_averaged')


def run_characterization_parallel(msid, limit, date, t_dwell1, step, init, model_spec, scale_factors=(200, 500, 1000,
                                                                                                      2000)):

    date = DateTime(date).date
    datestr = date[:4] + date[5:8]

    manager = Manager()
    cases = manager.list()
    jobs = []

    for scale_factor in scale_factors:
        p = Process(target=characterization, args=(msid, limit, date, t_dwell1, step, init, model_spec, scale_factor,
                                                   cases))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    dwell2_cases = [c[1] for c in cases]
    pitch1 = cases[0][3]
    pitch2 = cases[0][4]

    t_dwell2 = combine_characterizations(dwell2_cases, len(pitch1), len(pitch2))
    save_characterization(pitch1, pitch2, t_dwell2, msid, t_dwell1, datestr, step, pad_text='_averaged')


pseudo_names = dict(
    zip(['aacccdpt', 'pftank2t', '1dpamzt', '4rt700t', '1deamzt'], ['aca0', 'pf0tank2t', 'dpa0', 'oba0', None]))

model_init = {'aacccdpt': {'aacccdpt': -10., 'aca0': -10., 'eclipse': False},
              'pftank2t': {'pftank2t': FtoC(95.), 'pf0tank2t': FtoC(95.), 'eclipse': False, 'roll': 0},
              'tcylaft6': {'tcylaft6': FtoC(120.), 'cc0': FtoC(120.), 'eclipse': False, 'roll': 0},
              '4rt700t': {'4rt700t': FtoC(95.), 'oba0': FtoC(95.), 'eclipse': False},
              '1dpamzt': {'1dpamzt': 35., 'dpa0': 35., 'eclipse': False, 'roll': 0, 'vid_board': True,
                          'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000},
              '1deamzt': {'1deamzt': 35., 'eclipse': False, 'roll': 0, 'vid_board': True, 'clocking': True,
                          'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}}


if __name__ == '__main__':


    model_specs = load_model_specs()

    date = '2019:001:00:00:00'
    scale_factor = 500.
    t_dwell1 = 10000. / scale_factor

    msid = 'aacccdpt'
    limit = -9.5
    hot_pitch = 90
    cool_pitch = 160

    results, model_results = find_second_dwell_sp(date, hot_pitch, cool_pitch, t_dwell1, msid,
                                                             limit, model_specs[msid], model_init[msid], debug=True,
                                                             scale_factor=scale_factor, max_dwell=1e6/scale_factor)

    print(results)

