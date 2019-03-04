
import numpy as np
from urllib.request import urlopen
from urllib.error import URLError
import json
import scipy.optimize as opt
from pandas import DataFrame
from multiprocessing import Process, Manager

from os.path import expanduser

from Chandra.Time import DateTime
import xija

home = expanduser("~")

results_dtype = [('msid', '|U20'),
                 ('date', '|U8'),
                 ('obsid1', np.int32),
                 ('sequence1', np.int32),
                 ('duration1_fraction', np.float64),
                 ('obsid2', np.int32),
                 ('sequence2', np.int32),
                 ('limit', np.float64),
                 ('pitch1', np.float64),
                 ('pitch2', np.float64),
                 ('roll1', np.float64),
                 ('roll2', np.float64),
                 ('ccd_count1', np.int8),
                 ('ccd_count2', np.int8),
                 ('fep_count1', np.int8),
                 ('fep_count2', np.int8),
                 ('clocking1', np.bool),
                 ('clocking2', np.bool),
                 ('vid_board1', np.bool),
                 ('vid_board2', np.bool),
                 ('sim_z1', np.int32),
                 ('sim_z2', np.int32),
                 ('state_data1', np.object),
                 ('state_data2', np.object),
                 ('t_dwell1', np.float64),
                 ('t_dwell2', np.float64),
                 ('temp_min', np.float64),
                 ('temp_max', np.float64),
                 ('temp_mean', np.float64),
                 ('pseudo_min', np.float64),
                 ('pseudo_max', np.float64),
                 ('pseudo_mean', np.float64),
                 ('converged', np.bool),
                 ('unconverged_hot', np.bool),
                 ('unconverged_cold', np.bool),
                 ('hotter_state', np.int8),
                 ('colder_state', np.int8)]


def load_model_specs():
    """ Load Xija model parameters for all available models.

    Returns:
        dictionary: A dictionary containing the model specifications for all available Xija models

    Note:
        This will need to be updated as new models are approved or existing models are renamed.

    """

    def get_model(branch, internet):
        """ Load parameters for a single Xija model.

        Args:
            branch (str): Relative location of model file, starting from the chandra_models/chandra_models/xija/
                directory
            internet (bool): Availability of an internet connection, for accessing github.com

        Returns:
            dictionary: JSON file stored as a dictionary, containing Xija model parameters

        """

        url = 'https://raw.githubusercontent.com/sot/chandra_models/master/chandra_models/xija/'
        local_dir = '/AXAFLIB/chandra_models/chandra_models/xija/'

        if internet:
            model_spec_url = url + branch  # aca/aca_spec.json'
            with urlopen(model_spec_url) as url:
                response = url.read()
                f = response.decode('utf-8')
        else:
            with open(home + local_dir + branch) as fid:  # 'aca/aca_spec.json', 'rb') as fid:
                f = fid.read()
        return json.loads(f)


    model_specs = {}

    internet = True
    try:
        _ = urlopen('https://github.com')
    except URLError:
        internet = False

    model_specs['aacccdpt'] = get_model('aca/aca_spec.json', internet)
    model_specs['1deamzt'] = get_model('dea/dea_spec.json', internet)
    model_specs['1dpamzt'] = get_model('dpa/dpa_spec.json', internet)
    model_specs['fptemp'] = get_model('acisfp/acisfp_spec.json', internet)
    model_specs['1pdeaat'] = get_model('psmc/psmc_spec.json', internet)
    model_specs['pftank2t'] = get_model('pftank2t/pftank2t_spec.json', internet)
    model_specs['tcylaft6'] = get_model('tcylaft6/tcylaft6_spec.json', internet)
    model_specs['4rt700t'] = get_model('fwdblkhd/4rt700t_spec.json', internet)
    model_specs['pline03t'] = get_model('pline/pline03t_model_spec.json', internet)
    model_specs['pline04t'] = get_model('pline/pline04t_model_spec.json', internet)

    return model_specs


def c_to_f(temp):
    """ Convert Celsius to Fahrenheit

    Args:
        temp (int, float, numpy.ndarray, list, tuple): Temperature in Celsius

    Returns:
        (int, float, numpy.ndarray, list, tuple): Temperature in Fahrenheit

    """
    if type(temp) is list or type(temp) is tuple:
        return [c * 1.8 + 32 for c in temp]
    else:
        return temp * 1.8 + 32.0


def f_to_c(temp):
    """ Convert Fahrenheit to Celsius

    Args:
        temp (int, float, numpy.ndarray, list, tuple): Temperature in Fahrenheit

    Returns:
        (int, float, numpy.ndarray, list, tuple): Temperature in Celsius

    """
    if type(temp) is list or type(temp) is tuple:
        return [(c - 32) / 1.8 for c in temp]
    else:
        return (temp - 32.0) / 1.8


def setup_model(msid, t0, t1, model_spec, init):
    """ Create Xija model object

    This function creates a Xija model object with initial parameters, if any. This function is intended to create a
    streamlined method to creating Xija models that can take both single value data and time defined data
    (e.g. [pitch1, pitch2, pitch3], [time1, time2, time3]), defined in the `init` dictionary.

    Args:
        msid (str): Primary MSID for model; in this case it can be anything as it is only being used to name the model,
            however keeping the convention to name the model after the primary MSID being predicted reduces confusion
        t0 (str, float, int): Start time for model prediction; this can be any format that Chandra.Time.DateTime accepts
        t1 (str, float, int): End time for model prediction; this can be any format that Chandra.Time.DateTime accepts
        model_spec (dict, string): Dictionary of model parameters or file location where parameters can be imported
        init (dict): Dictionary of Xija model initialization parameters, can be empty

    Returns:
        (xija.model.XijaModel): Xija model object

    Example:

        model_specs = load_model_specs()

        init = {'1dpamzt': 35., 'dpa0': 35., 'eclipse': False, 'roll': 0, 'vid_board': True, 'pitch':155,
                          'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}

        model = setup_model('1dpamzt', '2019:001:00:00:00', '2019:010:00:00:00', model_specs['1dpamzt'], init)

    Notes:
        This does not run the model, only sets up the model to be run.

        Any parameters not specified in `init` will either need to be pulled from telemetry or explicitly defined
        outside of this function before running the model.

    """

    model = xija.ThermalModel(msid, start=t0, stop=t1, model_spec=model_spec)
    for key, value in init.items():
        if isinstance(value, dict):
            model.comp[key].set_data(value['data'], value['times'])
        else:
            model.comp[key].set_data(value)

    return model


def run_profile(times, schedule, msid, model_spec, init, pseudo=None):
    """ Run a Xija model for a given time and pitch profile.

    This was intended to be used to run a simple time/pitch profile, however further time changing input parameters can
    be defined in the `init` dictionary.

    Args:
        times (numpy.ndarray): Array of time values, in seconds from '1997:365:23:58:56.816' (Chandra.Time.DateTime
            epoch)
        schedule (dict): Dictionary of pitch, roll, etc. values that match the time values specified above in `times`
        msid (str): Primary MSID for model being run
        model_spec (dict, string): Dictionary of model parameters or file location where parameters can be imported
        init (dict): Dictionary of Xija model initialization parameters, can be empty but not recommended
        pseudo (:obj:`str`, optional): Name of one or more pseudo MSIDs used in the model, if any, only necessary if one
            wishes to retrieve model results for this pseudo node, if it exists

    Returns:
        dict: Dictionary of results, keys are node names (e.g. 'aacccdpt', 'aca0'), values are Xija model component
            objects

    Example:

        times = DateTime(['2019:001:00:00:00', '2019:001:12:00:00', '2019:002:00:00:00', '2019:003:00:00:00']).secs

        pitch = np.array([150, 90, 156, 156])

        model_specs = load_model_specs()

        init = {'1dpamzt': 20., 'dpa0': 20., 'eclipse': False, 'roll': 0, 'vid_board': True,
                          'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}

        results = run_profile(times, pitch, '1dpamzt', model_specs['1dpamzt'], init, pseudo='dpa0')

    Notes:
        Any parameters specified in `init` will be overwritten by those specified in the body of this function, if they
        happen to be defined in both places.

    """

    model = setup_model(msid, times[0], times[-1], model_spec, init)

    for key, value in schedule.items():
        model.comp[key].set_data(value, times=times)

    model.make()
    model.calc()
    tmsid = model.get_comp(msid)
    results = {msid: tmsid}

    if pseudo is not None:
        results[pseudo] = model.get_comp(pseudo_names[msid])

    return results


def calc_binary_schedule(datesecs, state1, state2, t_dwell1_scaled, t_dwell2, msid, model_spec, init, duration=2592000.,
                         t_backoff=1725000., scale_factor=1., max_dwell=1.e6, pseudo=None):
    """ Simulate a schedule that switches between two pitch values

    This runs the model over a "binary" schedule. This function is intended to be used to optimize the `t_dwell2`
    parameter so that the predicted temperature during the last `t_backoff` number of seconds peaks within a tolerance
    of a limit (limit used and specified in a different function).

    Args:
        datesecs (float, int): Date for start of simulation, in seconds from '1997:365:23:58:56.816'
            (Chandra.Time.DateTime epoch)
        state1 (dict): States for fixed dwell (pitch, roll, ccds, etc.)
        state2 (dict): States for variable dwell (pitch, roll, ccds, etc.)
        t_dwell1_scaled (float, int): Fixed dwell duration in seconds, this is in the SCALED format
        t_dwell2 (list, tuple): Variable dwell duration in seconds (this is the parameter that is optimized), the
            optimization routine returns a list, though in this case we are only interested in the first value (only
            value?), this is in the SCALED format
        msid (str): Primary MSID for model being run
        model_spec (dict, string): Dictionary of model parameters or file location where parameters can be imported
        init (dict): Dictionary of Xija model initialization parameters
        duration (:obj:`float`, optional): Duration for entire simulated schedule, defaults to 30 days (in seconds)
        t_backoff (:obj:`float`, optional): Duration for tail end of simulated schedule used to determine convergence,
            defaults to 10 days (in seconds)
        scale_factor (:obj:`float`, optional): scale factor for times, this is a way to scale the time values to help
            convergence, this does not affect the final `t_dwell2` value in any significant way (when it converges)
        max_dwell (:obj: `float`, optional): Maximum single dwell length
        pseudo (:obj:`str`, optional): Name of one or more pseudo MSIDs used in the model, if any, only necessary if one
            wishes to retrieve model results for this pseudo node, if it exists

    Returns:
        dict: Dictionary of results, keys are node names (e.g. 'aacccdpt', 'aca0'), values are Xija model component
            objects, this is the same object returned by `run_profile`

    Notes:
        Since the optimization routine used does not allow for bounded independent variables, a hard limit on `t_dwell2`
        needs to be forced. 0 seconds is used for the minimum value, and 1e6 seconds is used for the default maximum
        value. Failure to force these bounds results in convergence issues.

        Keys in state1 must match keys in state2.

        Keys in state1 must match Xija component names (e.g. 'pitch', 'ccd_count', 'sim_z')

    """

    t_dwell2 = t_dwell2[0] * scale_factor
    t_dwell1 = t_dwell1_scaled * scale_factor

    if t_dwell2 < 0:
        t_dwell2 = 0

    if t_dwell2 > max_dwell:
        t_dwell2 = max_dwell

    num = np.int(duration / (t_dwell1 + t_dwell2))
    reltimes = np.cumsum([1, t_dwell1 - 1, 1, t_dwell2 - 1] * num)
    # reltimes = list(reltimes)
    # reltimes.insert(0, 1)
    times = np.array(reltimes) - reltimes[0] + datesecs - t_backoff

    schedule = dict(zip(state1.keys(), []))
    for key, value in state1.items():
        layout = [state1[key], state1[key], state2[key], state2[key]] * num
        # layout.insert(0, state1[key])
        schedule[key] = np.array(layout)

    statekey = [1, 1, 2, 2] * num
    # statekey.insert(0, 1)
    statekey = np.array(statekey)

    model_results = run_profile(times, schedule, msid, model_spec, init, pseudo=pseudo)

    return model_results, times, statekey


def create_opt_fun(datesecs, dwell1_state, dwell2_state, t_dwell1_scaled, msid, limit, model_spec, init, t_backoff,
                   duration, scale_factor=1., max_dwell=1.e6):
    """ Generate a Xija model function with preset values, for use with an optimization routine.

    Args:
        datesecs (float, int): Date for start of simulation, in seconds from '1997:365:23:58:56.816'
            (Chandra.Time.DateTime epoch)
        dwell1_state (dict): States for fixed dwell (pitch, roll, ccds, etc.)
        dwell2_state (dict): States for variable dwell (pitch, roll, ccds, etc.)
        t_dwell1_scaled (float, int): Fixed dwell duration in seconds, this is in the SCALED format
        msid: msid (str): Primary MSID for model being run
        limit (float): Temperature limit for primary MSID in model for this simulation
        model_spec (dict, string): Dictionary of model parameters or file location where parameters can be imported
        init (dict): Dictionary of Xija model initialization parameters, can be empty
        t_backoff (float): Duration for tail end of simulated schedule used to determine convergence, defaults to 10
            days (in seconds)
        duration (float): Duration for entire simulated schedule, defaults to 30 days (in seconds)
        scale_factor (:obj:`float`, optional): scale factor for times, this is a way to scale the time values to help
            convergence, this does not affect the final `t_dwell2` value in any significant way (when it converges)
        max_dwell (:obj: `float`, optional): Maximum single dwell length

    Returns:
        function: Function generated from specified parameters, to be passed to optimization routine

    Notes:

        Keys in dwell1_state must match keys in dwell2_state.

        Keys in dwell1_state must match Xija component names (e.g. 'pitch', 'ccd_count', 'sim_z')

    """
    def opt_binary_schedule(t):
        model_results, _, _ = calc_binary_schedule(datesecs, dwell1_state, dwell2_state, t_dwell1_scaled, t, msid,
                                                   model_spec, init, duration=duration, t_backoff=t_backoff,
                                                   scale_factor=scale_factor, max_dwell=max_dwell)

        model_temps = model_results[msid].mvals
        model_times = model_results[msid].times
        ind = model_times > (model_times[-1] - t_backoff * 2)
        delta = np.abs(limit - np.max(model_temps[ind]))

        return delta

    return opt_binary_schedule


def find_second_dwell_sp(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, model_spec, init, scale_factor=1.,
                         max_iters=200, max_dwell=1.e6, temperature_precision=0.01, duration=2592000, t_backoff=1725000,
                         pseudo=None, shared_data=None):
    """ Determine the required dwell time at pitch2 to balance a given fixed dwell time at pitch1, if any exists.

    Args:
        date (float, int, str): Date for start of simulation, in seconds from '1997:365:23:58:56.816', or any other
            format readable by Chandra.Time.DateTime
        dwell1_state (dict): States for fixed dwell (pitch, roll, ccds, etc.)
        dwell2_state (dict): States for variable dwell (pitch, roll, ccds, etc.)
        t_dwell1 (float, int): Fixed dwell duration in seconds
        msid: msid (str): Primary MSID for model being run
        limit (float): Temperature limit for primary MSID in model for this simulation
        model_spec (dict, string): Dictionary of model parameters or file location where parameters can be imported
        init (dict): Dictionary of Xija model initialization parameters, can be empty
        scale_factor (:obj:`float`, optional): scale factor for times, this is a way to scale the time values to help
            convergence, this does not affect the final `t_dwell2` value in any significant way (when it converges)
        max_iters (:obj: `int`, optional): NOT CURRENTLY USED
        max_dwell (:obj: `float`, optional): Maximum single dwell length
        temperature_precision(:obj: `float`, optional): tolerance for determining convergence outside of the
            minimization routine, this should be greater than the `tol` keyword for the opt.minimization function call
        duration (float): Duration for entire simulated schedule, defaults to 30 days (in seconds)
        t_backoff (float): Duration for tail end of simulated schedule used to determine convergence, defaults to 10
            days (in seconds)
        debug (:obj: `bool`, optional): Boolean indicating whether or not to generate extra output
        pseudo (:obj:`str`, optional): Name of one or more pseudo MSIDs used in the model, if any, only necessary if one
            wishes to retrieve model results for this pseudo node, if it exists
        shared_data (:obj:`Multiprocessing.Manager.manager.list`, optional) Used to return data when using the
            multiprocessing package

    Returns:
        dict: Dictionary of results information
        dict: Dictionary of model run data, for debugging purposes, keys are node names (e.g. 'aacccdpt', 'aca0'),
            values are Xija model component objects, this is the same object returned by `run_profile` and
            `calc_binary_schedule`

    Example:

        model_specs = load_model_specs()

        init = {'1dpamzt': 20., 'dpa0': 20., 'eclipse': False, 'roll': 0, 'vid_board': True, 'clocking': True,
                'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}

        results = find_second_dwell_sp('2020:001:00:00:00', 156, 90, 20000, '1dpamzt', 36.5, model_specs['1dpamzt'],
                                       init, scale_factor=500., pseudo='dpa0')
    """

    datesecs = DateTime(date).secs

    results = {'converged': False, 'max_iteration': 0, 'iteration_limit': max_iters, 'unconverged_hot': False,
               'unconverged_cold': False, 'max_temp': np.nan, 'mean_temp': np.nan, 'temperature_limit': limit,
               'dwell_2_time': np.nan, 'dwell_2_time_limit': max_dwell, 'min_temp': np.nan, 'max_pseudo': np.nan,
               'min_pseudo': np.nan, 'hotter_state': np.nan, 'colder_state': np.nan}

    # Ensure t_dwell1 is a float
    t_dwell1 = np.float(t_dwell1)

    opt_fun = create_opt_fun(datesecs, dwell1_state, dwell2_state, t_dwell1/scale_factor, msid, limit, model_spec, init,
                             t_backoff, duration, scale_factor=scale_factor, max_dwell=max_dwell)

    res = opt.minimize(opt_fun, np.array([10,]), method='powell', options={'disp': False}, tol=0.00001)

    t_dwell2 = res.x * scale_factor

    # opt.minimize will return what it tried to use, not what I forced it to use. Unfortunately the bounded minimization
    # options did not provide adequate convergence performance, or required a jacobian.
    if t_dwell2 > max_dwell:
        t_dwell2 = max_dwell
    elif t_dwell2 < 0:
        t_dwell2 = 0

    results['dwell_2_time'] = t_dwell2
    results['max_iteration'] = res.nit

    model_results, state_times, state_nums = calc_binary_schedule(datesecs, dwell1_state, dwell2_state,
                                                                t_dwell1/scale_factor, [res.x, ], msid, model_spec,
                                                                init, t_backoff=t_backoff, duration=duration,
                                                                scale_factor=scale_factor, pseudo=pseudo)

    model_temps = model_results[msid].mvals
    model_times = model_results[msid].times

    ind_start = np.int(np.floor(len(model_times) / 3.))

    # state_times_mid = np.diff(state_times) / 2.0
    # state_times_mid = state_times_mid[state_times_mid > 0]
    # state_times_mid = state_times_mid + np.diff(state_times_mid) / 2.0


    ind = model_times > (model_times[-1] - t_backoff)
    max_temp = np.max(model_temps[ind])
    min_temp = np.min(model_temps[ind])
    mean_temp = np.mean(model_temps[ind])
    results['max_temp'] = max_temp
    results['min_temp'] = min_temp
    results['mean_temp'] = mean_temp

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
        state_nums_interp = np.around(np.interp(model_times[ind_start:], state_times, state_nums))
        model_temps_sign = np.array(list(np.sign(np.diff(model_temps[ind_start:],))) + [0,])
        state_1_sign = np.mean(model_temps_sign[state_nums_interp == 1])
        state_2_sign = np.mean(model_temps_sign[state_nums_interp == 2])

        # print('state1: {}, state2: {}'.format(state_1_sign, state_2_sign))

        results['hotter_state'] = 1 if state_1_sign > state_2_sign else 2
        results['colder_state'] = 2 if state_1_sign > state_2_sign else 1

    # print(results)
    # print(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, init, scale_factor,max_iters, max_dwell,
    #       temperature_precision, duration, t_backoff,debug, pseudo)

    if shared_data is not None:
        shared_data.append((scale_factor, results))
    else:
        return results, model_results


def run_state_pairs(msid, model_spec, init, limit, date, state_pairs, scale_factors=(200, 500, 1000, 2000),
                    pseudo=None):


    # NOTES:
    # There are some minimum fields that must be in state pairs


    duration = 30 * 24 * 3600.
    t_backoff = 2 * duration / 3


    datestr = DateTime(date).date[:8]

    results = []

    num = np.float(len(state_pairs))
    for n, pair in enumerate(state_pairs):
        print("Running simulations for state pair #: {} out of {}".format(n + 1, num))

        dwell1_state = pair[0]
        dwell2_state = pair[1]

        duration1 = dwell1_state.pop('duration1')
        sequence1 = dwell1_state.pop('sequence1')
        obsid1 = dwell1_state.pop('obsid1')
        sequence2 = dwell2_state.pop('sequence2')
        obsid2 = dwell2_state.pop('obsid2')
        duration1_fraction = dwell1_state.pop('duration1_fraction')

        arguments = (date, dwell1_state, dwell2_state, duration1, msid, limit, model_spec, init)

        manager = Manager()
        cases = manager.list()
        jobs = []

        for scale_factor in scale_factors:
            # results, model_results = find_second_dwell_sp(date, dwell1_state, dwell2_state, t_dwell1, msid, limit,
            #                                               model_spec, init, debug=False, scale_factor=scale_factor)

            keyword_arguments = {'scale_factor': scale_factor, 'pseudo': pseudo, 'shared_data': cases,
                                 't_backoff': t_backoff}
            p = Process(target=find_second_dwell_sp, args=arguments, kwargs=keyword_arguments)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        # results = {'converged': False, 'max_iteration': 0, 'iteration_limit': max_iters, 'unconverged_hot': False,
        #            'unconverged_cold': False, 'max_temp': np.nan, 'mean_temp': np.nan, 'temperature_limit': limit,
        #            'dwell_2_time': np.nan, 'dwell_2_time_limit': max_dwell, 'min_temp': np.nan, 'max_pseudo': np.nan,
        #            'min_pseudo': np.nan}

        # Join Results
        converged = np.array([case[1]['converged'] for case in cases])
        unconverged_hot = np.array([case[1]['unconverged_hot'] for case in cases])
        unconverged_cold = np.array([case[1]['unconverged_cold'] for case in cases])
        max_temp = np.array([case[1]['max_temp'] for case in cases])
        min_temp = np.array([case[1]['min_temp'] for case in cases])
        mean_temp = np.array([case[1]['mean_temp'] for case in cases])
        max_pseudo_temp = np.array([case[1]['max_pseudo_temp'] for case in cases]) if pseudo is not None else np.nan
        min_pseudo_temp = np.array([case[1]['min_pseudo_temp'] for case in cases]) if pseudo is not None else np.nan
        mean_pseudo_temp = np.array([case[1]['mean_pseudo_temp'] for case in cases]) if pseudo is not None else np.nan
        dwell2_time = np.array([case[1]['dwell_2_time'] for case in cases])
        hotter_state = np.array([case[1]['hotter_state'] for case in cases])
        colder_state = np.array([case[1]['colder_state'] for case in cases])

        row = (msid,
               datestr,
               obsid1,
               sequence1,
               duration1_fraction,
               obsid2,
               sequence2,
               limit,
               dwell1_state['pitch'],
               dwell2_state['pitch'],
               dwell1_state['roll'] if 'roll' in dwell1_state else 0,
               dwell2_state['roll'] if 'roll' in dwell1_state else 0,
               dwell1_state['ccd_count'] if 'ccd_count' in dwell1_state else 0,
               dwell2_state['ccd_count'] if 'ccd_count' in dwell1_state else 0,
               dwell1_state['fep_count'] if 'fep_count' in dwell1_state else 0,
               dwell2_state['fep_count'] if 'fep_count' in dwell1_state else 0,
               dwell1_state['clocking'] if 'clocking' in dwell1_state else 0,
               dwell2_state['clocking'] if 'clocking' in dwell1_state else 0,
               dwell1_state['vid_board'] if 'vid_board' in dwell1_state else 0,
               dwell2_state['vid_board'] if 'vid_board' in dwell1_state else 0,
               dwell1_state['sim_z'] if 'sim_z' in dwell1_state else 0,
               dwell2_state['sim_z'] if 'sim_z' in dwell1_state else 0,
               dwell1_state,
               dwell2_state,
               duration1,
               np.mean(dwell2_time[converged]) if np.any(converged) else np.nan,
               np.mean(min_temp[converged]) if np.any(converged) else np.nan,
               np.mean(max_temp[converged]) if np.any(converged) else np.nan,
               np.mean(mean_temp[converged]) if np.any(converged) else np.nan,
               np.mean(min_pseudo_temp[converged]) if pseudo is not None and np.any(converged) else np.nan,
               np.mean(max_pseudo_temp[converged]) if pseudo is not None and np.any(converged) else np.nan,
               np.mean(mean_pseudo_temp[converged]) if pseudo is not None and np.any(converged) else np.nan,
               True if np.any(converged) else False,
               True if ~np.any(converged) and np.mean(unconverged_hot) > 0.5 else False,
               True if ~np.any(converged) and np.mean(unconverged_cold) > 0.5 else False,
               np.around(np.mean(hotter_state[converged])) if np.any(converged) else 0,
               np.around(np.mean(colder_state[converged])) if np.any(converged) else 0,
               )

        results.append(row)

    return np.array(results, dtype=results_dtype)


# ----------------------------------------------------------------------------------------------------------------------
# All functions defined after this point will be deprecated in a future version
#
# Notes: A numpy structured array with named columns replaces the original pitch-focused grid of dwell times. This newer
#        format is less dependant on one particular input parameter (i.e. pitch) and is easier to work with in a
#        vectorized manner. It also runs faster, allows for all data for a particular model to be stored in a single
#        file, and is better suited to help evaluate a single schedule as it can take an arbitrary list of initial and
#        final states.
#
# ----------------------------------------------------------------------------------------------------------------------

def char_model(msid, model_spec, init, limit, date, t_dwell1, pitch_step=1, scale_factor=500., pseudo=None):
    """Characterize Xija model over a range of paired pitches, given a fixed dwell time for pitch #1.

    Args:
        msid: msid (str): Primary MSID for model being run
        model_spec (dict, string): Dictionary of model parameters or file location where parameters can be imported
        init (dict): Dictionary of Xija model initialization parameters, can be empty
        limit (float): Temperature limit for primary MSID in model for this simulation
        date (float, int, str): Date for start of simulation, in seconds from '1997:365:23:58:56.816', or any other
            format readable by Chandra.Time.DateTime
        t_dwell1 (float, int): Fixed dwell duration in seconds
        pitch_step (:obj: `int`, optional): Resolution in pitch
        scale_factor (:obj:`float`, optional): scale factor for times, this is a way to scale the time values to help
            convergence, this does not affect the final `t_dwell2` value in any significant way (when it converges)
        pseudo (:obj:`str`, optional): Name of one or more pseudo MSIDs used in the model, if any, only necessary if one
            wishes to retrieve model results for this pseudo node, if it exists

    Returns:
        (tuple): tuple containing:
            (numpy.ndarray) 2D Array of dwell 2 times in seconds, size is No. Pitch1 values by No. Pitch 2 values
            (numpy.ndarray) 2D Array of boolean values indicating if a particular case converged, same size as above
            (numpy.ndarray) 1D Array of initial pitch values
            (numpy.ndarray) 1D Array of final pitch values
            (numpy.ndarray) 3D Array of min and max primary node temperatures, first two dimensions are the same as the
                            first array, min and max form the two planes in the third dimension
            (numpy.ndarray) 3D Array of min and max pseudo node temperatures, if any, first two dimensions are the same
                            as the first array, min and max form the two planes in the third dimension

    """
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

        dwell1_state = {'pitch': p1}
        for m, p2 in enumerate(pitch2):

            dwell2_state = {'pitch': p2}

            results, model_results = find_second_dwell_sp(date, dwell1_state, dwell2_state, t_dwell1, msid, limit,
                                                          model_spec, init, debug=False, scale_factor=scale_factor,
                                                          pseudo='aca0')
            if results['converged']:
                converged[n, m] = True
                t_dwell2[n, m] = results['dwell_2_time']
                temperature_range[n, m, 0] = results['max_temp']
                temperature_range[n, m, 1] = results['min_temp']

                if pseudo:
                    pseudo_range[n, m, 0] = results['max_pseudo']
                    pseudo_range[n, m, 1] = results['min_pseudo']

    return t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range


def characterization(msid, limit, date, t_dwell1, step, init, model_spec, scale_factor, shared_data, pseudo=None):

    date = DateTime(date).date
    datestr = date[:4] + date[5:8]

    msg = 'Running: {} with a fixed initial dwell of {}ks, for {}, with a {} degree pitch step and {} scale factor,' + \
          ' Notes: {}'
    print(msg.format(msid, int(t_dwell1), datestr, step, scale_factor, 'None'))

    t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range = char_model(msid, model_spec, init, limit,
                                                                                      date, t_dwell1,
                                                                                      pitch_step=step,
                                                                                      scale_factor=scale_factor,
                                                                                      pseudo=pseudo)

    save_characterization(pitch1, pitch2, t_dwell2, msid, t_dwell1, datestr, limit, step,
                          pad_text='_{}_scale_factor'.format(int(scale_factor)))
    save_characterization(pitch1, pitch2, temperature_range[:, :, 0], msid, t_dwell1, datestr, limit, step,
                          pad_text='_{}_scale_factor_max_temp'.format(int(scale_factor)))
    save_characterization(pitch1, pitch2, temperature_range[:, :, 1], msid, t_dwell1, datestr, limit, step,
                          pad_text='_{}_scale_factor_min_temp'.format(int(scale_factor)))
    save_characterization(pitch1, pitch2, pseudo_range[:, :, 0], msid, t_dwell1, datestr, limit, step,
                          pad_text='_{}_scale_factor_max_pseudo'.format(int(scale_factor)))
    save_characterization(pitch1, pitch2, pseudo_range[:, :, 1], msid, t_dwell1, datestr, limit, step,
                          pad_text='_{}_scale_factor_min_pseudo'.format(int(scale_factor)))

    if shared_data is not None:
        shared_data.append((scale_factor, t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range))
    else:
        return t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range


def extract_data_from_file(filename):
    d = np.genfromtxt(filename, delimiter=',')
    pitch1 = d[1:, 0]
    pitch2 = d[0, 1:]
    t_dwell2 = d[1:, 1:]
    return t_dwell2, pitch1, pitch2


def save_characterization(pitch1, pitch2, t_dwell2, msid, t_dwell1, datestr, limit, step, pad_text=''):

    tc = DataFrame(t_dwell2)
    r = dict(zip(np.arange(0, len(pitch1)), pitch1))
    c = dict(zip(np.arange(0, len(pitch2)), pitch2))
    tc = tc.rename(index=r, columns=c)
    filename = '{}_fixed_initial_dwell_{}ks_{}_{}_{}deg_step{}.csv'.format(msid, int(t_dwell1), datestr, str(limit),
                                                                           step, pad_text)
    tc.to_csv(filename)
    print('Saved to: {}'.format(filename))


def combine_characterizations(cases, len1, len2):
    t2 = np.reshape(np.vstack(cases), (len(cases), len1, len2))
    t2[t2 == 0] = np.nan
    t2 = np.nanmean(t2, axis=0)
    t2[np.isnan(t2)] = 0
    return t2


def run_characterization_all(msid, limit, date, t_dwell1, step, init, model_spec, scale_factors=(200, 500, 1000, 2000)):

    date = DateTime(date).date
    datestr = date[:4] + date[5:8]

    cases = []
    pitch1 = None
    pitch2 = None

    for scale_factor in scale_factors:
        t_dwell2, converged, pitch1, pitch2, temperature_range, pseudo_range = \
            characterization(msid, limit, date, t_dwell1, step, init, model_spec, scale_factor, None)
        cases.append(t_dwell2)

    t_dwell2 = combine_characterizations(cases, len(pitch1), len(pitch2))
    save_characterization(pitch1, pitch2, t_dwell2, msid, t_dwell1, datestr, limit, step, pad_text='_averaged')


def run_characterization_parallel(msid, limit, date, t_dwell1, step, init, model_spec,
                                  scale_factors=(200, 500, 1000, 2000), pseudo=None):

    date = DateTime(date).date
    datestr = date[:4] + date[5:8]

    manager = Manager()
    cases = manager.list()
    jobs = []

    for scale_factor in scale_factors:
        p = Process(target=characterization, args=(msid, limit, date, t_dwell1, step, init, model_spec, scale_factor,
                                                   cases), kwargs={'pseudo': pseudo})
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    dwell2_cases = [c[1] for c in cases]
    pitch1 = cases[0][3]
    pitch2 = cases[0][4]

    t_dwell2 = combine_characterizations(dwell2_cases, len(pitch1), len(pitch2))
    save_characterization(pitch1, pitch2, t_dwell2, msid, t_dwell1, datestr, limit, step, pad_text='_averaged')


pseudo_names = dict(
    zip(['aacccdpt', 'pftank2t', '1dpamzt', '4rt700t', '1deamzt'], ['aca0', 'pf0tank2t', 'dpa0', 'oba0', None]))


if __name__ == '__main__':
    model_init = {'aacccdpt': {'aacccdpt': -10., 'aca0': -10., 'eclipse': False},
                  'pftank2t': {'pftank2t': f_to_c(95.), 'pf0tank2t': f_to_c(95.), 'eclipse': False},
                  'tcylaft6': {'tcylaft6': f_to_c(120.), 'cc0': f_to_c(120.), 'eclipse': False},
                  '4rt700t': {'4rt700t': f_to_c(95.), 'oba0': f_to_c(95.), 'eclipse': False},
                  '1dpamzt': {'1dpamzt': 35., 'dpa0': 35., 'eclipse': False, 'vid_board': True, 'clocking': True,
                              'dpa_power': 0.0, 'sim_z': 100000},
                  '1deamzt': {'1deamzt': 35., 'eclipse': False, 'vid_board': True, 'clocking': True, 'dpa_power': 0.0,
                              'sim_z': 100000}}

    model_specs = load_model_specs()

    date = '2019:098:00:00:00'
    scale_factor = 1000.
    t_dwell1 = 10000.

    msid = 'aacccdpt'
    limit = -9.5

    state_pairs = (({'sequence1': 1000, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': 10000, 'pitch': 144.2}, {'sequence2': 2000, 'obsid2': 22222, 'pitch': 154.95}),
                   ({'sequence1': 1000, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': 10000, 'pitch': 90}, {'sequence2': 3000, 'obsid2': 33333,'pitch': 170}),
                   ({'sequence1': 1000, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': 10000, 'pitch': 50}, {'sequence2': 4000, 'obsid2': 44444,'pitch': 140}),
                   ({'sequence1': 1000, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': 10000, 'pitch': 90}, {'sequence2': 5000, 'obsid2': 55555,'pitch': 100}),
                   ({'sequence1': 1000, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': 10000, 'pitch': 75}, {'sequence2': 6000, 'obsid2': 66666,'pitch': 130}),
                   ({'sequence1': 1000, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': 10000, 'pitch': 170}, {'sequence2': 7000, 'obsid2': 77777,'pitch': 90}),
                   ({'sequence1': 1000, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': 10000, 'pitch': 90}, {'sequence2': 8000, 'obsid2': 88888,'pitch': 170}))
    results = run_state_pairs(msid, model_specs[msid], model_init[msid], limit, date, state_pairs)

    print(results)

    dwell1_state = {'pitch': 144.2}
    dwell2_state = {'pitch': 154.95}

    results, _ = find_second_dwell_sp(date, dwell1_state, dwell2_state, t_dwell1, msid,
                                                             limit, model_specs[msid], model_init[msid],
                                                             max_dwell=1e6, scale_factor=200., t_backoff=1725000)

    print(results)

    results, _ = find_second_dwell_sp(date, dwell1_state, dwell2_state, t_dwell1, msid,
                                                             limit, model_specs[msid], model_init[msid],
                                                             max_dwell=1e6, scale_factor=100., t_backoff=1725000)

    print(results)

    results, _ = find_second_dwell_sp(date, dwell1_state, dwell2_state, t_dwell1, msid,
                                                             limit, model_specs[msid], model_init[msid],
                                                             max_dwell=1e6, scale_factor=1000., t_backoff=1725000)

    print(results)

    results, _ = find_second_dwell_sp(date, dwell1_state, dwell2_state, t_dwell1, msid,
                                                             limit, model_specs[msid], model_init[msid],
                                                             max_dwell=1e6, scale_factor=2000., t_backoff=1725000)

    print(results)