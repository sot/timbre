# Licensed under a 3-clause BSD style license - see LICENSE.rst

from hashlib import md5
from json import loads as json_loads
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urljoin
import json
from git import Repo
import numpy as np
from scipy import interpolate

from cxotime import CxoTime
import xija
from xija import get_model_spec

non_state_names = {'aacccdpt': ['aca0', ],
                   'pftank2t': ['pf0tank2t', ],
                   '4rt700t': ['oba0', ],
                   'pline03t': ['pline03t0', ],
                   'pline04t': ['pline04t0', ],
                   'pm1thv2t': ['mups0', ],
                   'pm2thv1t': ['mups0', 'mups1'],
                   '1deamzt': ['dea0', ],
                   '1dpamzt': ['dpa0', ],
                   'fptemp_11': ['fptemp', '1cbat', 'sim_px'],
                   '1pdeaat': ['pin1at', ]}


def get_github_chandra_models_version_info():
    """ Download a list of all tags and branches, along with associated information.

    :return: Dictionary of all tags and branches, along with associated information.
    :rtype: dict
    """
    with urlopen('https://api.github.com/repos/sot/chandra_models/tags') as url:
        response = url.read()
        tags = json.loads(response.decode('utf-8'))

    with urlopen('https://api.github.com/repos/sot/chandra_models/branches') as url:
        response = url.read()
        branches = json.loads(response.decode('utf-8'))

    all_versions_info = {t["name"]: t for t in tags}
    all_versions_info.update({b["name"]: b for b in branches})
    return all_versions_info


def load_github_model_specs(version='master'):
    """ Load Xija all model parameters for a specified version from https://github.com/sot/chandra_models.

    :param version: tag or branch to use
    :type version: str

    :return: A dictionary containing the model specifications for all available Xija models for the chandra_models
        version specified
    :rtype: dict

    Note:
    This will need to be updated as new models are approved or existing models are renamed.
    """

    def join_url_parts(repository_root, url_parts):
        return urljoin(repository_root, '/'.join(url_parts).replace('///', '/').replace('//', '/'))

    def get_model(model_location):
        """ Load parameters for a single Xija model.

        :param model_location: Relative location of model file, starting from the chandra_models root repository
            location

        :return: JSON file stored as a dictionary, md5 hash of file
        """

        repository_url = 'https://raw.githubusercontent.com/sot/chandra_models/'
        model_spec_url = join_url_parts(repository_url, [version, model_location])

        with urlopen(model_spec_url) as url:
            response = url.read()
            f = response.decode('utf-8')

        md5_hash = md5(f.encode('utf-8')).hexdigest()

        return json_loads(f), md5_hash

    model_locations = {
        'aacccdpt': '/chandra_models/xija/aca/aca_spec.json',
        '1deamzt': '/chandra_models/xija/dea/dea_spec.json',
        '1dpamzt': '/chandra_models/xija/dpa/dpa_spec.json',
        'fptemp': '/chandra_models/xija/acisfp/acisfp_spec_matlab.json',
        '1pdeaat': '/chandra_models/xija/psmc/psmc_spec.json',
        'pftank2t': '/chandra_models/xija/pftank2t/pftank2t_spec.json',
        '4rt700t': '/chandra_models/xija/fwdblkhd/4rt700t_spec.json',
        'pline03t': '/chandra_models/xija/pline/pline03t_model_spec.json',
        'pline04t': '/chandra_models/xija/pline/pline04t_model_spec.json',
        'pm1thv2t': '/chandra_models/xija/mups_valve/pm1thv2t_spec.json',
        'pm2thv1t': '/chandra_models/xija/mups_valve/pm2thv1t_spec_matlab.json',
    }

    all_versions_info = get_github_chandra_models_version_info()

    model_specs = {'sha': all_versions_info[version]['commit']['sha'], 'version_info': all_versions_info[version],
                   'version': version}

    for msid, path in model_locations.items():
        model_specs[msid], model_specs[msid + '_md5'] = get_model(path)
    model_specs['fptemp_11'] = model_specs['fptemp']  # For backwards compatibility
    model_specs['fptemp_11_md5'] = model_specs['fptemp_md5']  # For backwards compatibility

    return model_specs


def load_model_specs(version=None, local_repository_location=None):
    """ Load Xija model parameters for all available models.

    :param version: tag or branch to use
    :type version: str
    :param local_repository_location: location of chandra_models repository, defaults to `get_model_spec.REPO_PATH`
    :type local_repository_location: str

    :return: A dictionary containing the model specifications for all available Xija models, along with latest commit
        sha for version specified (or existing branch/tag), branch/tag, and repository state
    :rtype: dict

    Note:
    This will need to be updated as new models are approved or existing models are renamed.
    """

    def get_local_git_version_info(repo):
        """ Get latest git commit hash for current branch.

        :param repo: gitpython Repo object for repository
        :type repo: git.repo.base.Repo
        :return: Latest commit hash, branch/tag, repository state
        :return: dict
        """

        hexsha, version = repo.commit().name_rev.split()
        modified = repo.is_dirty()
        return {'sha': hexsha, 'version': version, 'modified': modified}

    def get_model(model_location):
        """ Load parameters for a single Xija model.

        :param model_location: Relative location of model file, starting from the chandra_models root repository
            location
        :type model_locations: str

        :return: JSON file stored as a dictionary, md5 hash of file
        :rtype: tuple
        """

        with open(Path.joinpath(local_repository_location, Path(model_location))) as fid:
            f = fid.read()
        md5_hash = md5(f.encode('utf-8')).hexdigest()
        return json_loads(f), md5_hash

    model_locations = {
        'aacccdpt': 'chandra_models/xija/aca/aca_spec.json',
        '1deamzt': 'chandra_models/xija/dea/dea_spec.json',
        '1dpamzt': 'chandra_models/xija/dpa/dpa_spec.json',
        'fptemp': 'chandra_models/xija/acisfp/acisfp_spec_matlab.json',
        '1pdeaat': 'chandra_models/xija/psmc/psmc_spec.json',
        'pftank2t': 'chandra_models/xija/pftank2t/pftank2t_spec.json',
        '4rt700t': 'chandra_models/xija/fwdblkhd/4rt700t_spec.json',
        'pline03t': 'chandra_models/xija/pline/pline03t_model_spec.json',
        'pline04t': 'chandra_models/xija/pline/pline04t_model_spec.json',
        'pm1thv2t': 'chandra_models/xija/mups_valve/pm1thv2t_spec.json',
        'pm2thv1t': 'chandra_models/xija/mups_valve/pm2thv1t_spec_matlab.json',
    }

    if local_repository_location is None:
        local_repository_location = get_model_spec.REPO_PATH
    else:
        local_repository_location = Path(local_repository_location).expanduser()

    with get_model_spec.temp_directory() as repo_path_local:
        repo = Repo.clone_from(local_repository_location, repo_path_local)
        if version is not None:
            _ = repo.git.checkout(version)
        model_specs = get_local_git_version_info(repo)

        for msid, path in model_locations.items():
            model_specs[msid], model_specs[msid + '_md5'] = get_model(path)
        model_specs['fptemp_11'] = model_specs['fptemp']  # For backwards compatibility
        model_specs['fptemp_11_md5'] = model_specs['fptemp_md5']  # For backwards compatibility

    return model_specs


def get_local_model(filename):
    """ Load parameters for a single Xija model.

    :param filename: File path to local model specification file
    :type filename: str
    :return: Model spec as a dictionary, md5 hash of model spec
    :rtype: tuple
    """

    with open(filename) as fid:  # 'aca/aca_spec.json', 'rb') as fid:
        f = fid.read()

    return json.loads(f), md5(f.encode('utf-8')).hexdigest()


def c_to_f(temp):
    """ Convert Celsius to Fahrenheit.

    :param temp: Temperature in Celsius
    :type temp: int or float or tuple or list or np.ndarray
    :return: Temperature in Fahrenheit
    :rtype: int or float or list or np.ndarray
    """
    if type(temp) is list or type(temp) is tuple:
        return [c * 1.8 + 32 for c in temp]
    else:
        return temp * 1.8 + 32.0


def f_to_c(temp):
    """ Convert Fahrenheit to Celsius.

    :param temp: Temperature in Fahrenheit
    :type temp: int or float or tuple or list or np.ndarray
    :return: Temperature in Celsius
    :rtype: int or float or list or np.ndarray
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

    :param msid: Primary MSID for model; in this case it can be anything as it is only being used to name the model,
           however keeping the convention to name the model after the primary MSID being predicted reduces confusion
    :type msid: str
    :param t0: Start time for model prediction; this can be any format that cxotime.CxoTime accepts
    :type t0: str or float or int
    :param t1: End time for model prediction; this can be any format that cxotime.CxoTime accepts
    :type t1: str or float or int
    :param model_spec: Dictionary of model parameters or file location where parameters can be imported
    :type model_spec: dict, str
    :param init: Dictionary of Xija model initialization parameters, can be empty
    :type init: dict
    :rtype: xija.model.XijaModel

    Example::

        model_specs = load_model_specs()
        init = {'1dpamzt': 35., 'dpa0': 35., 'eclipse': False, 'roll': 0, 'vid_board': True, 'pitch':155,
                'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}
        model = setup_model('1dpamzt', '2019:001:00:00:00', '2019:010:00:00:00', model_specs['1dpamzt'], init)

    Notes:

     - This does not run the model, only sets up the model to be run.
     - Any parameters not specified in `init` will either need to be pulled from telemetry or explicitly defined \
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
    """ Run a Xija model for a given time and state profile.

    :param times: Array of time values, in seconds from '1997:365:23:58:56.816' (cxotime.CxoTime epoch)
    :type times: np.ndarray
    :param schedule: Dictionary of pitch, roll, etc. values that match the time values specified above in `times`
    :type schedule: dict
    :param msid: Primary MSID for model being run
    :type msid: str
    :param model_spec: Dictionary of model parameters or file location where parameters can be imported
    :type model_spec: dict or string
    :param init: Dictionary of Xija model initialization parameters, can be empty but not recommended
    :type init: dict
    :param pseudo: Name of one or more pseudo MSIDs used in the model, if any, only necessary if one
            wishes to retrieve model results for this pseudo node, if it exists
    :type pseudo: str or None, optional
    :returns: Results, keys are node names (e.g. 'aacccdpt', 'aca0'), values are Xija model component objects
    :rtype: dict

    Example::

        times = np.array(CxoTime(['2019:001:00:00:00', '2019:001:12:00:00', '2019:002:00:00:00',
                                  '2019:003:00:00:00']).secs)
        pitch = np.array([150, 90, 156, 156])
        schedule = {'pitch': pitch}
        model_specs = load_model_specs()
        init = {'1dpamzt': 20., 'dpa0': 20., 'eclipse': False, 'roll': 0, 'vid_board': True, 'clocking': True,
                'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}
        results = run_profile(times, pitch, '1dpamzt', model_specs['1dpamzt'], init, pseudo='dpa0')

    Note:

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
        results[pseudo] = model.get_comp(pseudo)

    return results


def calc_binary_schedule(datesecs, state1, state2, t_dwell1, t_dwell2, msid, model_spec, init, duration=2592000.,
                         t_backoff=1725000., pseudo=None):
    """ Simulate a schedule that switches between two states

    This runs the model over a "binary" schedule. This function is intended to be used to optimize the `t_dwell2`
    parameter so that the predicted temperature during the last `t_backoff` number of seconds reaches a limit within a
    tolerance (limit used and specified in a different function).

    :param datesecs: Date for start of simulation, in seconds from '1997:365:23:58:56.816' (cxotime.CxoTime epoch)
    :type datesecs: float or int
    :param state1: States for fixed dwell (pitch, roll, ccds, etc.)
    :type state1: dict
    :param state2: States for variable dwell (pitch, roll, ccds, etc.)
    :type state2: dict
    :param t_dwell1: Fixed dwell duration in seconds
    :type t_dwell1: float or int
    :param t_dwell2: Variable dwell duration in seconds (this is the parameter that is optimized)
    :type t_dwell2: float or int
    :param msid: Primary MSID for model being run
    :type msid: str
    :param model_spec: Dictionary of model parameters or file location where parameters can be imported
    :type model_spec: dict, string
    :param init: Dictionary of Xija model initialization parameters
    :type init: dict
    :param duration: Duration for entire simulated schedule, defaults to 30 days (in seconds)
    :type duration: float, optional
    :param t_backoff: Duration for tail end of simulated schedule used to determine convergence, defaults to 10 days
        (in seconds)
    :type t_backoff: float, optional
    :param pseudo: Name of one or more pseudo MSIDs used in the model, if any, only necessary if one wishes to retrieve
        model results for this pseudo node, if it exists. This currently is not used but kept here as a placeholder.
    :type pseudo: str, optional
    :returns:
        - **results** (:py:class:`dict`) - keys are node names (e.g. 'aacccdpt', 'aca0'), values are Xija model
            component objects, this is the same object returned by `run_profile`
        - **times** (:py:class:`np.ndarray`) - time values input into Xija (may not exactly match Xija output)
        - **state_keys** (:py:class:`np.ndarray`) - defines state order, with elements matching the time array output
            (may not exactly match Xija output), this defines where to insert what state
    :rtype: tuple

    Notes:

        - Keys in state1 must match keys in state2.
        - Keys in state1 must match Xija component names (e.g. 'pitch', 'ccd_count', 'sim_z')
    """

    num = int(duration / (t_dwell1 + t_dwell2))
    reltimes = np.cumsum([1, t_dwell1 - 1, 1, t_dwell2 - 1] * num)
    times = np.array(reltimes) - reltimes[0] + datesecs - t_backoff

    schedule = dict(zip(state1.keys(), []))
    for key, value in state1.items():
        layout = [state1[key], state1[key], state2[key], state2[key]] * num
        schedule[key] = np.array(layout)

    state_keys = [1, 1, 2, 2] * num
    state_keys = np.array(state_keys)

    model_results = run_profile(times, schedule, msid, model_spec, init, pseudo=pseudo)

    return model_results, times, state_keys


def create_opt_fun(datesecs, dwell1_state, dwell2_state, t_dwell1, msid, model_spec, init, t_backoff, duration):
    """ Generate a Xija model function with preset values, for use with an optimization routine.

    :param datesecs: Date for start of simulation, in seconds from '1997:365:23:58:56.816' (cxotime.CxoTime epoch)
    :type datesecs: float or int
    :param dwell1_state: States for fixed dwell (pitch, roll, ccds, etc.)
    :type dwell1_state: dict
    :param dwell2_state: States for variable dwell (pitch, roll, ccds, etc.)
    :type dwell2_state: dict
    :param t_dwell1: Fixed dwell duration in seconds
    :type t_dwell1: float or int
    :param msid: Primary MSID for model being run
    :type msid: str
    :param model_spec: Dictionary of model parameters or file location where parameters can be imported
    :type model_spec: dict, string
    :param init: Dictionary of Xija model initialization parameters
    :type init: dict
    :param t_backoff: Duration for tail end of simulated schedule used to determine convergence, defaults to 10 days
        (in seconds)
    :type t_backoff: float, optional
    :param duration: Duration for entire simulated schedule, defaults to 30 days (in seconds)
    :type duration: float, optional
    :returns: Function generated from specified parameters, to be passed to optimization routine
    :rtype: function

    Notes:

        - Keys in state1 must match keys in state2.
        - Keys in state1 must match Xija component names (e.g. 'pitch', 'ccd_count', 'sim_z')
    """

    def opt_binary_schedule(t):
        model_results, _, _ = calc_binary_schedule(datesecs, dwell1_state, dwell2_state, t_dwell1, t, msid,
                                                   model_spec, init, duration=duration, t_backoff=t_backoff)

        model_temps = model_results[msid].mvals
        model_times = model_results[msid].times
        ind = model_times > (model_times[-1] - t_backoff)
        dmax = np.max(model_temps[ind])
        dmin = np.min(model_temps[ind])
        dmean = np.mean(model_temps[ind])

        return t, dmax, dmean, dmin

    return opt_binary_schedule


def find_second_dwell(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, model_spec, init, limit_type='max',
                      duration=2592000, t_backoff=1725000, n_dwells=10, min_dwell=None, max_dwell=None, pseudo=None):
    """ Determine the required dwell time at pitch2 to balance a given fixed dwell time at pitch1, if any exists.

    :param date: Date for start of simulation, in seconds from '1997:365:23:58:56.816' (cxotime.CxoTime epoch) or any
        other format readable by cxotime.CxoTime
    :type date: float or int or str
    :param dwell1_state: States for fixed dwell (pitch, roll, ccds, etc.)
    :type dwell1_state: dict
    :param dwell2_state: States for variable dwell (pitch, roll, ccds, etc.)
    :type dwell2_state: dict
    :param t_dwell1: Fixed dwell duration in seconds
    :type t_dwell1: float or int
    :param msid: Primary MSID for model being run
    :type msid: str
    :param limit: Temperature limit for primary MSID in model for this simulation
    :type limit: float
    :param model_spec: Dictionary of model parameters or file location where parameters can be imported
    :type model_spec: dict, string
    :param init: Dictionary of Xija model initialization parameters
    :type init: dict
    :param limit_type: Type of limit, defaults to 'max' (a maximum temperature limit), other option is 'min'
    :type limit_type: str, optional
    :param duration: Duration for entire simulated schedule, defaults to 30 days (in seconds)
    :type duration: float, optional
    :param t_backoff: Duration for tail end of simulated schedule used to determine convergence, defaults to 10 days
        (in seconds)
    :type t_backoff: float, optional
    :param n_dwells: Number of second dwell, `t_dwell2`,  possibilities to run (more dwells = finer resolution)
    :type n_dwells: int, optional
    :param min_dwell: Minimum duration for second dwell, can be used if the user wants to narrow the dwell time search,
        defaults to 1.0e-6s
    :type min_dwell: float, optional
    :param max_dwell: Maximum duration for second dwell, can be used if the user wants to narrow the dwell time search
    :type max_dwell: float, optional
    :param pseudo: Name of one or more pseudo MSIDs used in the model, if any, only necessary if one wishes to retrieve
        model results for this pseudo node, if it exists. This currently is not used but kept here as a placeholder.
    :type pseudo: str, optional
    :returns: Dictionary of results information
    :rtype: dict
    """

    datesecs = CxoTime(date).secs
    msid = msid.lower()

    if 'max' in limit_type.lower():
        limit_type = 'max'
    else:
        limit_type = 'min'

    if max_dwell is None:
        # This ensures three "cycles" of the two dwell states, within the portion of the schedule used for evaluation
        # (t_backoff).
        # Subtract 1000 sec for extra padding.
        max_dwell = (t_backoff - t_dwell1) / 3 - 1000

    if min_dwell is None:
        min_dwell = 1.0e-6

    results = {'converged': False, 'unconverged_hot': False, 'unconverged_cold': False,
               'min_temp': np.nan, 'mean_temp': np.nan, 'max_temp': np.nan, 'temperature_limit': limit,
               'dwell_2_time': np.nan, 'min_pseudo': np.nan, 'mean_pseudo': np.nan, 'max_pseudo': np.nan,
               'hotter_state': np.nan, 'colder_state': np.nan}

    # Ensure t_dwell1 is a float, may not be necessary anymore
    t_dwell1 = float(t_dwell1)

    opt_fun = create_opt_fun(datesecs, dwell1_state, dwell2_state, t_dwell1, msid, model_spec, init, t_backoff,
                             duration)

    # First just check the bounds to avoid unnecessary runs of `opt_fun`
    output = np.array([opt_fun(t) for t in [min_dwell, max_dwell]],
                      dtype=[('duration2', float), ('max', float), ('mean', float), ('min', float)])

    if 'max' in limit_type:

        # All cases report temperatures entirely below the limit.
        if np.all(output['max'] < limit):
            results = _handle_unconverged_cold(output, results)

        # All cases report temperatures entirely above the limit.
        elif np.all(output['max'] > limit):
            results = _handle_unconverged_hot(output, results)

        # Temperatures straddle the limit, so a refined dwell 2 time is possible.
        else:
            results, output = _refine_dwell2_time('max', n_dwells, min_dwell, max_dwell, limit, opt_fun, results)

    elif 'min' in limit_type:

        # All cases report temperatures entirely below the limit.
        if np.all(output['min'] < limit):
            results = _handle_unconverged_cold(output, results)

        # All cases report temperatures entirely above the limit.
        elif np.all(output['min'] > limit):
            results = _handle_unconverged_hot(output, results)

        # Temperatures straddle the limit, so a refined dwell 2 time is possible.
        else:
            results, output = _refine_dwell2_time('min', n_dwells, min_dwell, max_dwell, limit, opt_fun, results)

    if output['max'][0] > output['max'][-1]:
        results['hotter_state'] = 1
        results['colder_state'] = 2
    else:
        results['hotter_state'] = 2
        results['colder_state'] = 1

    return results


def _handle_unconverged_hot(output, results):
    """ Record useful information for the case where all output remains above the limit.

    This is intended to be run solely by find_second_dwell(). This modifies the `results` dictionary inherited from the
    parent function to provide information about the case that came the closest to converging.

    :param output: Numpy array of maximum, mean, and minimum temperatures for each simulation generated, within the last
        `t_backoff` duration (e.g. the last two thirds of `duration`) for the final refinement step.
    :type output: np.ndarray
    :param results: Results dictionary initialized in parent function
    :type results: dict
    :returns: Dictionary of results information
    :rtype: dict
    """

    # You want the data for the case that is closest to the limit, in this case that is the data with the min value.
    ind = np.argmin(output['min'])
    results['unconverged_hot'] = True
    results['dwell_2_time'] = np.nan
    results['max_temp'] = output['max'][ind]
    results['min_temp'] = output['min'][ind]
    results['mean_temp'] = output['mean'][ind]
    results['converged'] = False

    return results


def _handle_unconverged_cold(output, results):
    """ Record useful information for the case where all output remains below the limit.

    This is intended to be run solely by find_second_dwell(). This modifies the `results` dictionary inherited from the
    parent function to provide information about the case that came the closest to converging.

    :param output: Numpy array of maximum, mean, and minimum temperatures for each simulation generated, within the last
        `t_backoff` duration (e.g. the last two thirds of `duration`) for the final refinement step.
    :type output: np.ndarray
    :param results: Results dictionary initialized in parent function
    :type results: dict
    :returns: Dictionary of results information
    :rtype: dict
    """

    # You want the data for the case that is closest to the limit, in this case that is the data with the max value.
    ind = np.argmax(output['max'])
    results['unconverged_cold'] = True
    results['dwell_2_time'] = np.nan
    results['max_temp'] = output['max'][ind]
    results['min_temp'] = output['min'][ind]
    results['mean_temp'] = output['mean'][ind]
    results['converged'] = False

    return results


def _refine_dwell2_time(limit_type, n_dwells, min_dwell, max_dwell, limit, opt_fun, results):
    """ Refine the required dwell time at pitch2 to balance a given fixed dwell time at pitch1.

    This is intended to be run solely by find_second_dwell() to refine the amount of dwell 2 time is necessary to
    balance the dwell 1 time. This modifies the `results` dictionary inherited from the parent function, but also
    returns the `output` ndarray containing data from the final refinement operation.

    :param limit_type: Type of limit, either a minimum or maximum temperature limit (needs to have 'min' or 'max' in
        string passed to this argument
    :type limit_type: str
    :param n_dwells: Number of second dwell possibilities to run (more dwells = finer resolution)
    :type n_dwells: int
    :param min_dwell: Minimum duration for second dwell, can be used if the user wants to narrow the dwell time search,
        defaults to 1.0e-6s
    :type min_dwell: float
    :param max_dwell: Maximum duration for second dwell, can be used if the user wants to narrow the dwell time search
    :type max_dwell: float
    :param limit: Limit in Celsius for current simulation
    :type limit: float
    :param opt_fun: Function that runs the schedule defined by dwell1_state and dwell2_state
    :type opt_fun: function
    :param results: Results dictionary initialized in parent function
    :type results: dict
    :returns:
        - results (:py:class:`dict`) - Dictionary of results information
        - times (:py:class:`np.ndarray`) - Numpy array of maximum, mean, and minimum temperatures for each simulation
            generated, within the last`t_backoff` duration (e.g. the last two thirds of `duration`) for the final
            refinement step.
    """

    # This is the configuration for working with a max temperature limit (as opposed to a min temperature limit).
    max_min = 'max'
    min_max = 'min'

    if 'min' in limit_type:
        max_min = 'min'
        min_max = 'max'

    # dwell2_range defines the possible dwell 2 guesses, first defined in log space
    dwell2_range = np.logspace(1.0e-6, 1, n_dwells, endpoint=True) / n_dwells
    dwell2_range = min_dwell + \
                   (max_dwell - min_dwell) * (dwell2_range - dwell2_range[0]) / (dwell2_range[-1] - dwell2_range[0])

    # Run the dwell1_state-dwell2_state schedule using the possible dwell 2 guesses
    output = np.array([opt_fun(t) for t in dwell2_range], dtype=[('duration2', float), ('max', float),
                                                                 ('mean', float), ('min', float)])

    # Ensure the results are sorted. Although dwell2_range will be sorted, the output may not when two or more dwell
    # times are close, where temperature oscillations from instabilities in the Xija model can cause the results to lose
    # this order.
    #
    # The column that is used to sort the results also depends on the limit type.
    output_sorted = np.sort(output, order=max_min)
    ind = np.searchsorted(output_sorted[max_min], limit)

    if ind == 0:
        # np.searchsorted finds the first suitable location by default, so if ind == 0, then the duration must
        # fall at the bounded value. This is not true if ind == -1 (the last value).
        results[max_min + '_temp'] = limit
        results['dwell_2_time'] = output['duration2'][ind]
        results[min_max + '_temp'] = output[min_max][ind]
        results['mean_temp'] = output['mean'][ind]
        results['converged'] = True

    else:
        t_bound = (output_sorted['duration2'][ind - 1], output_sorted['duration2'][ind])
        dwell2_range = np.linspace(np.min(t_bound), np.max(t_bound), n_dwells, endpoint=True)
        output = np.array([opt_fun(t) for t in dwell2_range],
                          dtype=[('duration2', float), ('max', float), ('mean', float),
                                 ('min', float)])

        # In rare conditions where all 'x' values are very close and 'wobble' a bit, it may not be sorted. If it
        # is not sorted, the quadratic method will result in an error. The linear method is more tolerant of this
        # condition. Additionally, the quadratic has a tendency to produce some really weird results even when the
        # data appears sensible.
        f_dwell_2_time = interpolate.interp1d(output[max_min], output['duration2'], kind='linear', assume_sorted=False)
        f_non_limit_temp = interpolate.interp1d(output[max_min], output[min_max], kind='linear', assume_sorted=False)
        f_mean_temp = interpolate.interp1d(output[max_min], output['mean'], kind='linear', assume_sorted=False)

        results[max_min + '_temp'] = limit
        results['dwell_2_time'] = f_dwell_2_time(limit).item()
        results['mean_temp'] = f_mean_temp(limit).item()
        results[min_max + '_temp'] = f_non_limit_temp(limit).item()

    results['converged'] = True

    return results, output


def run_state_pairs(msid, model_spec, init, limit, date, dwell_1_duration, state_pairs, limit_type='max',
                    min_dwell=None, max_dwell=None, n_dwells=10, print_progress=True, shared_data=None):
    """ Determine dwell balance times for a set of cases.

    :param msid: Primary MSID for model being run
    :type msid: str
    :param model_spec: Dictionary of model parameters or file location where parameters can be imported
    :type model_spec: dict, string
    :param init: Dictionary of Xija model initialization parameters
    :type init: dict
    :param limit: Temperature limit for primary MSID in model for this simulation
    :type limit: float
    :param date: Date for start of simulation, in seconds from '1997:365:23:58:56.816' (cxotime.CxoTime epoch) or any
        other format readable by cxotime.CxoTime
    :type date: float or int or str
    :param dwell_1_duration: Duration in seconds of dwell 1, also viewed as the known or defined dwell duration, for
        which one wants to find a complementary dwell duration (dwell duration 2)
    :type dwell_1_duration: float or int
    :param state_pairs: Iterable of dictionary pairs, where each pair of dictionaries contain dwell1 and dwell2 states,
        see state_pair section below for further details
    :type state_pairs: list or tuple
    :param limit_type: Type of limit, defaults to 'max' (a maximum temperature limit), other option is 'min'
    :type limit_type: str, optional
    :param min_dwell: Minimum duration for second dwell, can be used if the user wants to narrow the dwell time search,
        defaults to 1.0e-6s
    :type min_dwell: float
    :param max_dwell: Maximum duration for second dwell, can be used if the user wants to narrow the dwell time search
    :type max_dwell: float
    :param n_dwells: Number of second dwell, `t_dwell2`,  possibilities to run (more dwells = finer resolution)
    :type n_dwells: int, optional
    :param shared_data: Shared list of results, used when running multiple `run_state_pairs` threads in parallel via
        the multiprocessing package
    :type shared_data: multiprocessing.managers.ListProxy, optoinal
    :returns: Structured numpy array of results
    :rtype: np.ndarray

    State Pairs Data Structure:

    The state pairs data structure, `state_pairs`, are pairs of dictionaries specifying the two conditions used for a
    Timbre simulation. The keys in these dictionaries must match the Xija component names they refer to (e.g. 'pitch',
    'ccd_count', 'cossrbx_on', etc.).

    State information that does not change from dwell1 to dwell2 can be specified in the model initialization
    dictionary. `init`. State information that does change from dwell1 to dwell2 should be specified in the state pairs
    dictionary described above. Dictionary names for states should match those expected by Xija (e.g. fep_count, roll,
    sim_z).

    Example::

        model_init = {'aacccdpt': {'aacccdpt': -7., 'aca0': -7., 'eclipse': False}, }
        model_specs = load_model_specs()
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
        state_pair_dtype = {'pitch', float}

        results = run_state_pairs(msid, model_specs[msid], model_init[msid], limit, date, t_dwell1, state_pairs,
            state_pair_dtype)
    """

    non_state_names = {'aacccdpt': ['aca0', ],
                       'pftank2t': ['pf0tank2t', ],
                       '4rt700t': ['oba0', ],
                       'pline03t': ['pline03t0', ],
                       'pline04t': ['pline04t0', ],
                       'pm1thv2t': ['mups0', ],
                       'pm2thv1t': ['mups0', ],
                       '1deamzt': ['dea0', ],
                       '1dpamzt': ['dpa0', ],
                       'fptemp': ['fptemp', '1cbat', 'sim_px'],
                       '1pdeaat': ['pin1at', ]}

    base_dtype = [('msid', 'U20'),
                  ('date', 'U8'),
                  ('datesecs', float),
                  ('limit', float),
                  ('t_dwell1', float),
                  ('t_dwell2', float),
                  ('min_temp', float),
                  ('mean_temp', float),
                  ('max_temp', float),
                  ('min_pseudo', float),
                  ('mean_pseudo', float),
                  ('max_pseudo', float),
                  ('converged', bool),
                  ('unconverged_hot', bool),
                  ('unconverged_cold', bool),
                  ('hotter_state', np.int8),
                  ('colder_state', np.int8)]

    duration = 30 * 24 * 3600.
    t_backoff = 2 * duration / 3
    datestr = CxoTime(date).date[:8]
    datesecs = CxoTime(date).secs
    msid = msid.lower()

    results = []

    num = float(len(state_pairs))
    for n, pair in enumerate(state_pairs):

        if print_progress and (np.mod(n, 1000) == 0):
            print("Running simulations for state pair #: {} out of {}".format(n + 1, num))

        dwell1_state = pair[0]
        dwell2_state = pair[1]

        dwell_results = find_second_dwell(date, dwell1_state, dwell2_state, dwell_1_duration, msid, limit, model_spec,
                                          init, limit_type=limit_type, duration=duration, t_backoff=t_backoff,
                                          n_dwells=n_dwells, min_dwell=min_dwell, max_dwell=max_dwell, pseudo=None)

        row = [msid.encode('utf-8'),
               datestr.encode('utf-8'),
               datesecs,
               limit,
               dwell_1_duration,
               dwell_results['dwell_2_time'],
               dwell_results['min_temp'],
               dwell_results['mean_temp'],
               dwell_results['max_temp'],
               dwell_results['min_pseudo'],
               dwell_results['mean_pseudo'],
               dwell_results['max_pseudo'],
               dwell_results['converged'],
               dwell_results['unconverged_hot'],
               dwell_results['unconverged_cold'],
               dwell_results['hotter_state'],
               dwell_results['colder_state']]

        for key, value in init.items():
            if key not in non_state_names[msid] and key not in dwell1_state and key not in msid:
                dwell1_state[key] = value
                dwell2_state[key] = value

        state_dtype = []
        for key, value in dwell1_state.items():
            row.append(value)
            state_dtype.append((key.lower() + '1', type(value)))

        for key, value in dwell2_state.items():
            row.append(value)
            state_dtype.append((key.lower() + '2', type(value)))

        results.append(tuple(row))

    dtype = base_dtype + state_dtype
    results_array = np.array(results, dtype=dtype)

    if shared_data is not None:
        shared_data.append(results_array)
    else:
        return results_array
