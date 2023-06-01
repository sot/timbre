# Licensed under a 3-clause BSD style license - see LICENSE.rst

from cxotime import CxoTime
import pandas as pd
import numpy as np
from os.path import expanduser
from multiprocessing import Pool, Manager
from copy import copy

from timbre import Composite, DEFAULT_ANCHORS, get_local_model, Balance2CEAHVPT

home = expanduser("~")

INPUT_COLUMNS = ['fptemp_11_limit', '1dpamzt_limit', '1deamzt_limit', '1pdeaat_limit', 'aacccdpt_limit',
                 '4rt700t_limit', 'pftank2t_limit', 'pm1thv2t_limit', 'pm2thv1t_limit', 'pline03t_limit',
                 'pline04t_limit', 'date', 'datesecs', 'dwell_type', 'roll', 'chips', 'pitch']


def add_inputs(p, limits, date, dwell_type, roll, chips):
    """
    Add input data to Timbre coposite results.

    :param p: Output data
    :type p: pd.DataFrame
    :param limits: MSID limits used to produce included results
    :type limits: dict, pd.Series
    :param date: Date simulated
    :type date: str
    :param dwell_type: Output data category (i.e. limited or offset)
    :type dwell_type: str
    :param roll: Body roll used to produce included results
    :type roll: int, float
    :param chips: Number of ACIS chips used to produce included results
    :type chips: int

    :returns: Pandas Dataframe of results data with associated simulation input data
    :rtype: pd.DataFrame
    """
    for msid, limit in limits.items():
        if 'limit' not in msid:
            p[msid + '_limit'] = float(limit)
        else:
            p[msid] = float(limit)
    p['date'] = date
    p['datesecs'] = CxoTime(date).secs
    p['dwell_type'] = dwell_type
    p['roll'] = roll
    p['chips'] = chips

    p.reset_index(inplace=True)
    p = p.rename(columns={'index': 'pitch'})
    return p


def run_instance(date, chips, limits, roll, max_dwell, pitch_step, model_specs=None, anchors=None, maneuvers=False):
    """
    Helper function to run a Composite analysis and add the results to the Multiprocessing shared list.

    :param date: Date simulated
    :type date: str
    :param chips: Number of ACIS chips used to produce included results
    :type chips: int
    :param limits: MSID limits used to produce included results
    :type limits: dict, pd.Series
    :param roll: Body roll used to produce included results
    :type roll: int, float
    :param max_dwell: Maximum dwell time for initial offset cooling
    :type max_dwell: int, float
    :param pitch_step: Pitch resolution of output results, see Composite API for limitations
    :type pitch_step: int
    :param model_specs: Dictionary of model parameters, if None then the `timbre.load_model_specs` method will be
        used to load all model specs.
    :type model_specs: dict or None, optional
    :param anchors: Dictionary of anchor values, if None is passed then default anchors are used
    :type anchors: dict or None, optional
    :param maneuvers: Boolean indicating whether to consider maneuver time
    :type maneuvers: bool, optional
    :returns: None
    """

    try:
        timbre_object = Composite(date, chips, roll, limits, max_dwell=max_dwell, pitch_step=pitch_step,
                                  model_specs=model_specs, anchors=anchors, maneuvers=maneuvers)
        limited_data = timbre_object.limited_results
        offset_data = timbre_object.offset_results

        t_limit = add_inputs(limited_data, limits, date, 'limit', roll, chips)
        t_offset = add_inputs(offset_data, limits, date, 'offset', roll, chips)
        data = pd.concat([t_limit, t_offset])
        return data

    except IndexError:
        print(f'Date: {date}, Chips: {chips}, Roll: {roll}, \nLimits: {limits}')


def run_all_permutations(input_sets, filename, max_dwell=200000, pitch_step=5, model_specs=None, anchors=None,
                         maneuvers=False):
    """ Run all permutations of dates x chip_nums x limit_sets

    :param input_sets: Dates to be simulated
    :type input_sets: pd.DataFrame
    :param filename: Results output file name
    :type filename: str
    :param max_dwell: Maximum dwell time for initial offset cooling
    :type max_dwell: int, float, optional
    :param pitch_step: Pitch resolution of output results, see Composite API for limitations
    :type pitch_step: int, optional
    :param model_specs: Dictionary of model parameters, if None then the `timbre.load_model_specs` method will be
        used to load all model specs.
    :type model_specs: dict or None, optional
    :param anchors: Dictionary of anchor values, if None is passed then default anchors are used
    :type anchors: dict or None, optional
    :param maneuvers: Boolean indicating whether to consider maneuver time
    :type maneuvers: bool, optional
    :returns: None

    """

    num_sets = len(input_sets)
    for n, input_set in input_sets.iterrows():
        limits = input_set.iloc[:-3]
        roll = input_set.iloc[-3]
        chips = input_set.iloc[-2]
        date = input_set.iloc[-1]
        data = run_instance(date, chips, limits, roll, max_dwell, pitch_step, model_specs=model_specs, anchors=anchors,
                            maneuvers=maneuvers)
        data.to_csv(filename, mode='a', index=False, header=False)

        text1 = f'{CxoTime().date}: Finished Limit Set {n + 1} out of '
        text2 = f'{num_sets}:\n{input_set}\nFor {date} and Chip Numbers {chips}\n\n'
        print(text1 + text2)


def _worker(arg, q):
    """ Run a Timbre case and post the results to the queue

    :param arg: Input arguments to run Timbre case
    :type arg: iterable
    :param q: Queue object
    :type q: mulitprocessing.Manager.Queue
    """

    (input_set, max_dwell, pitch_step, model_specs, n, num_sets, anchors, maneuvers) = arg
    limits = input_set.iloc[:-3]
    roll = input_set.iloc[-3]
    chips = input_set.iloc[-2]
    date = input_set.iloc[-1]
    res = run_instance(date, chips, limits, roll, max_dwell, pitch_step, model_specs=model_specs, anchors=anchors,
                       maneuvers=maneuvers)

    text1 = f'{CxoTime().date}: Finished Limit Set {n + 1} out of '
    text2 = f'{num_sets}:\n{input_set}\nFor {date} and Chip Numbers {chips}\n\n'
    print(text1 + text2)

    if n == 0:
        header = True
    else:
        header = False

    res = res.to_csv(index=False, header=header)
    q.put(res)

    return res


def _listener(filename, q):
    ''' Listen for messages on the q, write results to file.

    :param filename: File to write results
    :type filename: str
    :param q: Queue object
    :type q: mulitprocessing.Manager.Queue

    '''

    with open(filename, 'w') as fid:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            fid.write(m)
            fid.flush()


def process_queue(input_sets, max_dwell, pitch_step, filename, cpu_count=2, num_cases=None, model_specs=None,
                  anchors=None, maneuvers=False):
    """ Run a set of Timbre cases using a pool of CPUs

    :param input_sets: Dates to be simulated
    :type input_sets: pd.DataFrame
    :param max_dwell: Maximum dwell time for initial offset cooling
    :type max_dwell: int, float
    :param pitch_step: Pitch resolution of output results, see Composite API for limitations
    :type pitch_step: int
    :param filename: Results output file name
    :type filename: str
    :param cpu_count: Number of CPUs to use (min of 2, since 1 CPU will be used by the listener)
    :type cpu_count: int, optional
    :param num_cases: Number of cases in `input_sets` to process, used to run the first N cases, default is
        len(input_sets)
    :type num_cases: int, optional
    :param model_specs: Dictionary of model parameters, if None then the `timbre.load_model_specs` method will be
        used to load all model specs.
    :param anchors: Dictionary of anchor values, if None is passed then default anchors are used
    :type anchors: dict or None, optional
    :type model_specs: dict or None, optional
    :param maneuvers: Boolean indicating whether to consider maneuver time
    :type maneuvers: bool, optional

    :returns: None

    # Credit to the following URL for the multiprocessing example:
    # https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file

    """
    if num_cases is None:
        num_cases = len(input_sets)

    manager = Manager()
    q = manager.Queue()
    pool = Pool(cpu_count)

    # put listener to work first
    watcher = pool.apply_async(_listener, (filename, q,))

    # start workers
    jobs = []
    for n, input_set in input_sets.iloc[:num_cases].iterrows():
        arg = (input_set, max_dwell, pitch_step, model_specs, n, num_cases, anchors, maneuvers)
        job = pool.apply_async(_worker, (arg, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


def find_inputs_from_results(all_results, pitch=90):
    """ Determine the list of non-pitch input conditions that were used to generate a given set of Timbre results.

    :param all_results: Timbre results
    :type all_results: np.array
    :param pitch: pitch to pull input results for
    :type pitch: int, float, optional

    :returns: pd.DateFrame

    """

    pitch = int(pitch)
    single_pitch_limit_results = all_results.loc[(all_results['pitch'] == pitch).values &
                                                 (all_results['dwell_type'] == 'limit').values]
    duplicate_values = all_results.duplicated().values
    if any(duplicate_values):
        single_pitch_limit_results = single_pitch_limit_results.loc[~duplicate_values]
    return single_pitch_limit_results[INPUT_COLUMNS].reset_index(drop=True)


def generate_hrc_estimates(all_results, cea_model_spec, filename, limit=10, limited_matches_offset=False,
                           anchor_limited_pitch=None):

    inputs = find_inputs_from_results(all_results, pitch=90)
    indexed_results = all_results.set_index(INPUT_COLUMNS)

    # Initial parameters to create Balance2CEAHVPT object
    msid = '2ceahvpt'
    anchors = DEFAULT_ANCHORS
    anchor_offset_pitch = anchors[msid]['anchor_offset_pitch']
    if anchor_limited_pitch is None:
        anchor_limited_pitch = anchors[msid]['anchor_limited_pitch']
    constant_conditions = {'roll': 0, 'dh_heater': False}
    pitch_range = np.arange(45, 181, 1)
    placeholder_date = '2023:001'

    # Create Balance object
    model = Balance2CEAHVPT(placeholder_date, cea_model_spec, limit, constant_conditions, custom_offset_conditions=None,
                            custom_limited_conditions=None)

    # Initialize case_results dataframe
    case_results = pd.DataFrame(columns=['2ceahvpt', 'instrument'] + INPUT_COLUMNS)

    num_cases = len(inputs)
    for case_num in range(num_cases):

        print(f'{CxoTime().date}: Running case number {case_num} out of {num_cases}')

        # Update case parameters
        chips = inputs.iloc[case_num]['chips']
        model.offset_conditions.update({'ccd_count': chips, 'fep_count': chips})
        composite_case = indexed_results.loc[tuple(inputs.iloc[case_num].values[:-1])].min(axis=1)
        anchor_offset_time = composite_case.loc[anchor_offset_pitch]

        if limited_matches_offset is True:
            model.limited_conditions = copy(model.offset_conditions)

        # Run case
        model.find_anchor_condition(anchor_offset_pitch, anchor_limited_pitch, anchor_offset_time, limit)
        model.results = model.generate_balanced_pitch_dwells(model.datesecs,
                                                             anchor_limited_pitch,
                                                             model.anchor_limited_time,
                                                             anchor_offset_pitch,
                                                             anchor_offset_time,
                                                             limit,
                                                             pitch_range)

        # Create cols list to index into inputs dataframe, pitch is dealt with separately
        cols = copy(INPUT_COLUMNS)
        cols.remove('pitch')

        # Initialize temporary results dataframes
        case_limited_results = pd.DataFrame(index=pitch_range, columns=['2ceahvpt', ] + cols)
        case_offset_results = pd.DataFrame(index=pitch_range, columns=['2ceahvpt', ] + cols)

        # Copy inputs into results dataframes
        case_limited_results[cols] = inputs.iloc[case_num].loc[cols]
        case_offset_results[cols] = inputs.iloc[case_num].loc[cols]

        if model.results is not None:

            # Find indices to offset and limited dwell results
            limited_ind = model.results['pitch1'] == anchor_offset_pitch
            offset_ind = model.results['pitch1'] == anchor_limited_pitch

            # Copy results intp results dataframes
            case_limited_results['2ceahvpt'] = model.results['t_dwell2'][limited_ind]
            case_offset_results['2ceahvpt'] = model.results['t_dwell2'][offset_ind]

        else:
            case_limited_results['2ceahvpt'] = np.nan
            case_offset_results['2ceahvpt'] = np.nan

        # Add dwell type
        case_limited_results['dwell_type'] = 'limit'
        case_offset_results['dwell_type'] = 'offset'

        # Add instrument
        if limited_matches_offset is False:
            case_limited_results['instrument'] = 'hrc'
        else:
            case_limited_results['instrument'] = 'acis'
        case_offset_results['instrument'] = 'acis'

        # Add pitch from results index (same for both actually)
        case_limited_results['pitch'] = case_limited_results.index
        case_offset_results['pitch'] = case_offset_results.index

        # Add case results to full results dataframe
        case_results = pd.concat([case_results, case_limited_results, case_offset_results], ignore_index=True)

    case_results.to_csv(filename)






