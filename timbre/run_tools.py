# Licensed under a 3-clause BSD style license - see LICENSE.rst

from cxotime import CxoTime
import pandas as pd
from os.path import expanduser
from multiprocessing import Pool, Manager

from timbre import Composite

home = expanduser("~")


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


def run_instance(date, chips, limits, roll, max_dwell, pitch_step, model_specs=None):
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
    :returns: None
    """

    try:
        timbre_object = Composite(date, chips, roll, limits, max_dwell=max_dwell, pitch_step=pitch_step,
                                  model_specs=model_specs)
        limited_data = timbre_object.limited_results
        offset_data = timbre_object.offset_results

        t_limit = add_inputs(limited_data, limits, date, 'limit', roll, chips)
        t_offset = add_inputs(offset_data, limits, date, 'offset', roll, chips)
        data = pd.concat([t_limit, t_offset])
        return data

    except IndexError:
        print(f'Date: {date}, Chips: {chips}, Roll: {roll}, \nLimits: {limits}')


def run_all_permutations(input_sets, filename, max_dwell=200000, pitch_step=5, model_specs=None):
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
    :returns: None

    """

    num_sets = len(input_sets)
    for n, input_set in input_sets.iterrows():
        limits = input_set.iloc[:-3]
        roll = input_set.iloc[-3]
        chips = input_set.iloc[-2]
        date = input_set.iloc[-1]
        data = run_instance(date, chips, limits, roll, max_dwell, pitch_step, model_specs=model_specs)
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

    (input_set, max_dwell, pitch_step, model_specs, n, num_sets) = arg
    limits = input_set.iloc[:-3]
    roll = input_set.iloc[-3]
    chips = input_set.iloc[-2]
    date = input_set.iloc[-1]
    res = run_instance(date, chips, limits, roll, max_dwell, pitch_step, model_specs=model_specs)

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


def process_queue(input_sets, max_dwell, pitch_step, filename, cpu_count=2, num_cases=None, model_specs=None):
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
    :type model_specs: dict or None, optional

    :returns: None

    # Credit to the following URL for the multiprocessing example:
    # https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file

    """
    if num_cases is None:
        num_cases = len(input_sets)

    manager = Manager()
    q = manager.Queue()
    pool = Pool(cpu_count)

    #put listener to work first
    watcher = pool.apply_async(_listener, (filename, q,))

    # start workers
    jobs = []
    for n, input_set in input_sets.iloc[:num_cases].iterrows():
        arg = (input_set, max_dwell, pitch_step, model_specs, n, num_cases)
        job = pool.apply_async(_worker, (arg, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


