# Licensed under a 3-clause BSD style license - see LICENSE.rst

from cxotime import CxoTime
import pandas as pd
from os.path import expanduser

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
        p[msid + '_limit'] = float(limit)
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
    :param return_list: Multiprocessing shared list
    :type return_list: multiprocessing.Manager.list
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
    """
    Run all permutations of dates x chip_nums x limit_sets

    :param input_sets: Dates to be simulated
    :type input_sets: pd.DataFrame
    :param filename: Results output file name
    :type filename: str
    :param max_dwell: Maximum dwell time for initial offset cooling
    :type max_dwell: int, float, optional
    :param pitch_step: Pitch resolution of output results, see Composite API for limitations
    :type pitch_step: int, optional
    :returns: None

    """

    results = pd.DataFrame()
    num_sets = len(input_sets)
    for n, input_set in input_sets.iterrows():
        limits = input_set.iloc[:-3]
        roll = input_set.iloc[-3]
        chips = input_set.iloc[-2]
        date = input_set.iloc[-1]
        data = run_instance(date, chips, limits, roll, max_dwell, pitch_step, model_specs=model_specs)
        results = pd.concat((results, data))
        results.to_csv(filename, mode='a', index=False, header=False)

        text1 = f'{CxoTime().date}: Finished Limit Set {n + 1} out of '
        text2 = f'{num_sets}:\n{input_set}\nFor {date} and Chip Numbers {chips}\n\n'
        print(text1 + text2)
