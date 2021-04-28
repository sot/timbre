from multiprocessing import Process, Manager
import numpy as np
from cxotime import CxoTime
import itertools
import pandas as pd

from timbre import f_to_c, Composite


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
        p[msid + '_limit'] = np.float(limit)
    p['date'] = date
    p['datesecs'] = CxoTime(date).secs
    p['dwell_type'] = dwell_type
    p['roll'] = roll
    p['chips'] = chips

    p.reset_index(inplace=True)
    p = p.rename(columns={'index': 'pitch'})
    return p


def run_composite_multiprocessing_instance(date, chips, limits, roll, max_dwell, pitch_step, return_list):
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
        timbre_object = Composite(date, chips, roll, limits, max_dwell=max_dwell, pitch_step=pitch_step)
        limited_data = timbre_object.limited_results
        offset_data = timbre_object.offset_results

        t_limit = add_inputs(limited_data, limits, date, 'limit', roll, chips)
        t_offset = add_inputs(offset_data, limits, date, 'offset', roll, chips)
        data = pd.concat([t_limit, t_offset])
        return_list.append(data)

    except IndexError:
        print(f'Date: {date}, Chips: {chips}, Roll: {roll}, \nLimits: {limits}')


def run_all_permutations(dates, chip_nums, limit_sets, roll, filename, max_dwell=200000, pitch_step=5):
    """
    Run all permutations of dates x chip_nums x limit_sets

    :param dates: Dates to be simulated
    :type dates: list
    :param chip_nums: List of ACIS chip counts to be simulated
    :type chip_nums: list
    :param limit_sets: list of separate limit sets to be simulated
    :type limit_sets: list, tuple
    :param roll: Body roll used to produce included results
    :type roll: int, float
    :param filename: Results output file name
    :type filename: str
    :param max_dwell: Maximum dwell time for initial offset cooling
    :type max_dwell: int, float, optional
    :param pitch_step: Pitch resolution of output results, see Composite API for limitations
    :type pitch_step: int, optional
    :returns: None

    """

    manager = Manager()
    return_list = manager.list()
    for ind in limit_sets.index:
        limits = dict(limit_sets.loc[ind])

        jobs = []
        for chips in chip_nums:
            for date in dates:
                args = (date, chips, limits, roll, max_dwell, pitch_step, return_list)
                jobs.append(Process(target=run_composite_multiprocessing_instance, args=args))

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

    return_list_concat = pd.concat(return_list)
    return_list_concat = return_list_concat.reset_index(drop=True)
    return_list_concat.to_csv(filename)


if __name__ == '__main__':

    msids = ['fptemp_11', '1dpamzt', '1deamzt', '1pdeaat', 'aacccdpt', '4rt700t',  'pftank2t', 'pm1thv2t', 'pm2thv1t',
             'pline03t', 'pline04t']

    dates = ['2021:182:00:00:00',
             '2021:273:00:00:00']
    roll = 0

    acisfp_limits = [-109.0, -111.0, -112.0]
    dpa_limits = [37.5, ]
    dea_limits = [37.5, ]
    psmc_limits = [52.5, ]
    aca_limits = [-6.5, -7.5, -5.8, -6.8]
    oba_limits = [f_to_c(103), ]
    tank_limits = [f_to_c(115), ]
    mups1b_limits = [f_to_c(210), ]
    mups2a_limits = [f_to_c(210), ]
    pline03t_limits = [f_to_c(50), ]
    pline04t_limits = [f_to_c(50), ]

    chip_nums = [1, 2, 3, 4, 5, 6]

    combinations = list(itertools.product(acisfp_limits, dpa_limits, dea_limits, psmc_limits,
                                          aca_limits, oba_limits, tank_limits, mups1b_limits,
                                          mups2a_limits, pline03t_limits, pline04t_limits))
    limit_sets = pd.DataFrame(combinations, columns=msids)

    filename = 'timbre_composite_datasets_2021182_2021273.csv'

    run_all_permutations(dates, chip_nums, limit_sets, roll, filename, max_dwell=200000, pitch_step=5)
