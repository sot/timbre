from multiprocessing import Process, Manager
import numpy as np
from cxotime import CxoTime
import itertools
import pandas as pd

from timbre import f_to_c, generate_balanced_dwells


def add_inputs(p, limits, date, dwell_type, roll, chips):
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


def generate_balanced_dwells_multiprocessing(date, chips, roll, limits, return_list):

    try:
        limited_data, offset_data = generate_balanced_dwells(date, chips, roll, limits)
        t_limit = add_inputs(limited_data, limits, date, 'limit', roll, chips)
        t_offset = add_inputs(offset_data, limits, date, 'offset', roll, chips)
        data = pd.concat([t_limit, t_offset])
        return_list.append(data)

    except IndexError:
        print(f'Date: {date}, Chips: {chips}, Roll: {roll}, \nLimits: {limits}')


# dates = ['2021:001:00:00:00',
#          '2021:091:00:00:00',
#          '2021:182:00:00:00',
#          '2021:273:00:00:00',
#          '2022:001:00:00:00',
#          '2022:091:00:00:00',
#          '2022:182:00:00:00',
#          '2022:273:00:00:00']

dates = ['2021:091:00:00:00',
         '2021:182:00:00:00',
         '2021:273:00:00:00']


acisfp_limits = [-109.0, -111.0, -112.0]
dpa_limits = [37.5, ]
dea_limits = [37.5, ]
psmc_limits = [52.5, ]
aca_limits = [-6.5, -7.5, -5.8, -6.8]
oba_limits = [f_to_c(100), f_to_c(103)]
tank_limits = [f_to_c(110), f_to_c(115)]
mups1b_limits = [f_to_c(210), ]
mups2a_limits = [f_to_c(210), ]
pline03t_limits = [f_to_c(50), ]
pline04t_limits = [f_to_c(50), ]

combinations = list(itertools.product(acisfp_limits, dpa_limits, dea_limits, psmc_limits,
                                      aca_limits, oba_limits, tank_limits, mups1b_limits,
                                      mups2a_limits, pline03t_limits, pline04t_limits))
msids = ['fptemp_11', '1dpamzt', '1deamzt', '1pdeaat', 'aacccdpt', '4rt700t',
          'pftank2t', 'pm1thv2t', 'pm2thv1t', 'pline03t', 'pline04t']
limit_sets = pd.DataFrame(combinations, columns=msids)

roll = 0

manager = Manager()
return_list = manager.list()

for ind in limit_sets.index:
    limits = dict(limit_sets.loc[ind])

    jobs = []
    for chips in [1, 2, 3, 4, 5, 6]:
        for date in dates:
            args = (date, chips, roll, limits, return_list)
            jobs.append(Process(target=generate_balanced_dwells_multiprocessing, args=args))

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

return_list_concat = pd.concat(return_list)
return_list_concat = return_list_concat.reset_index(drop=True)
return_list_concat.to_csv('dwell_limits_offsets_2021091_2021182_2021273.csv')

