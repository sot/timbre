import sys
from os.path import expanduser
import pickle
from multiprocessing import Process, Manager

home = expanduser("~")
sys.path.append(home + '/AXAFLIB/timbre/')
from timbre import *

model_specs = load_model_specs()
msid = 'fptemp'


r = re.compile('solarheat__.*__P_(\d+).*')
pitch_vals = [np.int((r.findall(d['full_name'])[0])) for d in old_model.model.pars if r.findall(d['full_name'])]
pitch_vals = list(set(pitch_vals))

datestamp = DateTime().caldate[:9]
init = {'fptemp': -112.,
        '1cbat': -55.,
        'sim_px': -110.,
        'eclipse': False,
        'dpa_power': 0.0,
        'sim_z': 100000,
        'orbitephem0_x': 50000e3,
        'orbitephem0_y': 50000e3,
        'orbitephem0_z': 50000e3,
        'aoattqt1': 0.0,
        'aoattqt2': 0.0,
        'aoattqt3': 0.0,
        'aoattqt4': 1.0,
        'dh_heater': False}

date = '2020:001:00:00:00'


def get_run_fun(duration1, pitch_vals, ccds, return_list):

    def ret_fun():
        state_pairs = [({'sequence1': 1, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': duration1,
                         'pitch': pn1, 'roll': 0.0, 'ccd_count': ccd, 'fep_count': ccd, 'vid_board': 1, 'clocking': 1},
                        {'sequence2': 2, 'obsid2': 22222, 'pitch': pn2, 'roll': 0.0, 'ccd_count': ccd, 'fep_count': ccd,
                         'vid_board': 1, 'clocking': 1})
                       for pn1 in pitch_vals
                       for pn2 in pitch_vals
                       for ccd in ccds]

        # t1 = DateTime().secs
        results = run_state_pairs(msid, model_specs[msid], init, limit, date, state_pairs, max_dwell=200000,
                                  shared_data=return_list)
        # t2 = DateTime().secs

        # print('Set number {} took {} seconds, for {} state pairs'.format(filenum, t2 - t1, len(state_pairs)))
        # pickle.dump(results, open('1dpamzt_{}{}_duration1-{}.pkl'.format(date[:4], date[5:8], duration1), 'wb'))

        return results

    return ret_fun


if __name__ == "__main__":

    print(DateTime().caldate)
    # ------------------------------------------------------------------------------------------------------------------
    # sets = [10000, 20000, 30000, 40000]
    # # pitch_vals = [45, 60, 75] + list(range(90, 181, 5))
    # ccds = [2, 4, 6]
    # limit = -111.
    #
    # manager = Manager()
    # return_list = manager.list()
    # jobs = []
    #
    # for s in sets:
    #     jobs.append(Process(target=get_run_fun(s, pitch_vals, ccds, return_list), args=()))
    #
    # for j in jobs:
    #     j.start()
    #
    # for j in jobs:
    #     j.join()
    #
    # all_results = np.hstack(return_list)
    # pickle.dump(all_results, open('fptemp_{}{}_limit_{}_set_{}.pkl'.format(date[:4], date[5:8], limit, 1), 'wb'))

    # ------------------------------------------------------------------------------------------------------------------
    sets = [10000, 20000, 30000, 40000]
    # pitch_vals = [45, 60, 75] + list(range(90, 181, 5))
    ccds = [1, 3, 5]
    limit = -111.

    manager = Manager()
    return_list = manager.list()
    jobs = []

    for s in sets:
        jobs.append(Process(target=get_run_fun(s, pitch_vals, ccds, return_list), args=()))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    all_results = np.hstack(return_list)
    pickle.dump(all_results, open('fptemp_{}{}_limit_{}_set_{}.pkl'.format(date[:4], date[5:8], limit, 2), 'wb'))

    # ------------------------------------------------------------------------------------------------------------------
    # sets = [10000, 20000, 30000, 40000]
    # # pitch_vals = [45, 60, 75] + list(range(90, 181, 5))
    # ccds = [2, 4, 6]
    # limit = -112.
    #
    # manager = Manager()
    # return_list = manager.list()
    # jobs = []
    #
    # for s in sets:
    #     jobs.append(Process(target=get_run_fun(s, pitch_vals, ccds, return_list), args=()))
    #
    # for j in jobs:
    #     j.start()
    #
    # for j in jobs:
    #     j.join()
    #
    # all_results = np.hstack(return_list)
    # pickle.dump(all_results, open('fptemp_{}{}_limit_{}_set_{}.pkl'.format(date[:4], date[5:8], limit, 3), 'wb'))

    # ------------------------------------------------------------------------------------------------------------------
    sets = [10000, 20000, 30000, 40000]
    # pitch_vals = [45, 60, 75] + list(range(90, 181, 5))
    ccds = [1, 3, 5]
    limit = -112.

    manager = Manager()
    return_list = manager.list()
    jobs = []

    for s in sets:
        jobs.append(Process(target=get_run_fun(s, pitch_vals, ccds, return_list), args=()))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

    all_results = np.hstack(return_list)
    pickle.dump(all_results, open('fptemp_{}{}_limit_{}_set_{}.pkl'.format(date[:4], date[5:8], limit, 4), 'wb'))

    # ------------------------------------------------------------------------------------------------------------------

    print(DateTime().caldate)