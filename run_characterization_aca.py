import sys
from os.path import expanduser
import pickle
from multiprocessing import Process, Manager

home = expanduser("~")
sys.path.append(home + '/AXAFLIB/timbre/')
from timbre import *

model_specs = load_model_specs()
msid = 'aacccdpt'
limit = -9.5
datestamp = DateTime().caldate[:9]
init = {'aacccdpt': -10, 'aca0': -10, 'eclipse': False}

# r = re.compile('solarheat__.*__P_(\d+).*')
# pitch_vals = [np.int((r.findall(d['full_name'])[0])) for d in old_model.model.pars if r.findall(d['full_name'])]
# pitch_vals = list(set(pitch_vals))


def get_run_fun(date, duration1, pitch_vals, return_list):

    def ret_fun():
        state_pairs = [({'sequence1': 1, 'obsid1': 99999, 'duration1_fraction': 1.0, 'duration1': duration1,
                         'pitch': pn1}, {'sequence2': 2, 'obsid2': 22222, 'pitch': pn2})
                       for pn1 in pitch_vals
                       for pn2 in pitch_vals]

        # t1 = DateTime().secs
        results = run_state_pairs(msid, model_specs[msid], init, limit, date, state_pairs, max_dwell=200000,
                                  shared_data=return_list)
        # t2 = DateTime().secs

        # print('Set number {} took {} seconds, for {} state pairs'.format(filenum, t2 - t1, len(state_pairs)))
        # pickle.dump(results, open('1dpamzt_{}{}_duration1-{}.pkl'.format(date[:4], date[5:8], duration1), 'wb'))

        return results

    return ret_fun


if __name__ == "__main__":

    sets = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    pitch_vals = list(range(45, 181, 5))
    dates = ['2020:001:00:00:00',
             '2020:182:00:00:00',
             '2021:001:00:00:00',
             '2021:182:00:00:00',
             '2022:001:00:00:00',
             '2022:182:00:00:00']
    limits = [-9.5, -9.0, -8.5, -8.0, -7.5, -7.0]

    all_results = []

    print(DateTime().caldate)

    # ------------------------------------------------------------------------------------------------------------------

    for date in dates:
        for limit in limits:

            manager = Manager()
            return_list = manager.list()
            jobs = []

            for s in sets:
                jobs.append(Process(target=get_run_fun(date, s, pitch_vals, return_list), args=()))

            for j in jobs:
                j.start()

            for j in jobs:
                j.join()

            results = np.hstack(return_list)
            all_results.append(results)
            print(DateTime().caldate)

    # ------------------------------------------------------------------------------------------------------------------

    pickle.dump(all_results, open('aca_{}.pkl'.format(datestamp), 'wb'))

    print(DateTime().caldate)