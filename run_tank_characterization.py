import sys
from os.path import expanduser
from multiprocessing import Process, Manager
import h5py
import logging

logging.getLogger("xija").setLevel(logging.WARNING)

home = expanduser("~")
sys.path.append(home + '/AXAFLIB/timbre/')
from timbre import *


utf8_type_20 = h5py.string_dtype('utf-8', 20)
utf8_type_8 = h5py.string_dtype('utf-8', 8)

results_dtype = [('msid', utf8_type_20),
                 ('date', utf8_type_8),
                 ('datesecs', np.float64),
                 ('limit', np.float64),
                 ('t_dwell1', np.float64),
                 ('t_dwell2', np.float64),
                 ('min_temp', np.float64),
                 ('mean_temp', np.float64),
                 ('max_temp', np.float64),
                 ('min_pseudo', np.float64),
                 ('mean_pseudo', np.float64),
                 ('max_pseudo', np.float64),
                 ('converged', np.bool),
                 ('unconverged_hot', np.bool),
                 ('unconverged_cold', np.bool),
                 ('hotter_state', np.int8),
                 ('colder_state', np.int8)]


def get_local_model(filename):
    """ Load parameters for a single Xija model.
    """

    with open(filename) as fid:  # 'aca/aca_spec.json', 'rb') as fid:
        f = fid.read()

    md5_hash = md5(f.encode('utf-8')).hexdigest()

    return json.loads(f), md5_hash


def save_results_to_hdf5(filename, results):
    ind = np.argsort(results, order=['datesecs', 'pitch1', 'pitch2', 't_dwell1', 't_dwell2'])
    results = results[ind]
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('results', (np.shape(results)), dtype=results_dtype)
        dset[...] = results
        f.flush()


model_spec, model_hash = get_local_model('/Users/mdahmer/WIP/xija_model_updates/pftank2t/chandra_models/chandra_models/xija/pftank2t/pftank2t_spec.json')
msid = 'pftank2t'
datestamp = DateTime().caldate[:9]
init = {'pftank2t': -8, 'pf0tank2t': -8, 'eclipse': False}
state_pair_dtype = {'pitch': np.float64, 'roll': np.float64, 'cossrbx_on': np.bool}

for key, value in state_pair_dtype.items():
    results_dtype.append((key + '1', value))
for key, value in state_pair_dtype.items():
    results_dtype.append((key + '2', value))

if __name__ == "__main__":

    sets = [10000, 20000, 30000, 40000]
    pitch_vals = list(range(45, 180, 5))
    roll_vals = [-10, 0, 10]

    cases = {'2020:274:00:00:00': timbre.f_to_c([105.0, 110.0]),
             '2021:001:00:00:00': timbre.f_to_c([105.0, 110.0, 115.0]),
             '2021:091:00:00:00': timbre.f_to_c([105.0, 110.0, 115.0]),
             '2021:182:00:00:00': timbre.f_to_c([110.0, 115.0, 120.0]),
             '2021:274:00:00:00': timbre.f_to_c([110.0, 115.0, 120.0]),
             '2022:001:00:00:00': timbre.f_to_c([110.0, 115.0, 120.0])}

    state_pairs = [({'pitch': pn1, 'roll': rn1, 'cossrbx_on': ssr}, {'pitch': pn2, 'roll': rn2, 'cossrbx_on': ssr})
                   for pn1 in pitch_vals
                   for pn2 in pitch_vals
                   for rn1 in roll_vals
                   for rn2 in roll_vals
                   for ssr in [True, False]]

    print(f'Starting Timbre simulations on {DateTime().caldate}')

    # ------------------------------------------------------------------------------------------------------------------

    run_sets = [[10000, 20000, 30000, 40000], [50000, 60000, 70000, 80000], [90000, 100000] ]

    k = 0
    for date, limits in cases.items():
        datestr = DateTime(date).date[:4] + DateTime(date).date[5:8]

        for sets in run_sets:

            for limit in limits:
                k = k + 1

                manager = Manager()
                return_list = manager.list()
                jobs = []

                for s in sets:
                    args = (msid, model_spec, init, limit, date, s, state_pairs, state_pair_dtype)
                    kwargs = {'max_dwell': 200000, 'shared_data': return_list}
                    jobs.append(Process(target=run_state_pairs, args=args, kwargs=kwargs))

                for j in jobs:
                    j.start()

                for j in jobs:
                    j.join()

                results = np.hstack(return_list)
                filename = f'pftank2t_{datestamp}_5_deg_resolution_{datestr}_save_{k}.h5'
                save_results_to_hdf5(filename, results)

                print('Completed {}, limit={} on {}'.format(date, limit, DateTime().caldate))


    print(f'Completed all Timbre simulations on {DateTime().caldate}')