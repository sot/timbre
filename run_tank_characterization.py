import sys
from os.path import expanduser
from multiprocessing import Process, Manager
import logging
import h5py

logging.getLogger("xija").setLevel(logging.WARNING)

home = expanduser("~")
sys.path.append(home + '/AXAFLIB/timbre/')
from timbre import *


def save_results_to_hdf5(filename, results_array):
    """ Save Timbre results to an HDF5 file.

    Due to some compatibility issues with some string formats in HDF5 files, this may be deprecated in the future.

    :param filename: File name for HDF5 output file
    :param results_array: Numpy array of results

    """

    ind = np.argsort(results_array, order=['datesecs', 'pitch1', 'pitch2', 't_dwell1', 't_dwell2'])
    results_array = results_array[ind]
    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('results', (np.shape(results_array)), dtype=results_dtype)
        dset[...] = results_array
        f.flush()


def run_cases(msid_name, model_specification, model_md5, initial_params, dwell1_sets, binary_schedule_state_pairs,
              state_pair_numpy_dtype, date_and_limit_cases):
    """

    :param msid_name: Mnemonic name
    :param model_specification: Dictionary containing model specification
    :param model_md5: Model MD5 hash
    :param initial_params: Dictionary of initial parameters for the current model
    :param dwell1_sets: Two dimensional iterable of initial dwell times
    :param binary_schedule_state_pairs: Dictionary of first and second states, excluding dates and limits
    :param state_pair_numpy_dtype: List of name + Numpy data type pairs for each simulation, used to format Timbre
            results
    :param date_and_limit_cases: Dictionary of dates and limits, where the keys are dates, and the values for each key
            are a list of limits to run for the date

    """

    k = 0
    for full_date_str, limits in date_and_limit_cases.items():
        date_str = DateTime(full_date_str).date[:4] + DateTime(full_date_str).date[5:8]

        for sets in dwell1_sets:

            for limit_celsius in limits:
                k = k + 1

                manager = Manager()
                return_list = manager.list()

                jobs = []

                for dwell1_secs in sets:
                    args = (msid_name, model_specification, initial_params, limit_celsius, date_str, dwell1_secs,
                            binary_schedule_state_pairs, state_pair_numpy_dtype)
                    kwargs = {'max_dwell': 200000, 'shared_data': return_list}
                    jobs.append(Process(target=run_state_pairs, args=args, kwargs=kwargs))

                for j in jobs:
                    j.start()

                for j in jobs:
                    j.join()

                results_array = np.hstack(return_list)
                filename = f'{msid_name}_{model_md5}_{datestamp}_{date_str}_save_{k}.h5'
                save_results_to_hdf5(filename, results_array)

                print(f'Completed {date_str}, limit={limit_celsius}, dwell1 times={sets} on {DateTime().caldate}')


if __name__ == "__main__":

    model_spec, model_hash = get_local_model('/Users/mdahmer/AXAFLIB/chandra_models/chandra_models/xija/pftank2t/pftank2t_spec.json')
    msid = 'pftank2t'
    datestamp = DateTime().caldate[:9]
    init = {'pftank2t': -8, 'pf0tank2t': -8, 'eclipse': False}
    state_pair_dtype = {'pitch': np.float64, 'roll': np.float64}
    results_dtype = get_full_dtype(state_pair_dtype)

    pitch_vals = list(range(45, 181, 5))
    roll_vals = [-10, 0, 10]

    cases = {'2021:001:00:00:00': timbre.f_to_c([105.0, 110.0, 115.0]),
             '2021:091:00:00:00': timbre.f_to_c([105.0, 110.0, 115.0]),
             '2021:182:00:00:00': timbre.f_to_c([110.0, 115.0, 120.0]),
             '2021:274:00:00:00': timbre.f_to_c([110.0, 115.0, 120.0]),
             '2022:001:00:00:00': timbre.f_to_c([110.0, 115.0, 120.0])}

    state_pairs = [({'pitch': pn1, 'roll': rn1}, {'pitch': pn2, 'roll': rn2})
                   for pn1 in pitch_vals
                   for pn2 in pitch_vals
                   for rn1 in roll_vals
                   for rn2 in roll_vals]

    run_sets = [[10000, 20000, 30000, 40000], [50000, 60000, 70000, 80000], [90000, 100000]]

    print(f'Starting Timbre simulations on {DateTime().caldate}')

    run_cases(msid, model_spec, model_hash, init, run_sets, state_pairs, results_dtype, cases)

    print(f'Completed all Timbre simulations on {DateTime().caldate}')
