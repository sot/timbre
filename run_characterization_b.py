import sys
from os.path import expanduser
import pickle

home = expanduser("~")
sys.path.append(home + '/AXAFLIB/timbre/')
from timbre import *

model_specs = load_model_specs()
msid = 'pftank2t'
limit = 100
datestamp = DateTime().caldate[:9]
init = {'pftank2t': f_to_c(95.), 'pf0tank2t': f_to_c(95.), 'eclipse': False}

if __name__ == "__main__":

    state_pairs = [({'duration1': t1, 'duration1_fraction': 1.0, 'sequence1': -1, 'obsid1': 99999, 'pitch': p1, 'roll': 0.0},
                    {'pitch': p2, 'roll': 0.0, 'sequence2': -1, 'obsid2': 99999})
                   for p1 in range(45, 170, 1)
                   for p2 in range(45, 170, 1)
                   for t1 in [10000, 20000, 30000, 40000, 50000]]
    date = '2020:001:00:00:00'
    t1 = DateTime().secs
    results = run_state_pairs(msid, model_specs[msid], init, limit, date, state_pairs, max_dwell=200000)
    t2 = DateTime().secs
    print('took {} seconds, for {} state pairs'.format(t2 - t1, len(state_pairs)))
    pickle.dump(results, open('pftank2t_new_{}_{}.pkl'.format(datestamp, 2), 'wb'))
