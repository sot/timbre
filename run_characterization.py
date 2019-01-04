import sys
from os.path import expanduser

home = expanduser("~")
sys.path.append(home + '/AXAFLIB/timbre/')
from timbre import *

model_specs = load_model_specs()

msid = 'aacccdpt'
limit = -9.5
date = '2019:001:00:00:00'
step = 1

if __name__ == "__main__":

    date = '2019:182:00:00:00'

    run_characterization_parallel(msid, limit, date, 160000, step, model_init[msid], model_specs[msid],
                                  scale_factors=(200, 500, 1000, 2000))

    run_characterization_parallel(msid, limit, date, 140000, step, model_init[msid], model_specs[msid],
                                  scale_factors=(200, 500, 1000, 2000))

    run_characterization_parallel(msid, limit, date, 120000, step, model_init[msid], model_specs[msid],
                                  scale_factors=(200, 500, 1000, 2000))

    run_characterization_parallel(msid, limit, date, 100000, step, model_init[msid], model_specs[msid],
                                  scale_factors=(200, 500, 1000, 2000))

    run_characterization_parallel(msid, limit, date, 80000, step, model_init[msid], model_specs[msid],
                                  scale_factors=(200, 500, 1000, 2000))

