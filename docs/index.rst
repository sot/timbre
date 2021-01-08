.. Timbre documentation master file, created by
   sphinx-quickstart on Tue Dec 22 14:23:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


############################################
Timbre Thermal Dwell Balance Estimation Tool
############################################

The timbre package enables one to calculate the required contiguous cold time to balance a specified hot observation, or
alternatively the maximum contiguous hot time yielded by a given cold observation, for a given Xija thermal model.

The primary purpose of this package is to enable the Chandra SOT Mission Planning Team to estimate the required mix of
observations that will yield a feasible schedule, without depending on past dwell profiles that may no longer be
relevant for future schedules.

Introduction
============

As Chandra's external thermal protective surfaces degrade over time, resulting in steadily increasing heat absorption
from the sun, the duration at which Chandra can reside in any given orientation to the sun before reaching a temperature
limit generally decreases. A set of temperature measurements within Chandra are used to represent thermal performance of
key components throughout the vehicle. Predictive Xija models are maintained for these temperature measurements to
enable SOT and FOT Mission Planning to plan observation schedules that maintain temperatures within specified limits.
These limits are set based on original qualification testing, material-based capability, or other performance-based
criteria, and are re-evaluated periodically to balance scheduling challenges with vehicle performance towards optimizing
science goals.

Occasionally, new locations need to be added to this set of modeled components, resulting in new planning challenges and
rendering past dwell profiles incompatible with future scheduling requirements. The strategy Timbre uses to determine
dwell time is independent of past profiles, and is therefore more flexible than approaches that had worked well earlier
in the mission. This documentation will first review how to run Timbre, then will delve into how Timbre works.

Basic Usage
===========

The example below represents a situation where one has a 21500 second hot dwell at 90.2 degrees pitch that they want to
include in a schedule near January 1st, 2021. They want to know how much time at 148.95 degrees pitch that should be
included in the schedule to balance this hot dwell from an ACA perspective.

Package Imports
---------------

    >>> from cxotime import CxoTime
    >>> from timbre import get_local_model, find_second_dwell

Define Model Information
------------------------

This information defines the model being used, as well as initial starting conditions. These starting conditions do not
significantly impact the results, however it is good practice to set them to be close to realistic values.

    >>> msid = 'aacccdpt'
    >>> limit = -6.5
    >>> date = '2021:001:00:00:00'
    >>> aca_model_spec, aca_md5 = get_local_model('./timbre/tests/data/aca_spec.json')
    >>> model_init = {'aacccdpt': -6.5, 'aca0': -6.5, 'eclipse': False}

Define Dwell #1 Information
---------------------------

Define the state information for the observation that one wants to balance. This could be a known hot dwell to be
included in a schedule. For the ACA model, we only need to define dwell time and pitch. Other models may need to have
more information defined. For example, the DPA model would need FEP count, CCD count, SIM position, clocking state, and
vid board state defined. Any information that the Xija model needs to run, that isn't defined in `model_init` needs to
be defined in each of the dwell state definitions, `dwell1_state` and `dwell2_state`.

    >>> t_dwell1 = 21500. # Seconds
    >>> dwell1_state = {'pitch': 90.2}

Define Dwell #2 Information
---------------------------

This state can be considered a candidate balancing dwell, such as a cooling attitude one wants to use to balance a hot
attitude. In this case we want to calculate the cold time necessary to balance the hot dwell 1 state, so we don't define
time this as in input parameter. As with `dwell1_state`, `dwell2_state` still needs all information necessary to run the
underlying Xija model.

    >>> dwell2_state = {'pitch': 148.95}

Calculate Dwell #2 Time
-----------------------

    >>> results = find_second_dwell(date, dwell1_state, dwell2_state, t_dwell1, msid, limit, aca_model_spec, model_init, limit_type='max')
    >>> results
    {'converged': True,
     'unconverged_hot': False,
     'unconverged_cold': False,
     'min_temp': -7.276062645249995,
     'mean_temp': -6.856483123898791,
     'max_temp': -6.5,
     'temperature_limit': -6.5,
     'dwell_2_time': 53894.85392717416,
     'min_pseudo': nan,
     'mean_pseudo': nan,
     'max_pseudo': nan,
     'hotter_state': 1,
     'colder_state': 2}

Explanation of Results
----------------------

The following information is returned by `find_second_dwell`:
 - `converged`: This is a boolean value that indicates whether or not a solution was possible. Solutions will not be
   possible in a number of situations:

   - Both dwells heat the location being modeled.
   - Both dwells cool the location being modeled.
   - One of the states defined results in a temperature that neither sufficiently heats or cools the location being
     modeled (this will sometimes converge but not always reliably).
   - The `dwell1_state` is hot and its dwell is long enough to heat this location from a steady state cold temperature
     to the specified hot limit, assuming the location is associated with a maximum temperature limit. In the case where
     a location is associated with a minimum temperature limit (e.g. PLINE03T), the opposite would apply, the
     `dwell1_state` is cold and is long enough to cool this location from a steady state hot temperture to the specified
     cold limit.

 - `unconverged_hot`: If the solution didn't converge, this will be `True` if the input values resulted in temperatures
   outside (e.g. above) the specified limit.
 - `unconverged_cold`: If the solution didn't converge, this will be `True` if the input values resulted in all
   temperatures within (e.g. below) the specified limit.
 - `min_temp`: This is the minimum temperature observed during the simulation (latter 2/3 actually to allow the model to
   reach a repeatable pattern - more on this later). This will be the limit for a converged solution in the case where
   the location is associated with a minimum temperature limit (e.g. PLINE03T).
 - `mean_temp`: This is the mean temperature observed during the simulation (latter 2/3)
 - `max_temp`: This is the maximum temperature observed during the simulation (latter 2/3). This will be the limit for a
   converged solution with a maximum temperature limit (e.g. AACCCDPT).
 - `temperature_limit`: This is the limit being used.
 - **`dwell2_time`: The dwell #2 time is what you are looking to calculate, and represents the dwell #2 state duration
   that balances the dwell #1 state at the specified duration, `t_dwell1`.**
 - `min_pseudo`: This is the min pseudo node temperature observed during the evaluated portion of the simulation. This
   is not implemented yet but is intended to eventually yield additional insight into the results.
 - `mean_pseudo`: This is the mean pseudo node temperature observed during the evaluated portion of the simulation. This
   is not implemented yet.
 - `max_pseudo`: This is the max pseudo node temperature observed during the evaluated portion of the simulation. This
   is not implemented yet.
 - `hotter_state`: This is an integer indicating which state is hotter, 1 or 2.
 - `colder_state`: This is an integer indicating which state is colder, 1 or 2, and is actually redundant with
   `hotter_state`.

Batch Processing
================

Multiple sets of cases can be run using a single function call, simplifying the generation of larger datasets.

Package Imports
---------------

The only difference from above is the import of `run_state_pairs` and `numpy`.

    >>> import numpy as np
    >>> from cxotime import CxoTime
    >>> from timbre import get_local_model, run_state_pairs

Define Model Information
------------------------

This is the same setup as used above.

    >>> msid = 'aacccdpt'
    >>> limit = -6.5
    >>> date = '2021:001:00:00:00'
    >>> aca_model_spec, aca_md5 = get_local_model('./timbre/tests/data/aca_spec.json')
    >>> model_init = {'aacccdpt': -6.5, 'aca0': -6.5, 'eclipse': False}
    >>> t_dwell1 = 21500. # Seconds

Define Dwell Cases
------------------

This is the most significant departure from above, instead of defining separate `dwell1_state` and `dwell2_state`
dictionary objects for a single case, pairs of `dwell1_state` and `dwell2_state` dictionary objects are combined into a
larger data structure. The `run_state_pairs` function will run through this data structure one pair at a time.

It should be noted that this set of cases will all use the same initial dwell time `t_dwell1` listed above.

    >>> state_pairs = (({'pitch': 144.2}, {'pitch': 154.95}),
    >>>                ({'pitch': 90.2}, {'pitch': 148.95}),
    >>>                ({'pitch': 50}, {'pitch': 140}),
    >>>                ({'pitch': 90}, {'pitch': 100}),
    >>>                ({'pitch': 75}, {'pitch': 130}),
    >>>                ({'pitch': 170}, {'pitch': 90}),
    >>>                ({'pitch': 90}, {'pitch': 170}))

Defining Return Datatypes
-------------------------

When running a single case it is easy to track of the inputs, such as pitch, CCD count, etc., so the data returned by
the `find_second_dwell` function does not include this information. When running multiple cases this task can become
more challenging. To facilitate easier tracking of this information, the `run_state_pairs` function includes this data
with each model result in a Numpy structured array, however one needs to define the data types used for each case to
avoid the need to infer data types (to be explicit).

In this case (aacccdpt model), there is only one type of information supplied for a given case, pitch. Since there is
only one type of information supplied, the dictionary containing this data has only one entry.

    >>> state_pair_dtype = {'pitch': np.float64}

Calculate Results
-----------------

    >>> results = run_state_pairs(msid, aca_model_spec, model_init, limit, date, t_dwell1, state_pairs,  state_pair_dtype, limit_type='max')

Explanation of Results Format
-----------------------------

The description of the results shown above is still valid for the similarly named items, however that information is
still included here for completeness, along with descriptions of the additional included information.

 - `msid`: This is the MSID that represents the location of interest, and is the primary output of a given model.
 - `date`: This is the date for which the simulation is applicable.
 - `datesecs`: This is the same date described by `date` only in seconds using the standard Ska Chandra epoch
 - `limit`: This is the temperature limit being used.
 - `tdwell_1`: This is the initial, "known", time that corresponds to the dwell #1 state. This is fixed for a batch of
   `state_pairs`.
 - **`tdwell_2`: The dwell #2 time is what you are looking to calculate, and represents the dwell #2 state duration that
   balances the dwell #1 state at the specified duration, `t_dwell1`.**
 - `min_temp`: This is the minimum temperature observed during the simulation (latter 2/3 actually to allow the model to
   reach a repeatable pattern - more on this later). This will be the limit for a converged solution in the case where
   the location is associated with a minimum temperature limit (e.g. PLINE03T).
 - `mean_temp`: This is the mean temperature observed during the simulation (latter 2/3)
 - `max_temp`: This is the maximum temperature observed during the simulation (latter 2/3). This will be the limit for a
   converged solution with a maximum temperature limit (e.g. AACCCDPT).
 - `min_pseudo`: This is the min pseudo node temperature observed during the evaluated portion of the simulation. This
   is not implemented yet but is intended to eventually yield additional insight into the results.
 - `mean_pseudo`: This is the mean pseudo node temperature observed during the evaluated portion of the simulation. This
   is not implemented yet.
 - `max_pseudo`: This is the max pseudo node temperature observed during the evaluated portion of the simulation. This
   is not implemented yet.
 - `converged`: This is a boolean value that indicates whether or not a solution was possible. Solutions will not be
   possible in a number of situations:

  - Both dwells heat the location being modeled.
  - Both dwells cool the location being modeled.
  - One of the states defined results in a temperature that neither sufficiently heats or cools the location being
    modeled (this will sometimes converge but not always reliably).
  - The `dwell1_state` is hot and its dwell is long enough to heat this location from a steady state cold temperature to
    the specified hot limit, assuming the location is associated with a maximum temperature limit. In the case where a
    location is associated with a minimum temperature limit (e.g. PLINE03T), the opposite would apply, the `dwell1_state`
    is cold and is long enough to cool this location from a steady state hot temperture to the specified cold limit.

 - `unconverged_hot`: If the solution didn't converge, this will be `True` if the input values resulted in temperatures
   outside (e.g. above) the specified limit.
 - `unconverged_cold`: If the solution didn't converge, this will be `True` if the input values resulted in all
   temperatures within (e.g. below) the specified limit.
 - `hotter_state`: This is an integer indicating which state is hotter, 1 or 2.
 - `colder_state`: This is an integer indicating which state is colder, 1 or 2, and is actually redundant with
   `hotter_state`.
 - `pitch1`: This is the pitch used as an input to a given simulation corresponding to the dwell #1 state.
 - `pitch2`: This is the pitch used as an input to a given simulation corresponding to the dwell #2 state.

    >>> import astropy
    >>> astropy.table.Table(results)

.. _table-label:

.. table:: Table of results from `run_state_pairs`

 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+
 | msid     | date     | datesecs      | limit   | t_dwell1 | t_dwell2   | min_temp | mean_temp | max_temp | min_pseudo | mean_pseudo | max_pseudo | converged | unconverged_hot | unconverged_cold | hotter_state | colder_state | pitch1  | pitch2  |
 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+
 | bytes20  | bytes8   | float64       | float64 | float64  | float64    | float64  | float64   | float64  | float64    | float64     | float64    | bool      | bool            | bool             | int8         | int8         | float64 | float64 |
 +==========+==========+===============+=========+==========+============+==========+===========+==========+============+=============+============+===========+=================+==================+==============+==============+=========+=========+
 | aacccdpt | 2021:001 | 725846469.184 | -6.5    | 21500.0  | nan        | -8.212   | -8.162    | -8.125   | nan        | nan         | nan        | False     | False           | True             | 1            | 2            | 144.2   | 154.95  |
 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+
 | aacccdpt | 2021:001 | 725846469.184 | -6.5    | 21500.0  | 53870.627  | -7.279   | -6.849    | -6.5     | nan        | nan         | nan        | True      | False           | False            | 1            | 2            | 90.2    | 148.95  |
 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+
 | aacccdpt | 2021:001 | 725846469.184 | -6.5    | 21500.0  | 169599.705 | -6.966   | -6.730    | -6.5     | nan        | nan         | nan        | True      | False           | False            | 2            | 1            | 50.0    | 140.0   |
 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+
 | aacccdpt | 2021:001 | 725846469.184 | -6.5    | 21500.0  | nan        | 0.811    | 0.867     | 0.968    | nan        | nan         | nan        | False     | True            | False            | 1            | 2            | 90.0    | 100.0   |
 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+
 | aacccdpt | 2021:001 | 725846469.184 | -6.5    | 21500.0  | nan        | -2.006   | -1.933    | -1.782   | nan        | nan         | nan        | False     | True            | False            | 1            | 2            | 75.0    | 130.0   |
 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+
 | aacccdpt | 2021:001 | 725846469.184 | -6.5    | 21500.0  | 40794.685  | -7.861   | -7.200    | -6.5     | nan        | nan         | nan        | True      | False           | False            | 2            | 1            | 170.0   | 90.0    |
 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+
 | aacccdpt | 2021:001 | 725846469.184 | -6.5    | 21500.0  | 10489.298  | -7.001   | -6.766    | -6.5     | nan        | nan         | nan        | True      | False           | False            | 1            | 2            | 90.0    | 170.0   |
 +----------+----------+---------------+---------+----------+------------+----------+-----------+----------+------------+-------------+------------+-----------+-----------------+------------------+--------------+--------------+---------+---------+

Discussion of Results
---------------------

Each number corresponds to a row in the above results:

1. This simulation included a dwell #1 pitch of 144.2 degrees, and a dwell #2 pitch of 154.95 degrees. Although the
   first dwell state is warmer, this state does not sufficiently heat this model at the given date to reach the
   specified limit, so these two states together result in an unconverged cold simulation.
2. This simulation included a dwell #1 pitch of 90.2 degrees, and a dwell #2 pitch of 148.95 degrees. This solution
   converged with a dwell #2 duration of approximately 53871 seconds at 148.95 degrees pitch calculated to sufficiently
   balance 21500 seconds at 90.2 degrees pitch on 2021:001.
3. This simulation included a dwell #1 pitch of 50.0 degrees, and a dwell #2 pitch of 140.0 degrees. This solution
   converged with a dwell #2 duration of approximately 169600 seconds at 140.0 degrees pitch calculated to sufficiently
   balance 21500 seconds at 140.0 degrees pitch on 2021:001. Although this condition did result in a solution, the
   exceedingly long time necessary at the dwell #2 condition would likely not be possible due to other thermal constraints.
4. This simulation included a dwell #1 pitch of 90.0 degrees, and a dwell #2 pitch of 100.0 degrees. As both of these
   states heat this modeled location, no solution is possible.
5. This simulation included a dwell #1 pitch of 75.0 degrees, and a dwell #2 pitch of 130.0 degrees. As both of these
   states heat this modeled location, no solution is possible.
6. This simulation included a dwell #1 pitch of 170.0 degrees, and a dwell #2 pitch of 90.0 degrees. This solution
   converged with a dwell #2 duration of approximately 40795 seconds at 90.0 degrees pitch calculated to sufficiently
   balance 21500 seconds at 170.0 degrees pitch on 2021:001.
7. This simulation included a dwell #1 pitch of 90.0 degrees, and a dwell #2 pitch of 170.0 degrees. This solution
   converged with a dwell #2 duration of approximately 10489 seconds at 170.0 degrees pitch calculated to sufficiently
   balance 21500 seconds at 90.0 degrees pitch on 2021:001.




.. toctree::
   :maxdepth: 2
   :caption: Contents:


API documentation
=================

.. automodule:: timbre.timbre
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
