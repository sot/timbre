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
from the sun, many components have begun to approach their health and safety thermal limits. As these components begin
to approach their limits, predictive thermal models for these locations are built using the Xija thermal modeling
framework, enabling SOT and FOT Mission Planning to plan observation schedules that maintain temperatures within
specified limits. These limits are set based on original qualification testing, material-based capability, or other
performance-based criteria, and are re-evaluated periodically to balance scheduling challenges with vehicle performance
towards optimizing science goals.

Occasionally, new locations need to be added to this set of modeled components, resulting in new planning challenges and
rendering past scheduling strategies incompatible with future scheduling requirements. The algorithm Timbre uses to
determine dwell time is independent of past profiles, and is therefore more flexible than approaches that have used
these past observing profiles to characterize future capability, which had worked very well earlier in the mission.

This documentation will first cover how to run Timbre to calculate the estimated dwell time to balance two known
configurations (e.g. pitch, roll, fep_count, etc.) and a known initial dwell time. After covering very basic usage,
further examples will show how to use this data to characterize dwell capability and how to use the interaction of all
thermal models to generate composite maximum dwell estimates.

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents

   building_blocks
   model_characterization
   shorter_durations
   api_documentation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
