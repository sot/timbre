
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='timbre',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      description='Thermal Dwell Time Balance Estimation Tool',
      author='Matthew Dahmer',
      author_email='mdahmer@ipa.harvard.edu',
      packages=['timbre', 'timbre.tests'],
      package_data={'timbre.tests': ['data/*']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
