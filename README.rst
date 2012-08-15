====================
FitFarquharModel
====================

Fitting routine for the Farquhar model parameters to a series of measured A-Ci curve data. Fitting is carried out in a series of separate steps: (i) fitting Jmax, Vcmax and Rd at the measurement temperature, (ii) normalising the data to 25 degrees (iii) a series of stats tests to explore differences in the data (not part of the package), e.g. differences between season, species etc, and (iv) finally, based on (iii) fit the model parameters Eaj, Eav, deltaSj and deltaSv.

The Levenberg-Marquardt minimization algorithm is used to fit the non-linear
A-Ci curve data, using the lmfit package (see below).

The model is coded entirely in `Python 
<http://www.python.org/>`_.


Key References
==============
1). Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley, P.C., Kirschbaum, M.U.F., Leroux, X., Montpied, P., Strassemeyer, J., Walcroft, A., Wang, K. and Loustau, D. (2002) Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data. Plant, Cell and Enviroment 25, 1167-1179.

2). Farquhar, G.D., Caemmerer, S. V. and Berry, J. A. (1980) A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. Planta, 149, 78-90.

.. contents:: :local:

Installation
=============

They code depends on:

* `numpy <http://numpy.scipy.org/>`_ 
* `scipy <http://www.scipy.org/>`_ 
* `scipy <http://www.scipy.org/>`_  
* `matplotlib <http://matplotlib.sourceforge.net/>`_ 
* `lmfit <http://newville.github.com/lmfit-py/>`_  

All of these packages are widely used scientific packages which build easily on multiple platforms. For window users the Enthought python package is perhaps your simplest avenue (http://www.enthought.com/).

Once you have downloaded the source code, or clone the repository (go on...) there is a simple makefile, e.g. ::

    make install

or the standard python approach ::

    python setup.py install

Running the code
=================

The examples directory contains a series of self contained example scripts. Names should be obvious, but the typical order of running them would be... ::

    create_example_data.py
    fit_jmax_vcmax_rd.py
    normalise_data.py
    fit_ea_dels.py

Of course with real data a series of stats test should be carried out before
fitting the final Eav, Eaj, deltaSv and deltaSj parameters. This is left up to the individual user.
    
