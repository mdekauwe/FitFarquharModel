====================
FitFarquharModel
====================

Fitting routine for the Farquhar model parameters to a series of (or single) measured A-Ci curve data. 

Currently we have implemented two different fitting approaches. We now favour fitting all the A-Ci curves in a single step, whereas previously we fit each A-Ci curve separately (see below and Lin et al. 2013).

Our preferred fitting approach involves the assumption that the Vcmax, Rdfac and Jfac vary between leaves of a single species, however the temperature dependancies (Eav, Eaj, delSv and delSj) of these leaves do not. In addition we are fixing Hdv, Hdj and Ear.

Our previous approach (Lin et al. 2013), broke the fitting approach down into a series of steps:

(i) fitting Jmax, Vcmax and Rd at the measurement temperature, 
(ii) normalising the data to 25 degrees,
(iii) a series of stats tests to explore differences in the data (not part of the package - but we can help you out if you email one of us), e.g. differences between season, species etc.
(iv) finally, based on (iii) fit the model parameters Eaj, Eav, deltaSj and deltaSv.

In all cases the Levenberg-Marquardt minimization algorithm is used to fit the non-linear A-Ci curve data, using the lmfit package (see below).

The model is coded entirely in `Python 
<http://www.python.org/>`_.

Key References
==============
1). Farquhar, G.D., Caemmerer, S. V. and Berry, J. A. (1980) A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. Planta, 149, 78-90.

2). Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley, P.C., Kirschbaum, M.U.F., Leroux, X., Montpied, P., Strassemeyer, J., Walcroft, A., Wang, K. and Loustau, D. (2002) Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data. Plant, Cell and Enviroment 25, 1167-1179.

3). Lin, Y-S., Medlyn, B. E., De Kauwe, M. G., and Ellsworth D. E. (2013) Biochemical photosynthetic responses to temperature: how do interspecific differences compare with seasonal shifts? Tree Physiology, 33, 793-806.
.. contents:: :local:

Installation
=============

They code has very few dependancies and depending on your operating system the ease of setup will vary. The dependancies are:

* `numpy <http://numpy.scipy.org/>`_ 
* `scipy <http://www.scipy.org/>`_ 
* `matplotlib <http://matplotlib.sourceforge.net/>`_ 
* `lmfit <http://newville.github.com/lmfit-py/>`_  

All of these packages are widely used scientific packages which build easily on multiple platforms. For window users the Enthought ((http://www.enthought.com/) or Anaconda (http://continuum.io/downloads) python packages are perhaps your simplest avenue. We have set up the necessary packages above and it very straightforward. On a Linux machine it is simply as case of using whatever your default package manager is, e.g. sudo apt-get install python2.7. If you are on a mac then python comes as standard. However, in my personal experience I've found that it is easier to set up your own separate working copy using a package manager such as Macports (http://www.macports.org/) or Homebrew (http://brew.sh/). I read that all the cool kids are now using the later, but personally I've had no issues with Macports.

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
    
Documentation
=============
Minimal python docstring documentation output as html files in subdirectory
html_documentation.

created using... 
    pydoc -w ./
