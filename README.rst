====================
FitFarquharModel
====================

Fitting routine for the Farquhar model parameters to a series of, or single measured A-Ci curve(s). 

Currently we have implemented two different fitting approaches. We now favour fitting all the A-Ci curves in a single step with associated temperature dependancies, whereas previously we fit each A-Ci curve separately (see below and Lin et al. 2013).

Our preferred fitting approach involves the assumption that the Vcmax, Rdfac and Jfac vary between leaves of a single species, however the temperature dependancies (Eav, Eaj, delSv and delSj) of these leaves do not. In addition we are fixing Hdv, Hdj and Ear.

Our previous approach (Lin et al. 2013), broke the fitting approach down into a series of steps:

(i) fitting Jmax, Vcmax and Rd at the measurement temperature, 
(ii) normalising the data to 25 degrees,
(iii) a series of stats tests to explore differences in the data (not part of the package - but we can help you out if you email one of us), e.g. differences between season, species etc.
(iv) finally, based on (iii) fit the model parameters Eaj, Eav, deltaSj and deltaSv.

Note in all cases the code has some expectations that the data is supplied in a consistent format, i.e. not the original Licor file. This should be easy to work out from the examples files, but feel free to contact one of us. Each A-Ci curve should be assigned to a unique curve number and curves measured at the same leaf should have the same leaf number.

In all cases the Levenberg-Marquardt minimization algorithm is used to fit the non-linear A-Ci curve data, using the lmfit package (see below).

The model is coded entirely in `Python 
<http://www.python.org/>`_.



Installation
=============

They code has very few dependancies and depending on your operating system the ease of setup will vary. The dependancies are:

* `numpy <http://numpy.scipy.org/>`_ 
* `scipy <http://www.scipy.org/>`_ 
* `matplotlib <http://matplotlib.sourceforge.net/>`_ 
* `lmfit <http://newville.github.com/lmfit-py/>`_  

All of these packages are widely used scientific packages which build easily on multiple platforms. For window users the `Enthought <http://www.enthought.com/>`_ or `Anaconda <http://continuum.io/downloads>`_ python packages are perhaps your simplest avenue. We have set up the necessary packages above and it very straightforward. On a Linux machine it is simply as case of using whatever your default package manager is, e.g. sudo apt-get install python2.7. If you are on a mac then python comes as standard. However, in my personal experience I've found that it is easier to set up your own separate working copy using a package manager such as `Macports <http://www.macports.org/>`_ or `Homebrew <http://brew.sh/>`_. I read that all the cool kids are now using the later, but personally I've had no issues with Macports. Python, numpy, scipy and matplotlib should all install via your package manager described above. You may need to install lmfit yourself (not sure). But regardless it is simple, there are instructions on the webpage listed above or just download the code, extract it and type ::

    python setup.py install

It is that simple. The code is being widely used on multiple platforms without issues, so if you get stuck email one of us for help (see below).

Once you have downloaded the source code, or clone the repository (go on...) there is a simple makefile, e.g. ::

    make install

or the standard python approach ::

    python setup.py install

Running the code
=================

The examples directory contains a series of self contained example scripts. Names should be obvious, but the typical order of running them would be... ::

    create_example_data.py
    
If fitting everything in a single step then ::

    fit_all_curves_together.py

If using the two step fitting approach then ::    
    
    fit_jmax_vcmax_rd.py
    normalise_data.py
    fit_ea_dels.py

Of course with real data a series of stats test should be carried out before
fitting the final Eav, Eaj, deltaSv and deltaSj parameters. This is left up to the individual user.

Massive disclaimer, I made these example files a while again! I probably haven't made a great deal of effort to maintain them, but email me if they don't work!

It should be apparent that to translate these example scripts to your own workspace is easy. If you didn't want to change the defaults then you would need to create a series of directories: data, plots, results. Inside the data directory you would place all your measurements. The code loops over all the files in this directory! The example scripts can be edited and used in your personal workspace, just remove the junk at the bottom which is read the fits back in and printing to the screen. For example, fit_all_curves_together.py would be edited so that this was all it contained. ::

    import os
    import sys
    import glob

    from fit_farquhar_model.farquhar_model import FarquharC3
    from fit_farquhar_model.fit_dummy_version_ear_const import FitMe

    ofname = "fitting_results.csv"
    results_dir = "results"
    data_dir = "data"
    plot_dir = "plots"
    model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=False)

    F = FitMe(model, ofname, results_dir, data_dir, plot_dir)
    F.main(print_to_screen=False)     

    
Documentation
=============
Minimal python docstring documentation output as html files in subdirectory
html_documentation.

created using... 
     pydoc -w ../fit_farquhar_model/*.py


Key References
==============
1. Farquhar, G.D., Caemmerer, S. V. and Berry, J. A. (1980) A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. *Planta*, **149**, 78-90.

2. Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley, P.C., Kirschbaum, M.U.F., Leroux, X., Montpied, P., Strassemeyer, J., Walcroft, A., Wang, K. and Loustau, D. (2002) Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data. *Plant, Cell and Enviroment*, **25**, 1167-1179.

3. Lin, Y-S., Medlyn, B. E., De Kauwe, M. G., and Ellsworth D. E. (2013) Biochemical photosynthetic responses to temperature: how do interspecific differences compare with seasonal shifts? *Tree Physiology*, **33**, 793-806.

     
Contacts
========
Martin De Kauwe (mdekauwe at gmail.com)

Yan-Shih Lin (yanshihl at gmail.com)

Belinda Medlyn (bmedlyn at bio.mq.edu.au).
