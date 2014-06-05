====================
FitFarquharModel
====================

Fitting routine for the Farquhar model parameters to a series of, or single measured A-Ci curve(s). 

Currently we have implemented two different fitting approaches. We now favour fitting all the A-Ci curves in a single step with associated temperature dependancies, whereas previously we fit each A-Ci curve separately (see below and Lin *et al*. 2013).

Our preferred fitting approach involves the assumption that the Vcmax, Rdfac and Jfac vary between leaves of a single species, however the temperature dependancies (Eav, Eaj, delSv and delSj) of these leaves do not. In addition we are fixing Hdv, Hdj and Ear.

Our previous approach (Lin *et al*. 2013), split the fitting approach down into a series of steps:

1. fitting Jmax, Vcmax and Rd at the measurement temperature, 
2. normalising the data to 25 degrees,
3. a series of stats tests to explore differences in the data (not part of the package - but we can help you out if you email one of us), e.g. differences between season, species etc.
4. finally, based on these stats tests, fit the model parameters Eaj, Eav, deltaSj and deltaSv.

In all cases the `Levenberg-Marquardt minimization algorithm <http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_ is used to fit the non-linear A-Ci curve data, using the lmfit package (see below).

The model is coded entirely in `Python 
<http://www.python.org/>`_ and we have some example `R <http://www.r-project.org/>`_ scripts for any stats analysis (just ask).

We have extensive tested the robustness of the fitting approach to retrieving model parameters under a series of scenarios where different levels of noise are added to synthetic datasets and results obtained using a bayesian approach. The code for the later I could probably dig out and I may add it to this package if I remember.

Installation
=============

They code has been written such that it has very few dependancies to ease personal set up. It depends on a series of standard python packages which install easily on all systems and a single fitting package which has installed with a single command on windows, macs and linux machines. The dependancies are:

* `numpy <http://numpy.scipy.org/>`_ 
* `scipy <http://www.scipy.org/>`_ 
* `matplotlib <http://matplotlib.sourceforge.net/>`_ 
* `lmfit <http://newville.github.com/lmfit-py/>`_  

All of these packages are widely used scientific packages which build easily on multiple platforms. For window or mac users the `Enthought <http://www.enthought.com/>`_ or `Anaconda <http://continuum.io/downloads>`_ python packages are perhaps your simplest avenue. I think from memory all of these packages except lmfit are installed by default if you follow either of these routes. On a Linux machine it is simply as case of using whatever your default package manager is, e.g. sudo apt-get install python2.7. If you are on a mac and don't want to use `Enthought <http://www.enthought.com/>`_ or `Anaconda <http://continuum.io/downloads>`_ then python comes as standard with your system. However, in my personal experience I've found that it is easier to set up your own separate working copy using a package manager such as `Macports <http://www.macports.org/>`_ or `Homebrew <http://brew.sh/>`_. I read that all the cool kids are now using the later, but personally I've had no issues with Macports. Python, numpy, scipy and matplotlib should all install via your package manager described above. You may need to install lmfit yourself (not sure). But regardless it is simple, there are instructions on the webpage listed above or just download the code, extract it and type ::

    python setup.py install

It is that simple. The code is being widely used on multiple platforms without issues, so if you get stuck email one of us for help (see below).

Once you have downloaded the source code, or clone the repository (go on...) there is a simple makefile, e.g. ::

    make install

or the standard python approach ::

    python setup.py install


Setting the input files up
==========================

Note in all cases the code has some expectations that the data is supplied in a consistent format, i.e. not the original Licor file. This should be easy to work out from the examples files (examples/data/example.csv), but feel free to contact one of us. Each A-Ci curve should be assigned to a unique curve number and curves measured at the same leaf should have the same leaf number. The usual format of the input file has the following columns: Curve, Tleaf, Ci, Photo, Species, Season, Leaf, fitgroup.

If your data doesn't have a different season that just put a unique identifier which is the same for all curves. So for example just fill the season column with summer for example.

Tleaf is expected to be supplied in degrees C. I ought to write something that checks this is the case (note to self), but given that the code will convert back and forth if the output looks bogus this would be the first thing I would check!

```python
s = "Python syntax highlighting"
print s
```

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

Massive disclaimer, I made these example files a (long) while again! I (certainly) probably haven't made a great deal of effort to maintain them, but email me if they don't work!

It should be apparent that to translate these example scripts to your own workspace is easy. If you didn't want to change the defaults then you would need to create a series of directories: data, plots, results. Inside the data directory you would place CSV files with your measured data. The code loops over all the files in this directory. The example scripts can be edited and used in your personal workspace, just remove the junk at the bottom which I added just to read the fitted data back in and print to the screen for the examples. For example, fit_all_curves_together.py would be edited so that this was all it contained. ::

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
Each class/method/function is typical documented and I have built html documentation pages which will outline usage, parameters etc. If you open any of the html files in your web browser you should be able to se these.

created using... (for my own reference!)
     pydoc -w ../fit_farquhar_model/*.py


Key References
==============
1. Farquhar, G.D., Caemmerer, S. V. and Berry, J. A. (1980) A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. *Planta*, **149**, 78-90.

2. Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley, P.C., Kirschbaum, M.U.F., Leroux, X., Montpied, P., Strassemeyer, J., Walcroft, A., Wang, K. and Loustau, D. (2002) Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data. *Plant, Cell and Enviroment*, **25**, 1167-1179.

3. Lin, Y-S., Medlyn, B. E., De Kauwe, M. G., and Ellsworth D. E. (2013) Biochemical photosynthetic responses to temperature: how do interspecific differences compare with seasonal shifts? *Tree Physiology*, **33**, 793-806.

     
Contacts
========
* Martin De Kauwe (mdekauwe at gmail.com)
* Yan-Shih Lin (yanshihl at gmail.com)
* Belinda Medlyn (bmedlyn at bio.mq.edu.au).
