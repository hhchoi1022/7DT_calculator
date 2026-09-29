"""
7DT observation calculator: core library.

Sub-modules
-----------
config      : live access to the 7DT configuration folder (site, filters, observation modes)
filters     : filter-name helpers (ordering, nominal wavelength, colours)
photometry  : filter transmission curves, spectra, synthetic AB photometry
targets     : target container, coordinate parsing, name resolving
visibility  : (1) visibility of targets from the 7DT site
etc         : (2) exposure time / SNR calculator built on the empirical depth model
overhead    : (3) observation overhead estimate
tiles       : (4) 7DS tile matcher
"""
__version__ = '0.1.0'
