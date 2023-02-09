
# INPUT
#
# Scenario data                 Sector, gas, regions/global
# Historical data, at high-res. Sector, gas, country/global, dimensions may have size 1. ie global/total.
#
#
# OUTPUT
#
# 


# Steps:
# 1. Split operations into different sets with homogeneous dataset dimensions
#    (ie. particular sector/gas only defined globally, others per region/country)
#    Keep note of dimensions to operate on for each set.
# 2. Set up operation chain for each set.
# 2. a. Region aggregation as necessary for different operation steps. (aggregate historical data to model regions)
# 2. b. 
# 3. Postprocessing of results, ie. re-create a R5 regions f.ex. (maybe off-load to pyam/nomenclature)



# Notes
# -----
# 1. If historical data only has world emissions for sector/gas and model has it regionally defined, we raise
# 2. If model data is provided for World and model regions:
#    - We throw World away and warn about it
#    - if user has Transport CO2 for road per regions and for aircraft per World, then (s)he needs to split the sector explicitly.
# 3. Per-module configuration (yaml-based). Master configuration file, downstream module takes a configuration argument.

