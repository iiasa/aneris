from typing import Optional
from pandas import DataFrame, Series
"""
Harmonization module for harmonizing entry points and methods
"""

DEFAULT_INDEX = ("sector", "gas", "region")

class Harmonizer:
    @classmethod
    def from_pyam(cls, model: pyam.IamDataFrame, hist: pyam.IamDataFrame, ...):
        model = to_dataframe(model)
        hist = to_dataframe(heist) 
        ...
        return cls(model, hist, ..., return_type=pyam.IamDataFrame)

    def __init__(self, model: DataFrame, hist: DataFrame, year: int, index: Optional[tuple] = DEFAULT_INDEX, return_type = DataFrame):
        """Harmonize aligned model data to historical data

        Notes
        -----
        model.index contains at least the harmonization index levels, but also any other
        levels.

        hist.index contains at least the harmonization index levels other index levels are
        allowed, but only one value per harmonization index value.

        Parameters
        ----------
        model : DataFrame
            Model data, sector, gas, region index, year (int) as columns
        hist : DataFrame
            Historical data
        year : int
            Harmonization year
        index : tuple, optional
            Index levels over which to harmonise
        """
        assert (hist.groupby(index).count() <= 1).all(), "Ambiguous history"
        assert (
            projectlevel(model.index, index)
            .difference(projectlevel(hist[year].dropna().index, index))
            .empty
        ), "History missing for some"


        # Unit handling if there is a unit index level
        # ...


        self.model = model
        self.hist = hist
        self.year = year
        self.index = index


    def harmonize(
        methods: Series = None
    ) -> DataFrame:
        """Harmonize aligned model data to historical data

        Notes
        -----
        model.index contains at least the harmonization index levels, but also any other
        levels.

        hist.index contains at least the harmonization index levels other index levels are
        allowed, but only one value per harmonization index value.

        Parameters
        ----------
        methods : Series
            Methods to apply
        """

        # Harmonise each model emissions trajectory to historical base year data using `methods`
        # ...

        return self.return_type(harmonized)


    def methods(method_choice = DEFAULT_METHOD_CHOICE, overwrites):
        # Make default method choices based on selection logic
        for each model, scenario:
            method[model, scenario, sector, gas, ] = method_choice
        
        return methods


def harmonize_inspection(model, hist, methods):
    harmonizer = Harmonizer(model, hist, ...)
    quality = pd.Series()
    for method in methods:
        harmonisation_results = harmonizer.harmonize(method)
        quality[idx, method] = qa(hist, harmonisation_results)
    
    return quality.groupby(idx).argmax()

