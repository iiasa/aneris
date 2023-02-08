from pandas import DataFrame, Series

from .compat import have_pyam, pyam

DEFAULT_INDEX = ("sector", "gas")


class Downscaler:
    methods = {
        "ipat_2100_gdp",
        "ipat_2150_pop": partial(intensity_convergence, convergence_year=2150, reference_name="pop"),
    }

    def add_method("..."):
        methods.update({


        })

    if have_pyam:
        @classmethod
        def from_pyam(cls, model: pyam.IamDataFrame, hist: pyam.IamDataFrame, region_mapping: Series, additional_data: pyam.IamDataFrame, ...):
            model = to_dataframe(model)
            hist = to_dataframe(heist) 

            ...
            return cls(model, hist, ..., return_type=pyam.IamDataFrame)
    
    def __init__(self, model: DataFrame, hist: DataFrame, region_mapping: Series, **additional_data: DataFrame):
        self.model = model
        self.hist = hist
        self.region_mapping = region_mapping
        self.data = additional_data
        
        assert (
            hist.groupby(["sector", "gas", "region"]).count() <= 1
        ).all(), "More than one hist"
        assert (
            projectlevel(model.index, ["sector", "gas", "region"])
            .difference(projectlevel(hist.index, ["sector", "gas", "region"]))
            .empty
        ), "History missing for some"


    def downscale(methods: Series) -> DataFrame:
        """Downscale aligned model data from historical data, and socio-economic scenario data.

        Notes
        -----
        model.index contains at least the downscaling index levels, but also any other
        levels.

        hist.index contains at least the downscaling index levels other index levels are
        allowed, but only one value per downscaling index value.

        Parameters
        ----------
        methods : Series
            Methods to apply
        config :
            Configuration settings like base_year
        """


        # Check that data contains what is needed for all methods in use, ie. inspect partial keywords 

        # Harmonise each model emissions trajectory to historical base year data using `methods`
        # ...

        return downscaled

    def methods(method_choice, overwrites):

        return methods
