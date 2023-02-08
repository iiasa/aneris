
DEFAULT_INDEX = ("sector", "gas", "region")

class Downscaler:

    @classmethod
    def from_pyam(cls, model: pyam.IamDataFrame, hist: pyam.IamDataFrame, region_mapping: Series, ...):
        model = to_dataframe(model)
        hist = to_dataframe(heist) 
        ...
        return cls(model, hist, ..., return_type=pyam.IamDataFrame)


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
        model : DataFrame
            Model data, year (int) as columns
        hist : DataFrame
            Historical data, year (int) as columns
        scenariodata : DataFrame

        methods : Series
            Methods to apply
        config :
            Configuration settings like base_year
        """

        assert (
            hist.groupby(["sector", "gas", "region"]).count() <= 1
        ).all(), "More than one hist"
        assert (
            projectlevel(model.index, ["sector", "gas", "region"])
            .difference(projectlevel(hist.index, ["sector", "gas", "region"]))
            .empty
        ), "History missing for some"

        # Harmonise each model emissions trajectory to historical base year data using `methods`
        # ...

        return downscaled

    def methods(method_choice, overwrites):

        return methods