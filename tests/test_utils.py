import pandas as pd
import pandas.testing as pdt
import pytest

import aneris.utils as utils


def combine_rows_df():
    df = pd.DataFrame(
        {
            "sector": [
                "sector1",
                "sector2",
                "sector1",
                "extra_b",
                "sector1",
            ],
            "region": ["a", "a", "b", "b", "c"],
            "2010": [1.0, 4.0, 2.0, 21, 42],
            "foo": [-1.0, -4.0, 2.0, 21, 42],
            "unit": ["Mt"] * 5,
            "gas": ["BC"] * 5,
        }
    ).set_index(utils.df_idx)
    return df


def test_isin():
    df = combine_rows_df()
    exp = pd.DataFrame(
        {
            "sector": [
                "sector1",
                "sector2",
                "sector1",
            ],
            "region": ["a", "a", "b"],
            "2010": [1.0, 4.0, 2.0],
            "foo": [-1.0, -4.0, 2.0],
            "unit": ["Mt"] * 3,
            "gas": ["BC"] * 3,
        }
    ).set_index(utils.df_idx)
    obs = exp.loc[
        utils.isin(sector=["sector1", "sector2"], region=["a", "b", "non-existent"])
    ]
    pdt.assert_frame_equal(obs, exp)

    with pytest.raises(KeyError):
        utils.isin(df, region="World", non_existing_level="foo")
