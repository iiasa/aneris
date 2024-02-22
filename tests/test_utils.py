import pandas as pd

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
