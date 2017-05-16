import pandas as pd

from numpy.testing import assert_almost_equal
import nose.tools as nt
import pandas.util.testing as pdt

from aneris import utils


# def test_16_to_9():
#     df16 = pd.read_csv('subset_iso_16.csv')
#     df9 = utils.sectors16to9(df16, verify=False).set_index(utils.df_idx)

#     # a few observations
#     assert_almost_equal(df9.loc['aut', 'BC', 'Industrial Sector', 'kt'][
#                         '2010'], 1.109518, 6)


# def test_iso_to_Native():
#     df = pd.read_csv('subset_iso_16.csv')
#     obs = utils.regionISOtoNative(df, verify=False).set_index(utils.df_idx)

#     # a few observations
#     idx = ('NAM', 'BC', 'Industrial Combustion', 'kt')
#     can = 42
#     usa = 48.5703026463
#     assert_almost_equal(obs.loc[idx]['2010'], can + usa, 6)


# def test_iso_to_5():
#     df = pd.read_csv('subset_iso_16.csv')
#     obs = utils.regionISOto5(df, verify=False).set_index(utils.df_idx)

#     # a few observations
#     idx = ('R5OECD', 'BC', 'Industrial Combustion', 'kt')
#     can = 42
#     usa = 48.5703026463
#     aut = 1.1095181444
#     assert_almost_equal(obs.loc[idx]['2010'], can + usa + aut, 6)


# def test_Native_to_5():
#     df = pd.read_csv('subset_iso_16.csv')
#     df = utils.regionISOtoNative(df, verify=False).set_index(utils.df_idx)
#     obs = utils.regionNativeto5(df, verify=False)

#     # a few observations
#     idx = ('R5OECD', 'BC', 'Industrial Combustion', 'kt')
#     can = 42
#     usa = 48.5703026463
#     aut = 1.1095181444
#     assert_almost_equal(obs.loc[idx]['2010'], can + usa + aut, 6)


def test_remove_trailing_numbers():
    nt.assert_equal('foo', utils.remove_trailing_numbers('foo'))
    nt.assert_equal('foo', utils.remove_trailing_numbers('foo|42'))
    nt.assert_equal('foo|42', utils.remove_trailing_numbers('foo|42|9000'))


def test_remove_emissions_prefix():
    nt.assert_equal('foo', utils.remove_emissions_prefix('foo'))
    nt.assert_equal('foo', utils.remove_emissions_prefix('Emissions|XXX|foo'))
    nt.assert_equal('Emissions|bar|foo',
                    utils.remove_emissions_prefix('Emissions|bar|foo'))
    nt.assert_equal('foo', utils.remove_emissions_prefix(
        'Emissions|bar|foo', gas='bar'))


# def test_generic_map():
#     df = pd.DataFrame({
#         'sector': [
#             '1A1a_Electricity-autoproducer',
#             '1A1a_Electricity-public',
#             '1A3di_International-shipping',
#         ],
#         'region': ['a'] * 3,
#         '2010': [1.0, 4.0, 2.0],
#         'units': ['Mt'] * 3,
#         'gas': ['BC'] * 3,
#     }).set_index(utils.df_idx).sort_index()

#     exp = pd.DataFrame({
#         'sector': [
#             'Energy|Supply|Electricity',
#             'Energy|Demand|Transportation|Shipping|International|Enroute',
#         ],
#         'region': ['a'] * 2,
#         '2010': [5.0, 2.0],
#         'units': ['Mt'] * 2,
#         'gas': ['BC'] * 2,
#     }).set_index(utils.df_idx).sort_index()
#     obs = utils.agg_sectors(df, sfrom='CEDS_59', sto='IAMC')
#     pdt.assert_frame_equal(obs, exp)


def test_region_agg_funky_name():
    df = pd.DataFrame({
        'sector': ['foo', 'foo'],
        'region': ['a', 'b'],
        '2010': [1.0, 4.0],
        'units': ['Mt'] * 2,
        'gas': ['BC'] * 2,
    }).set_index(utils.df_idx).sort_index()
    mapping = pd.DataFrame(
        [['fOO_Bar', 'a'], ['fOO_Bar', 'b']], columns=['x', 'y'])
    exp = pd.DataFrame({
        'sector': ['foo'],
        'region': ['fOO_Bar'],
        '2010': [5.0],
        'units': ['Mt'],
        'gas': ['BC'],
    }).set_index(utils.df_idx).sort_index()
    obs = utils.agg_regions(df, rfrom='y', rto='x', mapping=mapping)
    print(obs)
    pdt.assert_frame_equal(obs, exp)


# def test_aggregator_add_variables():
#     df = pd.DataFrame({
#         'sector': [
#             'AFOLU|Agriculture',
#             'AFOLU|Biomass Burning',
#             'Energy',
#             'Industrial Processes',
#         ],
#         'region': ['a'] * 4,
#         '2010': [5.0, 2.0, -1.0, 42.],
#         'units': ['kt'] * 4,
#         'gas': ['BC'] * 4,
#     })
#     obs = (
#         utils.EmissionsAggregator(df)
#         .add_variables(totals='', aggregates=True, ceds_types='Unharmonized')
#         .df
#     ).set_index(utils.df_idx).sort_index()
#     sectors = {
#         'AFOLU|Agriculture': 5.0,
#         'AFOLU|Biomass Burning': 2.0,
#         'Energy': -1.0,
#         'Industrial Processes': 42.0,
#         '': 5.0 + 2.0 + -1.0 + 42.0,
#         'Aggregate - Fossil Fuels and Industry': -1.0 + 42.0,
#         'CEDS+|16+ Sectors|Agriculture|Unharmonized': 5.0,
#         'CEDS+|9+ Sectors|Agriculture|Unharmonized': 5.0,
#         'CEDS+|16+ Sectors|Agricultural Waste Burning|Unharmonized': 2.0,
#         'CEDS+|9+ Sectors|Agricultural Waste Burning|Unharmonized': 2.0,
#         'CEDS+|16+ Sectors|Industrial Process and Product Use|Unharmonized': 42.0,
#         'CEDS+|9+ Sectors|Industrial Sector|Unharmonized': 42.0,
#         'CEDS+|16+ Sectors|Unharmonized': 5.0 + 2.0 + 42.0,
#         'CEDS+|9+ Sectors|Unharmonized': 5.0 + 2.0 + 42,
#     }
#     exp = pd.DataFrame({
#         'sector': list(sectors.keys()),
#         'region': ['a'] * len(sectors),
#         '2010': list(sectors.values()),
#         'units': ['kt'] * len(sectors),
#         'gas': ['BC'] * len(sectors),
#     }).set_index(utils.df_idx).sort_index()
#     pdt.assert_frame_equal(obs, exp, check_index_type=False)


# def test_sec_map():
#     m = utils.sec_map(59).set_index('CEDS_59')
#     obs = m.loc['1A3Di_International-Shipping']['IAMC']
#     exp = 'Energy|Demand|Transportation|Shipping|International|Enroute'
#     nt.assert_equal(obs, exp)

#     m = utils.sec_map(16).set_index('CEDS_16')
#     obs = m.loc['Electricity And Heat Production']['IAMC']
#     exp = 'Energy|Supply|Aggregate - Electricity and Heat'
#     nt.assert_equal(obs, exp)

#     m = utils.sec_map(9).set_index('CEDS_9')
#     obs = m.loc['Energy Sector']['IAMC'].tolist()
#     exp = [u'Energy|Supply', u'Fossil Fuel Fires']
#     nt.assert_equal(obs, exp)


def test_no_repeat_gases():
    gases = utils.all_gases
    pdt.assert_equal(len(gases), len(set(gases)))


def test_gases():
    var_col = pd.Series(['foo|Emissions|CH4|bar', 'Emissions|N2O|baz|zing'])
    exp = pd.Series(['CH4', 'N2O'])
    obs = utils.gases(var_col)
    pdt.assert_series_equal(obs, exp)


def test_units():
    var_col = pd.Series(['foo|Emissions|CH4|bar', 'Emissions|N2O|baz|zing'])
    exp = pd.Series(['Mt CH4/yr', 'kt N2O/yr'])
    obs = utils.units(var_col)
    pdt.assert_series_equal(obs, exp)


def test_naked_vars():
    var_col = pd.Series(['foo|Emissions|CH4|bar', 'Emissions|N2O|baz|zing'])
    exp = pd.Series(['bar', 'baz|zing'])
    obs = utils.naked_vars(var_col)
    pdt.assert_series_equal(obs, exp)


def test_prefixes():
    var_col = pd.Series(['foo|Emissions|CH4|bar', 'Emissions|N2O|baz|zing'])
    exp = pd.Series(['foo|Emissions|CH4', 'Emissions|N2O'])
    obs = utils.var_prefix(var_col)
    pdt.assert_series_equal(obs, exp)


def test_suffixes():
    var_col = pd.Series(['foo|Emissions|CH4|bar',
                         'Emissions|N2O|baz|Harmonized',
                         'Emissions|N2O|baz|Unharmonized'])
    exp = pd.Series(['', 'Harmonized', 'Unharmonized'])
    obs = utils.var_suffix(var_col)
    pdt.assert_series_equal(obs, exp)


def test_formatter_to_std():
    df = pd.DataFrame({
        'Variable': [
            'CEDS+|9+ Sectors|Emissions|BC|foo|Unharmonized',
            'Emissions|BC|bar|baz',
        ],
        'Region': ['a', 'b'],
        '2010': [5.0, 2.0],
        '2020': [-1.0, 3.0],
        'Unit': ['Mt foo/yr'] * 2,
        'Model': ['foo'] * 2,
        'Scenario': ['foo'] * 2,
    })

    fmt = utils.FormatTranslator(df.copy())
    obs = fmt.to_std()
    exp = pd.DataFrame({
        'sector': [
            'CEDS+|9+ Sectors|foo|Unharmonized',
            'bar|baz',
        ],
        'region': ['a', 'b'],
        '2010': [5000.0, 2000.0],
        '2020': [-1000.0, 3000.0],
        'units': ['kt'] * 2,
        'gas': ['BC'] * 2,
    })
    pdt.assert_frame_equal(obs.set_index(utils.df_idx),
                           exp.set_index(utils.df_idx))


def test_formatter_to_template():
    df = pd.DataFrame({
        'Variable': [
            'CEDS+|9+ Sectors|Emissions|BC|foo|Unharmonized',
            'Emissions|BC|bar|baz',
        ],
        'Region': ['a', 'b'],
        '2010': [5.0, 2.0],
        '2020': [-1.0, 3.0],
        'Unit': ['Mt BC/yr'] * 2,
        'Model': ['foo'] * 2,
        'Scenario': ['foo'] * 2,
    }).set_index(utils.iamc_idx)
    fmt = utils.FormatTranslator(df)
    std = fmt.to_std()
    obs = fmt.to_template()
    exp = df.reindex_axis(obs.columns, axis=1)
    pdt.assert_frame_equal(obs, exp)


def combine_rows_df():
    df = pd.DataFrame({
        'sector': [
            '1A1a_Electricity-autoproducer',
            '1A1a_Electricity-public',
            '1A1a_Electricity-autoproducer',
            'extra_b',
            '1A1a_Electricity-autoproducer',
        ],
        'region': ['a', 'a', 'b', 'b', 'c'],
        '2010': [1.0, 4.0, 2.0, 21, 42],
        'foo': [-1.0, -4.0, 2.0, 21, 42],
        'units': ['Mt'] * 5,
        'gas': ['BC'] * 5,
    }).set_index(utils.df_idx)
    return df


def test_combine_rows_default():
    df = combine_rows_df()
    exp = pd.DataFrame({
        'sector': [
            '1A1a_Electricity-autoproducer',
            '1A1a_Electricity-public',
            'extra_b',
            '1A1a_Electricity-autoproducer',
        ],
        'region': ['a', 'a', 'a', 'c'],
        '2010': [3.0, 4.0, 21, 42],
        'foo': [1.0, -4.0, 21, 42],
        'units': ['Mt'] * 4,
        'gas': ['BC'] * 4,
    }).set_index(utils.df_idx)
    obs = utils.combine_rows(df, 'region', 'a', ['b'])
    pdt.assert_frame_equal(obs, exp.reindex_axis(obs.columns, axis=1))


def test_combine_rows_dropothers():
    df = combine_rows_df()
    exp = pd.DataFrame({
        'sector': [
            '1A1a_Electricity-autoproducer',
            '1A1a_Electricity-public',
            'extra_b',
            '1A1a_Electricity-autoproducer',
            'extra_b',
            '1A1a_Electricity-autoproducer',
        ],
        'region': ['a', 'a', 'a', 'b', 'b', 'c'],
        '2010': [3.0, 4.0, 21, 2.0, 21, 42],
        'foo': [1.0, -4.0, 21, 2.0, 21, 42],
        'units': ['Mt'] * 6,
        'gas': ['BC'] * 6,
    }).set_index(utils.df_idx)
    obs = utils.combine_rows(df, 'region', 'a', ['b'], dropothers=False)
    pdt.assert_frame_equal(obs, exp.reindex_axis(obs.columns, axis=1))


def test_combine_rows_sumall():
    df = combine_rows_df()
    exp = pd.DataFrame({
        'sector': [
            '1A1a_Electricity-autoproducer',
            'extra_b',
            '1A1a_Electricity-autoproducer',
        ],
        'region': ['a', 'a', 'c'],
        '2010': [2.0, 21, 42],
        'foo': [2.0, 21, 42],
        'units': ['Mt'] * 3,
        'gas': ['BC'] * 3,
    }).set_index(utils.df_idx)
    obs = utils.combine_rows(df, 'region', 'a', ['b'], sumall=False)
    pdt.assert_frame_equal(obs, exp.reindex_axis(obs.columns, axis=1))
