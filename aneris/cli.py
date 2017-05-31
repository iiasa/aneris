import argparser

from aneris import harmonize, utils, _io


def main(inf, history=None, regions=None, output_prefix=None, add_5region=True):
    # construct parser
    descr = """
    Harmonize historical trajectories to data in the IAMC template format.

    Example usage:

    aneris input.xlsx --history history.csv --regions regions.csv
    """
    parser = argparse.ArgumentParser(
        description=descr,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    inf = 'Input data file.'
    parser.add_argument('inf', help=inf)
    history = 'Historical emissions in the base year.'
    parser.add_argument('--history', help=history,
                        default=utils.hist_path('history.csv'))
    regions = 'Mapping of country iso-codes to native regions.'
    parser.add_argument('--regions', help=regions,
                        default=utils.region_path('message.csv'))
    rc = 'Runcontrol YAML file (see <WEBSITE> for examples).'
    parser.add_argument('--rc', help=rc, default=None)
    output_prefix = 'Prefix to use for output file names.'
    parser.add_argument('--output_prefix', help=output_prefix, default=None)

    # parse cli
    args = parser.parse_args()
    inf = args.inf
    history = args.history
    regions = args.regions
    rc = args.rc
    output_prefix = args.output_prefix

    # read input
    hist = utils.pd_read(history)
    if hist.empty:
        raise ValueError('History file is empty')
    regions = utils.pd_read(regions)
    if regions.empty:
        raise ValueError('Region definition is empty')
    model, overrides, config = utils.read_excel(inf)
    rc = _io.RunControl(rc=rc)
    rc['config'] = _io.recursive_update(rc['config'], config)

    # do core harmonization
    driver = harmonize.HarmonizationDriver(rc, model, hist, overrides, regions)
    for scenario in driver.scenarios():
        driver.harmonize(scenario)
    model, metadata = driver.harmonized_results()

    # write to excel
    prefix = output_prefix or inf.split('.')[0]
    fname = '{}_harmonized.xlsx'.format(prefix)
    print('Writing result to: {}'.format(fname))
    utils.pd_write(model, fname, sheet_name='data')

    # save data about harmonization
    fname = '{}_metadata.xlsx'.format(prefix)
    print('Writing metadata to: {}'.format(fname))
    utils.pd_write(metadata, fname)


if __name__ == '__main__':
    main()
