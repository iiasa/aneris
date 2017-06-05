"""Harmonization CLI for aneris.
"""
import argparse
import logging
import os

import aneris
from aneris.utils import hist_path, region_path, logger


def read_args():
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
    input_file = 'Input data file.'
    parser.add_argument('input_file', help=input_file)
    history = 'Historical emissions in the base year.'
    parser.add_argument('--history', help=history,
                        default=hist_path('history.csv'))
    regions = 'Mapping of country iso-codes to native regions.'
    parser.add_argument('--regions', help=regions,
                        default=region_path('message.csv'))
    rc = 'Runcontrol YAML file (see <WEBSITE> for examples).'
    parser.add_argument('--rc', help=rc, default=None)
    output_path = 'Path to use for output file names.'
    parser.add_argument('--output_path', help=output_path, default='.')
    output_prefix = 'Prefix to use for output file names.'
    parser.add_argument('--output_prefix', help=output_prefix, default=None)

    args = parser.parse_args()
    return args


def harmonize(inf, history, regions, rc, output_path, output_prefix):
    # read input
    hist = aneris.pd_read(history, str_cols=True)
    if hist.empty:
        raise ValueError('History file is empty')
    regions = aneris.pd_read(regions, str_cols=True)
    if regions.empty:
        raise ValueError('Region definition is empty')
    model, overrides, config = aneris.read_excel(inf)
    rc = aneris.RunControl(rc=rc)
    rc.recursive_update('config', config)

    # do core harmonization
    driver = aneris.HarmonizationDriver(rc, hist, model, overrides, regions)
    for scenario in driver.scenarios():
        driver.harmonize(scenario)
    model, metadata = driver.harmonized_results()

    # write to excel
    prefix = output_prefix or inf.split('.')[0]
    fname = os.path.join(output_path, '{}_harmonized.xlsx'.format(prefix))
    logger().info('Writing result to: {}'.format(fname))
    aneris.pd_write(model, fname, sheet_name='data')

    # save data about harmonization
    fname = os.path.join(output_path, '{}_metadata.xlsx'.format(prefix))
    logger().info('Writing metadata to: {}'.format(fname))
    aneris.pd_write(metadata, fname)


def main():
    # parse cli
    args = read_args()

    # run program
    harmonize(args.inf, args.history, args.regions,
              args.rc, args.output_path, args.output_prefix)


if __name__ == '__main__':
    main()
