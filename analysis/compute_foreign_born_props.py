import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--issue', type=str, default='immigration',
    #                  help='Issue: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    df = pd.read_csv(os.path.join('data', 'foreign_born_1850-2019.csv'), header=0, index_col=None)
    print(df.shape)

    years = [int(y) for y in df.columns[3:]]

    totals = [int(df.loc[1, str(y)]) for y in years]

    subset = df[df['Type'] == 'Country']
    countries = [str(c) for c in subset['Place'].values]

    values = subset[[str(y) for y in years]].values
    props = np.zeros_like(values)
    for i, year in enumerate(years):
        for row, country in enumerate(countries):
            if not np.isnan(values[row, i]):
                props[row, i] = values[row, i] / totals[i]

    max_props = props.max(axis=1)
    order = np.argsort(max_props)[::-1]
    for i in range(len(order)):
        index = order[i]
        argmax = np.argmax(props[index, :])
        prop = props[index, argmax]
        if prop >= 0.01:
            print('{:d} {:.3f} {:.0f} {:d} {:s}'.format(i, prop, values[index, argmax], years[argmax], countries[index]))

    df = pd.DataFrame(props, index=subset.index, columns=years)
    df['Country'] = countries
    df.to_csv(os.path.join('data', 'foreign_born_props_1850-2019.csv'))


if __name__ == '__main__':
    main()
