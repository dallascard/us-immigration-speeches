import os
import re
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm


def main():
    usage = "%prog uscr_raw_dir outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--first-year', type=int, default=1995,
                      help='First year: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    basedir = args[0]
    outdir = args[1]

    first_year = options.first_year

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    months = {'January': 1,
              'February': 2,
              'March': 3,
              'April': 4,
              'May': 5,
              'June': 6,
              'July': 7,
              'August': 8,
              'September': 9,
              'October': 10,
              'November': 11,
              'December': 12}

    chamber_counter = Counter()
    month_counter = Counter()
    year_counter = Counter()
    day_counter = Counter()
    len_counter = Counter()
    kind_counter = Counter()

    outlines_by_congress = defaultdict(list)

    for year in (range(first_year, 2021)):
        files = sorted(glob(os.path.join(basedir, str(year), '*', 'json', '*.json')))
        print(year, len(files))
        for infile in tqdm(files):
            with open(infile) as f:
                data = json.load(f)
                page_id = data['id']
                header = data['header']
                chamber = header['chamber']
                chamber_counter[chamber] += 1
                year = int(header['year'])
                year_counter[year] += 1
                month = months[header['month']]
                month_counter[month] += 1
                day = int(header['day'])
                day_counter[day] += 1
                extension = header['extension']
                content = data['content']
                len_counter[len(content)] += 1
                congress = date_to_congress(year, month, day)
                for e_i, entry in enumerate(content):
                    kind = entry['kind']
                    kind_counter[kind] += 1
                    # drop title, recorder, line breaks, and other
                    if kind == 'speech':
                        if 'speaker' in entry and 'speaker_bioguide' in entry and 'text' in entry:
                            speaker = entry['speaker']
                            if speaker == 'None':
                                speaker = None

                            bioguide = entry['speaker_bioguide']
                            text = entry['text'].strip()
                            # standardize spacing
                            text = re.sub('\s+', ' ', text).strip()
                            # only keep those lines with an identified speaker (name or bioguide)
                            if speaker is not None or bioguide is not None:
                                outlines_by_congress[congress].append({'id': page_id + '_' + str(e_i).zfill(4),
                                                 'congress': congress,
                                                 'year': year,
                                                 'month': month,
                                                 'day': day,
                                                 'date': str(year) + str(month).zfill(2) + str(day).zfill(2),
                                                 'chamber': chamber,
                                                 'ext': extension,
                                                 'speaker': speaker,
                                                 'bioguide': bioguide,
                                                 'text': text})

    for congress in sorted(outlines_by_congress):
        outlines = outlines_by_congress[congress]
        print(congress, len(outlines))
        with open(os.path.join(outdir, 'speeches_' + str(congress).zfill(3) + '.jsonlist'), 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')

    print("Chamber")
    for k, c in chamber_counter.most_common():
        print(k, c)

    print("Year")
    for k, c in year_counter.most_common():
        print(k, c)

    print("Month")
    for k, c in month_counter.most_common():
        print(k, c)

    print("Day")
    for k, c in day_counter.most_common():
        print(k, c)

    print("Kinds")
    for k, c in kind_counter.most_common():
        print(k, c)

    print(sorted(len_counter)[-4:])


def date_to_congress(year, month, day):
    # just assume all sessions start on January 3
    if month > 1 or day > 2:
        congress = (year - 2013) // 2 + 113
    else:
        congress = (year - 1 - 2013) // 2 + 113
    return congress


if __name__ == '__main__':
    main()
