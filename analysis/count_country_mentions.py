import os
import re
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from analysis.group_terms import get_countries, get_nationalities, add_american, get_regions_and_regionalities


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Congress/imm_segments_with_tone_and_metadata.jsonlist',
                      help='infile: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/country_mentions/',
                      help='outdir: default=%default')
    parser.add_option('--lower', action="store_true", default=False,
                      help='Lower case text: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outdir = options.outdir
    lower = options.lower

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    countries = get_countries()
    nationalities = get_nationalities()
    regions, regionalities = get_regions_and_regionalities()

    american_terms, substitutions = add_american(nationalities)
    american_regional_terms, regional_subs = add_american(regionalities)

    if lower:
        countries = {country: [t.lower() for t in terms] for country, terms in countries.items()}
        nationalities = {country: [t.lower() for t in terms] for country, terms in nationalities.items()}
        american_terms = {country: [t.lower() for t in terms] for country, terms in american_terms.items()}
        regions = {region: [t.lower() for t in terms] for region, terms in regions.items()}
        regionalities = {region: [t.lower() for t in terms] for region, terms in regionalities.items()}
        american_regional_terms = {region: [t.lower() for t in terms] for region, terms in american_regional_terms.items()}

    nationality_terms = {}
    for country, terms in nationalities.items():
        nationality_terms[country] = set(terms).union(american_terms[country])

    combined_terms = {}
    for country, terms in countries.items():
        combined_terms[country] = set(terms).union(nationalities[country]).union(american_terms[country])

    combined_regional_terms = {}
    for region, terms in regions.items():
        combined_regional_terms[region] = set(terms).union(regionalities[region]).union(american_regional_terms[region])

    country_counter_by_congress = defaultdict(Counter)
    nationality_counter_by_congress = defaultdict(Counter)
    combined_counter_by_congress = defaultdict(Counter)
    region_counter_by_congress = defaultdict(Counter)
    regionality_counter_by_congress = defaultdict(Counter)
    combined_regional_counter_by_congress = defaultdict(Counter)

    with open(infile) as f:
        lines = f.readlines()

    speech_ids_by_country = defaultdict(set)
    speech_ids_by_nationality = defaultdict(set)
    speech_ids_by_nationality_plus_country = defaultdict(set)
    speech_ids_by_region = defaultdict(set)
    speech_ids_by_regionality = defaultdict(set)
    speech_ids_by_region_or_regionality = defaultdict(set)

    for line in tqdm(lines):
        line = json.loads(line)
        speech_id = line['speech_id']
        congress = int(line['congress'])
        text = line['text']
        # remove spaces and dashes from <nationality>-American terms
        for query, replacement in substitutions.items():
            text = re.sub(query, replacement, text)
        if lower:
            text = text.lower()
        tokens = set(text.split())

        for country, terms in countries.items():
            if len(tokens.intersection(terms)) > 0:
                country_counter_by_congress[country][congress] += 1
                speech_ids_by_country[country].add(speech_id)
        for country, terms in nationality_terms.items():
            if len(tokens.intersection(terms)) > 0:
                nationality_counter_by_congress[country][congress] += 1
                speech_ids_by_nationality[country].add(speech_id)
        for country, terms in combined_terms.items():
            if len(tokens.intersection(terms)) > 0:
                combined_counter_by_congress[country][congress] += 1
                speech_ids_by_nationality_plus_country[country].add(speech_id)
        for region, terms in regions.items():
            if len(tokens.intersection(terms)) > 0:
                region_counter_by_congress[region][congress] += 1
                speech_ids_by_region[region].add(speech_id)
        for region, terms in regionalities.items():
            if len(tokens.intersection(terms)) > 0:
                regionality_counter_by_congress[region][congress] += 1
                speech_ids_by_regionality[region].add(speech_id)
        for region, terms in combined_regional_terms.items():
            if len(tokens.intersection(terms)) > 0:
                combined_regional_counter_by_congress[region][congress] += 1
                speech_ids_by_region_or_regionality[region].add(speech_id)

    country_list = sorted(country_counter_by_congress)
    sums = [sum(country_counter_by_congress[country].values()) for country in country_list]

    with open(os.path.join(outdir, 'imm_country_counts_country_mentions.json'), 'w') as f:
        json.dump(dict(zip(country_list, sums)), f, indent=2)

    region_list = sorted(region_counter_by_congress)
    region_sums = [sum(region_counter_by_congress[region].values()) for region in region_list]

    with open(os.path.join(outdir, 'imm_region_counts_region_mentions.json'), 'w') as f:
        json.dump(dict(zip(region_list, region_sums)), f, indent=2)

    country_list = sorted(nationality_counter_by_congress)
    sums = [sum(nationality_counter_by_congress[country].values()) for country in country_list]
    order = np.argsort(sums)[::-1]
    print("Number of mentions per nationality:")
    for i in order:
        print(country_list[i], sums[i])

    with open(os.path.join(outdir, 'imm_country_counts_nationality_mentions.json'), 'w') as f:
        json.dump(dict(zip(country_list, sums)), f, indent=2)

    region_list = sorted(regionality_counter_by_congress)
    region_sums = [sum(regionality_counter_by_congress[region].values()) for region in region_list]
    order = np.argsort(region_sums)[::-1]
    print("Number of mentions per regionality:")
    for i in order:
        print(region_list[i], region_sums[i])

    with open(os.path.join(outdir, 'imm_region_counts_regionality_mentions.json'), 'w') as f:
        json.dump(dict(zip(region_list, region_sums)), f, indent=2)

    country_list = sorted(combined_counter_by_congress)
    sums = [sum(combined_counter_by_congress[country].values()) for country in country_list]

    with open(os.path.join(outdir, 'imm_country_counts_nationality_or_country_mentions.json'), 'w') as f:
        json.dump(dict(zip(country_list, sums)), f, indent=2)

    with open(os.path.join(outdir, 'imm_country_counts_nationality_or_country_mentions_by_congress.json'), 'w') as f:
        json.dump(combined_counter_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'imm_country_speech_ids_by_country_mentions.json'), 'w') as f:
        json.dump({country: sorted(ids) for country, ids in speech_ids_by_country.items()}, f, indent=2)

    with open(os.path.join(outdir, 'imm_country_speech_ids_by_nationality_mentions.json'), 'w') as f:
        json.dump({country: sorted(ids) for country, ids in speech_ids_by_nationality.items()}, f, indent=2)

    with open(os.path.join(outdir, 'imm_country_speech_ids_by_nationality_or_country_mentions.json'), 'w') as f:
        json.dump({country: sorted(ids) for country, ids in speech_ids_by_nationality_plus_country.items()}, f, indent=2)


    region_list = sorted(combined_regional_counter_by_congress)
    sums = [sum(combined_regional_counter_by_congress[region].values()) for region in region_list]

    with open(os.path.join(outdir, 'imm_region_counts_regionality_or_region_mentions.json'), 'w') as f:
        json.dump(dict(zip(region_list, sums)), f, indent=2)

    with open(os.path.join(outdir, 'imm_region_counts_regionality_or_region_by_congress.json'), 'w') as f:
        json.dump(combined_regional_counter_by_congress, f, indent=2)

    with open(os.path.join(outdir, 'imm_region_speech_ids_by_region_mentions.json'), 'w') as f:
        json.dump({region: sorted(ids) for region, ids in speech_ids_by_region.items()}, f, indent=2)

    with open(os.path.join(outdir, 'imm_region_speech_ids_by_regionality_mentions.json'), 'w') as f:
        json.dump({region: sorted(ids) for region, ids in speech_ids_by_regionality.items()}, f, indent=2)

    with open(os.path.join(outdir, 'imm_region_speech_ids_by_regionality_or_region_mentions.json'), 'w') as f:
        json.dump({region: sorted(ids) for region, ids in speech_ids_by_region_or_regionality.items()}, f, indent=2)



if __name__ == '__main__':
    main()
