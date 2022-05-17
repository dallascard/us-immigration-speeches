import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pystan


party_region_chamber_nodata_model = """
data { 
  int<lower=1> n_years;
  int<lower=1> n_segments;
  int<lower=1, upper=n_years> years[n_segments];
  vector[n_segments] dems;
  vector[n_segments] reps;
  vector[n_segments] north;
  vector[n_segments] south;
  vector[n_segments] west;
  vector[n_segments] nodata;
  vector[n_segments] senate;
  real tones[n_segments];
}
parameters {
  real<lower=0> tone_std;
  real<lower=0> year_std;    
  vector[n_years] overall_means;
  real<lower=0> party_std;
  //real<lower=0> dem_std;
  //real<lower=0> rep_std;
  //real dem_bias;
  //real rep_bias; 
  vector[n_years] dem_offsets;
  vector[n_years] rep_offsets;
  //real<lower=0> nodata_std;
  //real nodata_bias;
  vector[n_years] nodata_offsets;   
  //real<lower=0> north_std;
  //real<lower=0> south_std;
  //real<lower=0> west_std;
  real<lower=0> region_std;
  //real north_bias;
  //real south_bias;
  //real west_bias;
  vector[n_years] north_offsets;
  vector[n_years] south_offsets;
  vector[n_years] west_offsets;
  real<lower=0> senate_std;
  vector[n_years] senate_offsets;    
}
model {
  // Priors
  tone_std ~ normal(0, 2);
  year_std ~ normal(0, 2);  
  overall_means ~ normal(0, year_std);
  
  party_std ~ normal(0, 1);
  //dem_std ~ normal(0, 1);
  //rep_std ~ normal(0, 1); 
  //dem_bias ~ normal(0, 2);
  //rep_bias ~ normal(0, 2);
  dem_offsets ~ normal(0, party_std);
  rep_offsets ~ normal(0, party_std);
  
  //nodata_std ~ normal(0, 1);
  //nodata_bias ~ normal(0, 2);
  nodata_offsets ~ normal(0, party_std);

  region_std ~ normal(0, 1);
  //north_std ~ normal(0, 1);
  //south_std ~ normal(0, 1);
  //west_std ~ normal(0, 1);  
  //north_bias ~ normal(0, region_std);
  //south_bias ~ normal(0, region_std);
  //west_bias ~ normal(0, region_std);
  north_offsets ~ normal(0, region_std);
  south_offsets ~ normal(0, region_std);
  west_offsets ~ normal(0, region_std);

  senate_std ~ normal(0, 1);
  senate_offsets ~ normal(0, senate_std);

  //tones ~ normal(overall_means[years] + dems .* dem_offsets[years] + reps .* rep_offsets[years] + senate .* senate_offsets[years] + north .* north_offsets[years] + south .* south_offsets[years] + west .* west_offsets[years], tone_std);
  tones ~ normal(overall_means[years] + dems .* dem_offsets[years] + reps .* rep_offsets[years] + nodata .* nodata_offsets[years] + north .* north_offsets[years] + south .* south_offsets[years] + west .* west_offsets[years] + senate .* senate_offsets[years], tone_std);
}

"""


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='data/speeches/Congress/imm_segments_with_tone_and_metadata.jsonlist',
                      help='Input file: default=%default')
    parser.add_option('--outdir', type=str, default='data/speeches/Congress/stan/',
                      help='Output dir: default=%default')
    parser.add_option('--subsample', type=float, default=1.0,
                      help='Use a random subset of data: default=%default')
    parser.add_option('--iter', type=int, default=5000,
                      help='Number of sampling iterations: default=%default')
    parser.add_option('--chains', type=int, default=5,
                      help='Number of sampling chains: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outdir = options.outdir
    subsample = options.subsample
    np.random.seed(options.seed)
    exclude_nodata = True

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    northeast_states = {'ME', 'MA', 'RI', 'CT', 'NH', 'VT', 'NY', 'PA', 'NJ', 'DE'}
    southeast_states = {'WV', 'VI', 'VA', 'KY', 'TN', 'NC', 'SC', 'GA', 'AL', 'MS', 'AR', 'LA', 'FL'}
    midwest_states = {'OH', 'IN', 'MI', 'IL', 'MO', 'WI', 'MN', 'IA', 'KS', 'NE', 'SD', 'ND', 'DK'}
    southwest_states = {'TX', 'OK', 'NM', 'AZ'}
    west_states = {'CO', 'WY', 'MT', 'ID', 'WA', 'OR', 'UT', 'NV', 'CA', 'AK', 'HI'}

    def state_to_region(state):
        if state in northeast_states:
            return 'Northeast+Midwest'
        elif state in southeast_states:
            return 'Southeast+Southwest'
        elif state in midwest_states:
            return 'Northeast+Midwest'
        elif state in southwest_states:
            return 'Southeast+Southwest'
        elif state in west_states:
            return 'West'
        else:
            return 'Other'

    print("Loading data")
    with open(infile) as f:
        lines = f.readlines()
    print("Converting to json")
    segments = [json.loads(line) for line in lines]

    print(len(segments))
    if exclude_nodata:
        print("Excluding segments without party")
        segments = [line for line in segments if line['party'] == 'D' or line['party'] == 'R']
        print(len(segments))

    print("Extracting values")
    dates = [line['date'] for line in segments]
    years = [int(date[:4]) for date in dates]

    states = [line['state'] for line in segments]
    regions = [state_to_region(s) for s in states]
    chambers = [line['chamber'] for line in segments]
    parties = [line['party'] for line in segments]
    tones = [line['pro'] - line['anti'] for line in segments]

    senate = [1. if c == 'S' else 0. for c in chambers]
    house = [1. if c == 'H' else 0. for c in chambers]
    north = [1. if r == 'Northeast+Midwest' else 0. for r in regions]
    south = [1. if r == 'Southeast+Southwest' else 0. for r in regions]
    west = [1. if r == 'West' else 0. for r in regions]
    nodata = [1. if r == 'Other' else 0. for r in regions]

    print("Making party vector")
    dems = []
    reps = []
    for p in parties:
        if p == 'D':
            dems.append(1.)
            reps.append(0.)
        elif p == 'R':
            dems.append(0.)
            reps.append(1.)
        else:
            dems.append(0.)
            reps.append(0.)

    print("Making year vector")
    first_year = min(set(years))
    # convert to congressional sessions (approximately)
    year_vals = [1+int(y - first_year)//2 for y in years]
    n_years = max(year_vals)

    n_segments = len(segments)
    print("{:d} segments".format(n_segments))

    if subsample < 1:
        n_segments = int(n_segments * subsample)
        print("Taking a subsample of {:d} semgents".format(n_segments))
        indices = np.random.choice(np.arange(len(segments)), size=n_segments, replace=False)
        year_vals = [year_vals[i] for i in indices]
        tones = [tones[i] for i in indices]
        dems = [dems[i] for i in indices]
        reps = [reps[i] for i in indices]
        senate = [senate[i] for i in indices]
        house = [house[i] for i in indices]
        north = [north[i] for i in indices]
        south = [south[i] for i in indices]
        west = [west[i] for i in indices]
        nodata = [nodata[i] for i in indices]

    data = {'n_years': n_years,
            'n_segments': n_segments,
            'years': year_vals,
            'dems': dems,
            'reps': reps,
            'nodata': nodata,
            'senate': senate,
            'north': north,
            'south': south,
            'west': west,
            'tones': tones}

    print("Making model")
    sm = pystan.StanModel(model_code=party_region_chamber_nodata_model)
    print("Fitting model")
    fit = sm.sampling(data=data, iter=options.iter, chains=options.chains)

    print("Saving data")
    overall_means = fit.extract('overall_means')['overall_means']
    tone_std = fit.extract('tone_std')['tone_std']
    year_std = fit.extract('year_std')['year_std']
    party_std = fit.extract('party_std')['party_std']
    dem_offsets = fit.extract('dem_offsets')['dem_offsets']
    rep_offsets = fit.extract('rep_offsets')['rep_offsets']
    nodata_offsets = fit.extract('nodata_offsets')['nodata_offsets']
    senate_std = fit.extract('senate_std')['senate_std']
    senate_offsets = fit.extract('senate_offsets')['senate_offsets']
    region_std = fit.extract('region_std')['region_std']
    north_offsets = fit.extract('north_offsets')['north_offsets']
    south_offsets = fit.extract('south_offsets')['south_offsets']
    west_offsets = fit.extract('west_offsets')['west_offsets']

    file_prefix = 'party_region_nodata_model'
    if subsample < 1.0:
        file_prefix += '_' + str(subsample)
    filename = file_prefix + '.npz'

    np.savez(os.path.join(outdir, filename),
             tone_std=tone_std,
             year_std=year_std,
             overall_means=overall_means,
             party_std=party_std,
             dem_offsets=dem_offsets,
             rep_offsets=rep_offsets,
             nodata_offsets=nodata_offsets,
             region_std = region_std,
             north_offsets=north_offsets,
             south_offsets=south_offsets,
             west_offsets=west_offsets,
             senate_std=senate_std,
             senate_offsets=senate_offsets
             )


if __name__ == '__main__':
    main()
