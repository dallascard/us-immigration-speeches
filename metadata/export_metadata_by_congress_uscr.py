import os
import re
import json
import datetime as dt
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm


def main():
    usage = "%prog outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--uscr-dir', type=str, default='data/speeches/Congress/uscr/',
                      help='Issue: default=%default')
    parser.add_option('--people-file', type=str, default='data/speeches/Congress/uscr-legistlators/legislators-all.json',
                      help='Issue: default=%default')
    parser.add_option('--first', type=int, default=104,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=116,
                      help='Last congress: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    outdir = args[0]

    uscr_dir = options.uscr_dir
    people_file = options.people_file
    first = options.first
    last = options.last

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(people_file) as f:
        bioguide = json.load(f)

    bioguide = {person['id']['bioguide']: person for person in bioguide}

    chamber_counter = Counter()
    speaker_counter = Counter()
    gender_counter = Counter()
    party_counter = Counter()
    state_counter = Counter()
    not_found = []
    unknown_speaker_counter = Counter()

    for congress in range(first, last+1):

        bioguide_ids_found = set()
        bioguide_ids_not_found = set()
        terms_found = 0
        terms_not_found = 0
        infile = os.path.join(uscr_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist')

        # then load the speech descriptions (less reliable but more common)
        speech_desc = {}
        with open(infile) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            speech_id = line['id']
            year = line['year']
            month = line['month']
            day = line['day']
            date = dt.date(year=year, month=month, day=day)
            datestring = line['date']
            chamber = line['chamber'][0]
            speaker = line['speaker']
            last_name = 'Unknown'
            first_name = 'Unknown'
            party = 'Unknown'
            state = 'Unknown'
            gender = 'Unknown'
            bioguide_id = line['bioguide']
            if bioguide_id == 'None':
                bioguide_id = 'Unknown'

            if bioguide_id in bioguide:
                bioguide_ids_found.add(bioguide_id)
                person = bioguide[bioguide_id]
                last_name = person['name']['last']
                first_name = person['name']['first']
                gender = person['bio']['gender']
                terms = person['terms']
                for term in terms:
                    start_year, start_month, start_day = [int(k) for k in term['start'].split('-')]
                    end_year, end_month, end_day = [int(k) for k in term['end'].split('-')]
                    start_date = dt.date(year=start_year, month=start_month, day=start_day)
                    end_date = dt.date(year=end_year, month=end_month, day=end_day)
                    if start_date <= date <= end_date:
                        party = term['party'][0]
                        state = term['state']
                if party == 'Unknown':
                    terms_not_found += 1
                    not_found.append({'congress': congress, 'date': datestring, 'year': year, 'bioguide_id': bioguide_id, 'speaker': speaker, 'chamber': chamber})
                else:
                    terms_found += 1
            else:
                unknown_speaker_counter[speaker] += 1
                bioguide_ids_not_found.add(bioguide_id)

            chamber_counter[chamber] += 1
            speaker_counter[speaker] += 1
            gender_counter[gender] += 1
            party_counter[party] += 1
            state_counter[state] += 1

            speech_desc[speech_id] = {'date': datestring,
                                      'year': year,
                                      'month': month,
                                      'day': day,
                                      'speaker_id': bioguide_id,
                                      'speaker': speaker,
                                      'last_name': last_name,
                                      'first_name': first_name,
                                      'chamber': chamber,
                                      'gender': gender,
                                      'state': state,
                                      'party': party,
                                      }

        print(congress, len(speech_desc), len(bioguide_ids_found), len(bioguide_ids_not_found), terms_found, terms_not_found)
        # save the combined metadata
        with open(os.path.join(outdir, 'uscr_metadata_' + str(congress).zfill(3) + '.json'), 'w') as f:
            json.dump(speech_desc, f, indent=2, sort_keys=True)

        if congress == last:
            print("bioguide IDs not found:")
            print(bioguide_ids_not_found)

    with open(os.path.join(outdir, 'unknown_speaker_counter.json'), 'w') as f:
        json.dump(unknown_speaker_counter.most_common(), f, indent=2, sort_keys=False)



if __name__ == '__main__':
    main()


