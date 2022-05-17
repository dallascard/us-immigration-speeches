import os
import re
import datetime as dt
from collections import defaultdict, Counter

from tqdm import tqdm


def get_white_house_party(date):
    if date < dt.date(1869, 3, 4):
        raise ValueError("Party not available for date", date)
    elif date < dt.date(1877, 3, 4):
        # US Grant
        return 'R'
    elif date < dt.date(1881, 3, 4):
        # RB Hayes
        return 'R'
    elif date < dt.date(1881, 9, 19):
        # JA Garfield
        return 'R'
    elif date < dt.date(1885, 3, 4):
        # CA Arthur
        return 'R'
    elif date < dt.date(1889, 3, 4):
        # G Cleveland
        return 'D'
    elif date < dt.date(1893, 3, 4):
        # B Harrison
        return 'R'
    elif date < dt.date(1897, 3, 4):
        # G Cleveland
        return 'D'
    elif date < dt.date(1901, 9, 14):
        # W McKinley
        return 'R'
    elif date < dt.date(1909, 3, 4):
        # T Roosevelt
        return 'R'
    elif date < dt.date(1913, 3, 4):
        # WH Taft
        return 'R'
    elif date < dt.date(1921, 3, 4):
        # W Wilson
        return 'D'
    elif date < dt.date(1923, 8, 2):
        # WG Harding
        return 'R'
    elif date < dt.date(1929, 3, 4):
        # C Coolidge
        return 'R'
    elif date < dt.date(1933, 3, 4):
        # H Hoover
        return 'R'
    elif date < dt.date(1945, 4, 12):
        # FDR
        return 'D'
    elif date < dt.date(1953, 1, 20):
        # Truman
        return 'D'
    elif date < dt.date(1961, 1, 20):
        # Eisenhower
        return 'R'
    elif date < dt.date(1963, 11, 22):
        # JFK
        return 'D'
    elif date < dt.date(1969, 1, 20):
        # LBJ
        return 'D'
    elif date < dt.date(1974, 8, 9):
        # Nixon
        return 'R'
    elif date < dt.date(1977, 1, 20):
        # Ford
        return 'R'
    elif date < dt.date(1981, 1, 20):
        # Carter
        return 'D'
    elif date < dt.date(1989, 1, 20):
        # Reagan
        return 'R'
    elif date < dt.date(1993, 1, 20):
        # Bush 1
        return 'R'
    elif date < dt.date(2001, 1 ,20):
        # Clinton
        return 'D'
    elif date < dt.date(2009, 1, 20):
        # Bush 2
        return 'R'
    elif date < dt.date(2017, 1, 20):
        # Obama
        return 'D'
    else:
        raise ValueError("Party not available for date", date)


def year_to_period(year):
    if year < 1882:
        return '1873-1881'
    elif year < 1891:
        return '1882-1890'
    elif year < 1903:
        return '1891-1902'
    elif year < 1913:
        return '1903-1912'
    elif year < 1920:
        return '1913-1919'
    elif year < 1925:
        return '1920-1924'
    elif year < 1932:
        return '1925-1931'
    elif year < 1965:
        return '1932-1965'
    else:
        return '1965+'

def load_metadata(hein_bound_dir, first=43, last=71, hein_daily_dir=None, use_executive_party=True):

    # load metadata
    print("Loading metadata")
    metadata = {}
    metadata_counters = defaultdict(Counter)
    for congress in tqdm(range(first, last+1)):
        if congress < 112:
            indir = hein_bound_dir
        else:
            indir = hein_daily_dir
        speaker_file = os.path.join(indir, str(congress).zfill(3) + '_SpeakerMap.txt')
        speaker_info = {}
        with open(speaker_file, encoding='Windows-1252') as f:
            for line_i, line in enumerate(f):
                if line_i > 0:
                    parts = line.strip().split('|')
                    try:
                        speakerid, speech_id, lastname, firstname, chamber, state, gender, party, district, nonvoting = parts
                    except Exception as e:
                        print(parts)
                        raise e
                    if lastname == '':
                        lastname = 'None'
                    if firstname == '':
                        firstname = 'None'
                    if chamber == '':
                        chamber = 'None'
                    if state == '':
                        state = 'None'
                    if gender == '':
                        gender = 'None'
                    if party == '':
                        party = 'None'
                    if district == '':
                        district = 'None'
                    if nonvoting == '':
                        nonvoting = 'None'
                    speaker_info[speech_id] = [speakerid, lastname, firstname, chamber, state, gender, party, district, nonvoting]

        infile = os.path.join(indir, 'descr_' + str(congress).zfill(3) + '.txt')
        with open(infile, encoding='Windows-1252') as f:
            for line_i, line in enumerate(f):
                if line_i > 0:
                    parts = line.strip().split('|')
                    speech_id, chamber, date, _, speaker, _, lastname, state, gender, _, _, _, _, _ = parts
                    year = int(date[:4])
                    month = int(date[4:6])
                    day = int(date[6:])
                    if chamber == '':
                        chamber = 'None'
                    if state == '':
                        state = 'None'
                    if speaker == '':
                        speaker = 'None'
                    if lastname == '':
                        lastname = 'None'
                    if gender == '':
                        gender = 'None'

                    # clean up speaker
                    if speaker.startswith('The') or speaker.startswith('Tie'):
                        speaker = 'The ' + speaker[3:].strip()
                    elif speaker.startswith('Tile'):
                        speaker = 'The ' + speaker[4:].strip()

                    # fix possible missing space before "pro tempore"
                    if speaker.endswith('pro tempore'):
                        speaker = speaker[:-11].strip() + ' pro tempore'

                    speaker = re.sub(r'-', ' ', speaker)

                    # look for author info for this speech
                    if speech_id in speaker_info:
                        _, lastname2, firstname, chamber2, state2, gender2, party, district, nonvoting = speaker_info[speech_id]
                        if lastname2 == '':
                            lastname2 = 'None'
                        if firstname == '':
                            firstname = 'None'
                        if chamber2 == '':
                            chamber2 = 'None'
                        if state2 == '':
                            state2 = 'None'
                        if gender2 == '':
                            gender2 = 'None'
                        if party == '':
                            party = 'None'
                        if district == '':
                            district = 'None'
                        if nonvoting == '':
                            nonvoting = 'None'
                    else:
                        lastname2 = 'None'
                        firstname = 'None'
                        chamber2 = 'None'
                        state2 = 'None'
                        gender2 = 'None'
                        party = 'None'
                        district = 'None'
                        nonvoting = 'None'

                    # Use the chamber from the speech description file, unless None, then use speaker's chamber
                    if chamber == 'None':
                        chamber = chamber2

                    # if there is no lastname in the author file, this is usually an officer
                    if lastname2 == 'None':
                        name = speaker
                        # look for the president and vice president and infer their party
                        if name == 'The PRESIDENT' or name == 'The VICE PRESIDENT':
                            if use_executive_party:
                                inferred_party = get_white_house_party(dt.date(year, month, day))
                            else:
                                inferred_party = 'Executive'
                            best_state = 'None'

                        elif name.startswith('The'):
                            inferred_party = 'None'
                            best_state = 'None'

                        else:
                            inferred_party = party
                            best_state = state

                    else:
                        name = lastname2 + ', ' + firstname
                        inferred_party = party
                        best_state = state2


                    metadata[speech_id] = [year, month, day, chamber, name, speaker, lastname, lastname2, firstname, party, inferred_party, gender, gender2, best_state, state, state2, district, nonvoting]

    return metadata

