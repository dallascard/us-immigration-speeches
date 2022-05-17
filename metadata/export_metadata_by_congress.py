import os
import re
import json
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm


def main():
    usage = "%prog outdir"
    parser = OptionParser(usage=usage)
    parser.add_option('--hein-bound-dir', type=str, default='data/speeches/Congress/hein-bound/',
                      help='Issue: default=%default')
    parser.add_option('--hein-daily-dir', type=str, default='data/speeches/Congress/hein-daily/',
                      help='Issue: default=%default')
    parser.add_option('--first', type=int, default=43,
                      help='First congress: default=%default')
    parser.add_option('--last', type=int, default=114,
                      help='Last congress: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    outdir = args[0]

    hein_bound_dir = options.hein_bound_dir
    hein_daily_dir = options.hein_daily_dir
    first = options.first
    last = options.last

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for congress in range(first, last+1):
        if congress < 112:
            indir = hein_bound_dir
        else:
            indir = hein_daily_dir

        infile = os.path.join(indir, 'speeches_' + str(congress).zfill(3) + '.txt')
        descr_file = os.path.join(indir, 'descr_' + str(congress).zfill(3) + '.txt')
        speaker_file = os.path.join(indir, str(congress).zfill(3) + '_SpeakerMap.txt')
        print(speaker_file)
        basename = os.path.splitext(os.path.basename(infile))[0]

        # first load the speaker info for those speeches for which it is available
        speaker_info = {}
        with open(speaker_file, encoding='Windows-1252') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split('|')
            speaker_id, speech_id, last_name, first_name, chamber, state, gender, party, district, voting = parts
            if speaker_id != 'speaker_id' and speaker_id != 'speakerid':
                try:
                    assert len(last_name) > 0
                    assert len(first_name) > 0
                    assert len(chamber) > 0
                    assert len(state) > 0
                    assert len(gender) > 0
                    assert len(speaker_id) > 0
                except AssertionError as e:
                    print(line)
                    raise e
                if party == '':
                    party = 'Unknown'

                speaker_info[speech_id] = {'last': last_name,
                                           'first': first_name,
                                           'chamber': chamber,
                                           'state': state,
                                           'gender': gender,
                                           'party': party,
                                           'speaker_id': speaker_id}

        # then load the speech descriptions (less reliable but more common)
        speech_desc = {}
        excluded_speech_ids = set()
        with open(descr_file, encoding='Windows-1252') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split('|')
            speech_id, chamber, date, line_num, speaker, first_name, last_name, state, gender, _, _, _, _, _ = parts
            # exclude a possible header row and one day with corrupted data
            if speech_id != 'speech_id':
                if date == '18940614':
                    excluded_speech_ids.add(speech_id)
                else:
                    # clean up common OCR errors for speaker
                    if speaker.startswith('Tile'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tlie'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Time'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tihe'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tire'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Thie'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tite'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tlhe'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tihe'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tise'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tlfe'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tlle'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tfhe'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tiie'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Thle'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tlte'):
                        speaker = 'The ' + speaker[4:].strip()
                    elif speaker.startswith('Tle'):
                        speaker = 'The ' + speaker[3:].strip()
                    elif speaker.startswith('Tme'):
                        speaker = 'The ' + speaker[3:].strip()
                    elif speaker.startswith('Tne'):
                        speaker = 'The ' + speaker[3:].strip()
                    elif speaker.startswith('Tie'):
                        speaker = 'The ' + speaker[3:].strip()
                    elif speaker.startswith('Tue'):
                        speaker = 'The ' + speaker[3:].strip()
                    elif speaker.startswith('Tte'):
                        speaker = 'The ' + speaker[3:].strip()
                    elif speaker.startswith('Tse'):
                        speaker = 'The ' + speaker[3:].strip()

                    # normalize space
                    speaker = ' '.join(speaker.strip().split())

                    parts = speaker.split()
                    if len(parts) > 1 and parts[1].startswith('VICE'):
                        speaker = 'The VICE PRESIDENT'

                    # fix possible missing space before "pro tempore"
                    if speaker.endswith('pro tempore'):
                        speaker = speaker[:-11].strip() + ' pro tempore'
                    elif speaker.endswith('protempore'):
                        speaker = speaker[:-10].strip() + ' pro tempore'

                    # seems like all (or nearly all) of these are parsing errors
                    if speaker == 'The PRESIDENT' or speaker == 'The ACTING PRESIDENT':
                        speaker += ' pro tempore'

                    speaker = re.sub(r'-', ' ', speaker)

                    if chamber == 'None':
                        chamber = 'Unknown'

                    speech_desc[speech_id] = {'date': date,
                                              'year': int(date[:4]),
                                              'month': int(date[4:6]),
                                              'day': int(date[6:8]),
                                              'speaker_id': 'Unknown',
                                              'speaker': speaker,
                                              'last_name': 'Unknown',
                                              'first_name': 'Unknown',
                                              'chamber': chamber,
                                              'gender': gender,
                                              'state': 'Unknown',
                                              'party': 'Unknown',
                                              }

                    if speech_id in speaker_info:
                        # overwrite these attributes if available
                        speech_desc[speech_id]['gender'] = speaker_info[speech_id]['gender']
                        speech_desc[speech_id]['chamber'] = speaker_info[speech_id]['chamber']
                        speech_desc[speech_id]['last_name'] = speaker_info[speech_id]['last']
                        speech_desc[speech_id]['first_name'] = speaker_info[speech_id]['first']
                        speech_desc[speech_id]['state'] = speaker_info[speech_id]['state']
                        speech_desc[speech_id]['party'] = speaker_info[speech_id]['party']
                        speech_desc[speech_id]['speaker_id'] = speaker_info[speech_id]['speaker_id']

        # save the combined metadata
        with open(os.path.join(outdir, 'metadata_' + str(congress).zfill(3) + '.json'), 'w') as f:
            json.dump(speech_desc, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()


