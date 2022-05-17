import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
from tqdm import tqdm


# Script to identify sentences which mention immigrants
# The basic idea is to start with speeches that have been classified as having some content related to immigration.
# We then identify terms likely to refer directly to immigrants (e.g., "immigrants", "aliens", etc.)
# as well as terms that are likely to be such references when used with a gropu modifier (e.g., "Mexican workers").
# Some terms are converted from bigrams to unigrams to consolidate OCR approaches across two different datasets.


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--pred-dir', type=str, default='data/speeches/Congress/',
                      help='Input file from export.export_imm_speeches_parsed.py: default=%default')
    parser.add_option('--min-length', type=int, default=4,
                      help='Min length in tokens (per sentence): default=%default')
    parser.add_option('--prop', type=float, default=1.00,
                      help='Keep this percentage of generic mentions: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    pred_dir = options.pred_dir
    min_length = options.min_length
    prop_to_keep = options.prop
    seed = options.seed
    np.random.seed(seed)

    infile = os.path.join(pred_dir, 'imm_speeches_parsed.jsonlist')
    outfile = os.path.join(pred_dir, 'imm_mention_sents_parsed.jsonlist')
    countfile = os.path.join(pred_dir, 'imm_mention_sents_parsed_counts.json')
    prof_outfile = os.path.join(pred_dir, 'generic_mention_sents_parsed.jsonlist')
    prof_countfile = os.path.join(pred_dir, 'generic_mention_sents_parsed_counts.json')

    print("Building list of terms")
    initial_filter, target_terms, hyphenated_replacements, compound_replacements = create_target_terms_and_replacements()

    outlines = []
    generic_outlines = []

    term_counter = Counter()
    for term in basic_terms:
        term_counter[term] = 0
    
    generic_counter = Counter()
    for term in generic_terms:
        generic_counter[term] = 0

    print("Loading data")
    with open(infile) as f:
        lines = f.readlines()

    for line in tqdm(lines):
        line = json.loads(line)
        speech_id = line['id']
        tokens = line['tokens']
        lemmas = line['lemmas']
        tags = line['tags']

        for sent_i, orig_tokens in enumerate(tokens):

            # convert to lower case and do an initial filter
            lower_tokens = [t.lower() for t in orig_tokens]
            # combine the lower tokens into a string
            temp_string = ' '.join(lower_tokens)

            if len(set(lower_tokens).intersection(initial_filter)) > 0:
                # apply initial replacements  (e.g., "displaced persons" -> "displacedpersons")
                for s, r in initial_replacements.items():
                    if s in temp_string:
                        temp_string = re.sub(s, r, temp_string)
                # apply national replacements (e.g., "asian american" -> "asianamerican")
                for s, r in hyphenated_replacements.items():
                    if s in temp_string:
                        temp_string = re.sub(s, r, temp_string)
                # apply all other replacements (e.g., "mexican workers" -> "mexicanworkers"))
                for s, r in compound_replacements.items():
                    if s in temp_string:
                        temp_string = re.sub(s, r, temp_string)

                # re-split into tokens
                final_tokens = temp_string.split()

                # only keep sentences with at least one target term (and more than a min number of tokens)
                if len(final_tokens) >= min_length and len(set(final_tokens).intersection(target_terms)) > 0:
                    # Save the sentence with metadata
                    outlines.append({'id': speech_id, 'sent_index': sent_i, 'tokens': orig_tokens, 'lemmas': lemmas[sent_i], 'tags': tags[sent_i], 'simplified': temp_string})
                    # Keep track of which terms are found most frequently
                    term_counter.update(set(final_tokens).intersection(target_terms))

            # also keep a set of lines for professions
            if len(lower_tokens) >= min_length and len(set(lower_tokens).intersection(generic_terms)) > 0:
                if np.random.rand() < prop_to_keep:
                    # Save the sentence with metadata
                    generic_outlines.append({'id': speech_id, 'sent_index': sent_i, 'tokens': orig_tokens, 'lemmas': lemmas[sent_i], 'tags': tags[sent_i], 'simplified': temp_string})
                    # Keep track of which terms are found most frequently
                    generic_counter.update(set(lower_tokens).intersection(generic_terms))

    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    with open(countfile, 'w') as f:
        json.dump(term_counter.most_common(), f, indent=2, sort_keys=False)


    with open(prof_outfile, 'w') as f:
        for line in generic_outlines:
            f.write(json.dumps(line) + '\n')

    with open(prof_countfile, 'w') as f:
        json.dump(generic_counter.most_common(), f, indent=2, sort_keys=False)


# replacements (terms which may appear with or without a space or hyphen)
initial_replacements = {'asylum seeker': 'asylumseeker',
                        'displaced person': 'displacedperson',
                        'displaced people': 'displacedpeople',
                        'foreign born': 'foreignborn',
                        'latin american': 'latinamerican',
                        'central american': 'centralamerican',
                        'middle eastern': 'middleeastern',
                        }

# target mention: Count these as immigrants, whenever they appear
basic_terms = {'immigrants',
               'inmigrants',
               'imnigrants',
               'aliens',
               'emigrants',
               'migrants',
               'foreigner',
               'foreigners',
               'refugees',
               'asylees',
               'newcomer',
               'newcomers',
               'entrants',
               'arrivals',
               'coolies',
               'coolys',
               'emigres',
               'undesirables',
               'illegals',
               'exiles',
               'asylumseeker',
               'asylumseekers',
               'displacedperson',
               'displacedpersons',
               'displacedpeople',
               'displacedpeoples',
               'foreignborn'
               }


# Count these as immigrant mentions if preceded by a nationality or group name
post_group_terms = {'immigrant',
                    'inmigrant',
                    'imnigrant',
                    'immigrants',
                    'inmigrants',
                    'imnigrants',
                    'alien',
                    'aliens',
                    'emigrant',
                    'emigrants',
                    'migrant',
                    'migrants',
                    'foreigner',
                    'foreigners',
                    'refugee',
                    'refugees',
                    'asylee',
                    'asylees',
                    'newcomer',
                    'newcomers',
                    'entrant',
                    'entrants',
                    'arrival',
                    'arrivals',
                    'coolie',
                    'coolies',
                    'cooly',
                    'coolys',
                    'emigre',
                    'emigres',
                    'undesirables',
                    'illegals',
                    'exile',
                    'exiles',
                    'asylumseeker',
                    'asylumseekers',
                    'displacedperson',
                    'displacedpersons',
                    'displacedpeople',
                    'displacedpeoples',
                    'horde',
                    'hordes',
                    'emigre',
                    'emigres',
                    'stranger',
                    'strangers',
                    'national',
                    'nationals',
                    'worker',
                    'workers',
                    'workman',
                    'workmen',
                    'workingman',
                    'workingmen',
                    'laborer',
                    'laborers',
                    'farmworker',
                    'farmworkers',
                    'prostitute',
                    'prostitutes',
                    'artisan',
                    'artisans',
                    'student',
                    'students',
                    'people',
                    'person',
                    'persons',
                    'population',
                    'populations',
                    'citizen',
                    'citizens',
                    'settler',
                    'settlers',
                    'community',
                    'communities',
                    'voter',
                    'voters',
                    'resident',
                    'residents',
                    'man',
                    'men',
                    'male',
                    'males',
                    'woman',
                    'women',
                    'female',
                    'females',
                    'child',
                    'children',
                    'youth',
                    'youths',
                    'youngster',
                    'youngsters',
                    'teenager',
                    'teenagers',
                    'minor',
                    'minors',
                    'adult',
                    'adults',
                    'family',
                    'families',
                    'husband',
                    'husbands',
                    'wife',
                    'wives',
                    'bride',
                    'brides',
                    'couple',
                    'couples',
                    'parent',
                    'parents',
                    'mother',
                    'mothers',
                    'father',
                    'fathers',
                    }


# nationality / group adjectives, and ways of referring to those groups
# count the values as mentions if found
# use the keys as possible prefixes for post_group_terms
group_terms = {'foreign': [],
               'foreignborn': [],
               'european': ['europeans'],
               'scandinavian': ['scandinavians'],
               'asian': ['asians', 'asiatics'],
               'indochinese': [],
               'oriental': ['orientals'],
               'latinamerican': ['latinamericans'],
               'centralamerican': ['centralamericans'],
               'middleeastern': [],
               'arab': ['arabs'],
               'irish': ['irishman', 'irishmen', 'celts', 'celtics'],
               'german': ['germans'],
               'prussian': ['prussians'],
               'mexican': ['mexicans'],
               'wetback': ['wetbacks'],
               'bracero': ['braceros'],
               'italian': ['italians'],
               'sicilian': ['sicilians'],
               'english': ['englishman', 'englishmen'],
               'british': ['brits'],
               'canadian': ['canadians'],
               'russian': ['russians'],
               'polish': ['poles'],
               'chinese': ['chinaman', 'chinamen'],
               'mongolian': ['mongolians'],
               'celestial': ['celestials'],
               'swedish': ['swedes'],
               'austrian': ['austrians'],
               'filipino': ['filipinos', 'filipinas'],
               'cuban': ['cubans'],
               'hungarian': ['hungarians'],
               'norwegian': ['norwegians'],
               'czech': ['czechs'],
               'slovak': ['slvoaks'],
               'slovakian': ['slovakians'],
               'czechoslovakian': ['czechoslovakians'],
               'slavic': ['Slavs', 'slavics'],
               'bohemian': ['bohemians'],
               'vietnamese': [],
               'scottish': ['scotsman', 'scotsmen'],
               'salvadoran': ['salvadorans'],
               'salvadorian': ['salvadorians'],
               'korean': ['koreans'],
               'french': ['frenchman', 'frenchmen'],
               'dominican': ['dominicans'],
               'guatemalan': ['guatemalans'],
               'greek': ['greeks'],
               'colombian': ['colombians'],
               'jamaican': ['jamaicans'],
               'yugoslavian': ['yugoslavians'],
               'serbian': ['serbians'],
               'serb': ['serbs'],
               'croatian': ['croatians'],
               'croat': ['croats'],
               'macedonian': ['macedonians'],
               'bosnian': ['bosnians'],
               'honduran': ['hondurans'],
               'japanese': ['japs'],
               'haitian': ['haitians'],
               'portuguese': [],
               'danish': ['danes'],
               'lithuanian': ['lithuanians'],
               'swiss': [],
               'welsh': ['welshman', 'welshmen'],
               'taiwanese': [],
               'dutch': [],
               'brazilian': ['brazilians'],
               'finnish': ['finns'],
               'iranian': [ 'iranians'],
               'ecuadorian': ['ecuadorians'],
               'venezuelan': ['venezuelans'],
               'romanian': ['romanians'],
               'rumanian': ['rumanians'],
               'roumanian': ['roumanians'],
               'peruvian': ['peruvians'],
               'jewish': ['jews'],
               'catholic': ['catholics'],
               'mormon': ['mormons'],
               'muslim': ['muslims', 'moslems'],
               'hindu': ['hindus', 'hindoos'],
               'latino': ['latino', 'latinos', 'latina', 'latinas'],
               'hispanic': ['hispanic', 'hispanics']}


generic_terms = {'man', 'woman', 'men', 'women'}


def do_hyphenation(group_terms):
    # Deal with terms like "Asian American"
    hyphenated_group_terms = defaultdict(list)   # group and names, corresponding to group_terms
    hyphenated_replacements = {}                 # replacements to make for searching
    for group, terms in group_terms.items():
        if 'american' not in group:
            hyphenated_group_terms[group + 'american'].append(group + 'americans')
            hyphenated_replacements[group + ' american'] = group + 'american'
    return hyphenated_group_terms, hyphenated_replacements


def create_target_terms_and_replacements():

    # hyphenated terms to do replacements on
    hyphenated_group_terms, hyphenated_replacements = do_hyphenation(group_terms)

    # create an initial unigram filter which includes group names and adjectives, basic immigrant terms, and a few more
    initial_filter = {'latin', 'central', 'middle'}
    initial_filter.update(basic_terms)
    for group, terms in group_terms.items():
        initial_filter.add(group)
        initial_filter.update(terms)
    for group, terms in hyphenated_group_terms.items():
        initial_filter.add(group)
        initial_filter.update(terms)

    # make all combinations of adjectival group names with basic and other terms
    # these will define conversion of bigrams to single tokens
    compound_replacements = {}
    for group in group_terms:
        for post_term in post_group_terms:
            compound_replacements[group + ' ' + post_term] = group + post_term  # e.g., mexican workers -> mexicanworkers
    for group in hyphenated_group_terms:
        for post_term in post_group_terms:
            compound_replacements[group + ' ' + post_term] = group + post_term  # e.g., mexicanamerican workers -> mexicanamericanworkers

    # create the full list of target terms (post transformation)
    target_terms = set()
    # add basic terms (e.g., immigrants)
    target_terms.update(basic_terms)
    # add noun form group terms to target terms (e.g., mexicans)
    for group, terms in group_terms.items():
        target_terms.update(terms)
    # add plural hyphenated terms as target terms (e.g., mexicanamericans)
    for group, terms in hyphenated_group_terms.items():
        target_terms.update(terms)
    # add concatenated terms to target terms (e.g., mexicanworkers)
    target_terms.update(compound_replacements.values())

    return initial_filter, target_terms, hyphenated_replacements, compound_replacements


if __name__ == '__main__':
    main()
