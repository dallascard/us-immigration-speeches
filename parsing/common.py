import re


def normalize_to_stanford(text):
    """
    Convert a string (e.g., from USCR to the style used in the Stanford/Gentzkow data)
    :param text: input str
    :return: normalized str
    """
    # remove apostrophes from common contractions
    text = re.sub(r"'s", 's', text)
    text = re.sub(r"'m", 'm', text)
    text = re.sub(r"'d", 'd', text)
    text = re.sub(r"'t", 't', text)
    text = re.sub(r"'ll", 'll', text)
    text = re.sub(r"'re", 're', text)
    text = re.sub(r"'ve", 've', text)
    # remove possessive s'
    text = re.sub(r"s'\s", 's ', text)
    # remove apostrophes from O' names and words (e.g., o'clock, O'Neil)
    text = re.sub(r"o'(?=[a-z])", 'o', text)
    text = re.sub(r"O'(?=[A-Z])", 'O', text)
    # remove hyphens from the middle of words
    text = re.sub(r'(?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])', '', text)
    return text
