import re
import string

replace_punct = re.compile('[%s]' % re.escape(string.punctuation))


def simplify_text(text):
    """
    Simplify text by lower casing and removing all punctuation and excess spaces
    :param text: a string (a congressional speech)
    :return: a cleaned up string
    """
    # drop all punctuation
    text = replace_punct.sub('', text)
    # lower case the text
    text = text.strip().lower()
    # convert all white space spans to single spaces
    text = re.sub(r'\s+', ' ', text)
    return text

