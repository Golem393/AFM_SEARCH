"""File with Useful Utility Functions

List of functions:
    parse_keywords: Extracts list of strings from comma-seperated string of keywords

"""

def parse_keywords(keyword_string: str):
    """Extracts list of keyword strings from comma-seperated string of keywords

    args:
        keyword: str string with comma-separated keywords
    """
    return [keyword.strip() for keyword in keyword_string.split(',') if keyword.strip()]