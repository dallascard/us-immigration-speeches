periods = [(1869, 1877, 'R', 'Ulysses S. Grant', 'Grant'),
           (1877, 1881, 'R', 'Rutherford B. Hayes', 'Hayes'),
           (1881, 1885, 'R', 'Chester A. Arthur', 'Arther'),
           (1885, 1889, 'D', 'Grover Cleveland (85-89)', 'Cleveland'),
           (1889, 1893, 'R', 'Benjamin Harrison', 'Harrison'),
           (1893, 1897, 'D', 'Grover Cleveland (93-97)', 'Cleveland'),
           (1897, 1902, 'R', 'William McKinley', 'McKinley'),
           (1902, 1909, 'R', 'Theodore Roosevelt', 'Roosevelt'),
           (1909, 1913, 'R', 'William Howard Taft', 'Taft'),
           (1913, 1921, 'D', 'Woodrow Wilson', 'Wilson'),
           (1921, 1924, 'R', 'Warren G. Harding', 'Harding'),
           (1924, 1929, 'R', 'Calvin Coolidge', 'Coolidge'),
           (1929, 1933, 'R', 'Herbert Hoover', 'Hoover'),
           (1933, 1945, 'D', 'Franklin D. Roosevelt', 'FDR'),
           (1945, 1953, 'D', 'Harry S. Truman', 'Truman'),
           (1953, 1961, 'R', 'Dwight D. Eisenhower', 'Eisenhower'),
           (1961, 1964, 'D', 'John F. Kennedy', 'Kennedy'),
           (1964, 1969, 'D', 'Lyndon B. Johnson', 'Johnson'),
           (1969, 1975, 'R', 'Richard Nixon', 'Nixon'),
           (1975, 1977, 'R', 'Gerald R. Ford', 'Ford'),
           (1977, 1981, 'D', 'Jimmy Carter', 'Carter'),
           (1981, 1989, 'R', 'Ronald Reagan', 'Reagan'),
           (1989, 1993, 'R', 'George Bush', 'GWH Bush'),
           (1993, 2001, 'D', 'William J. Clinton', 'Clinton'),
           (2001, 2009, 'R', 'George W. Bush', 'GW Bush'),
           (2009, 2017, 'D', 'Barack Obama', 'Obama'),
           (2017, 2021, 'R', 'Donald J. Trump', 'Trump')]


def year_to_congress(year):
    # return the congressional session associated with each year (pretending a Jan 1st start)
    return (year-1873)//2 + 43


def congress_to_year(congress):
    # return second year of each congressional section (rough midpoint)
    return (congress-43)*2 + 1874


def congress_to_decade(congress):
    # return the midpoint of the decade in which a congressional session falls
    return (congress - 47) // 5 * 10 + 1880 + 5


def get_early_congress_range():
    return 43, 73


def get_mid_congress_range():
    return 70, 88


def get_modern_congress_range():
    return 85, 114


def get_uscr_congress_range():
    return 112, 116
