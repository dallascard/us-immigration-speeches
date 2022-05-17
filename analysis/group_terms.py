
# Countries that have represented at least 1% of the US foreign-born population in some decadal census

countries = {'Ireland': ['Ireland'],
             'Germany': ['Germany'],
             'Mexico': ['Mexico'],
             'Italy': ['Italy'],
             'England': ['England'],
             'Canada': ['Canada'],
             'Russia': ['Russia', 'USSR'],
             'Poland': ['Poland'],
             'China': ['China'],
             'India': ['India'],
             'Sweden': ['Sweden'],
             'Austria': ['Austria'],
             'Philippines': ['Philippines', 'Philippine'],
             'Cuba': ['Cuba'],
             'Hungary': ['Hungary'],
             'Norway': ['Norway'],
             'Czechoslovakia': ['Czechoslovakia', 'Czech', 'Slovakia', 'Slovak'],
             'Vietnam': ['Vietnam'],
             'Scotland': ['Scotland'],
             'El Salvador': ['El Salvador'],
             'Korea': ['Korea'],
             'France': ['France'],
             'Dominican Republic': ['Dominican'],
             'Guatemala': ['Guatemala'],
             'Greece': ['Greece'],
             'Colombia': ['Colombia'],
             'Jamaica': ['Jamaica'],
             'Yugoslavia': ['Yugoslavia', 'Serbia', 'Croatia', 'Macedonia', 'Bosnia', 'Herzegovina', 'Montenegro'],
             'Honduras': ['Honduras'],
             'Japan': ['Japan'],
             'Haiti': ['Haiti'],
             'Portugal': ['Portugal'],
             'Denmark': ['Denmark'],
             'Lithuania': ['Lithuania'],
             'Switzerland': ['Switzerland'],
             'Wales': ['Wales'],
             'Taiwan': ['Taiwan'],
             'Netherlands': ['Netherlands', 'Holland'],
             'Brazil': ['Brazil'],
             'Finland': ['Finland'],
             'Iran': ['Iran'],
             'Ecuador': ['Ecuador'],
             'Venezuela': ['Venezuela'],
             'Romania': ['Romania', 'Rumania', 'Roumania'],
             'Peru': ['Peru']
             }


# Regions excluding England, France, Canada, and India, due to potential for name confusion
regions = {'Latin America': ['Mexico', 'Cuba', 'El Salvador', 'Guatemala', 'Colombia', 'Honduras', 'Ecuador', 'Venezuela', 'Peru'],
           'Europe': ['Ireland', 'Germany', 'Italy', 'Russia', 'USSR', 'Poland', 'Sweden', 'Austria', 'Hungary', 'Norway', 'Czechoslovakia', 'Czech', 'Slovakia', 'Slovak', 'Greece', 'Yugoslavia', 'Serbia', 'Croatia', 'Macedonia', 'Bosnia', 'Herzegovina', 'Montenegro', 'Portugal', 'Denmark', 'Lithuania', 'Switzerland', 'Netherlands', 'Holland', 'Finland', 'Romania', 'Rumania', 'Roumania'],
           'Asia': ['China', 'Philippines', 'Philippine', 'Vietnam', 'Korea', 'Japan', 'Taiwan']
           }

def get_countries():
    return countries


nationalities = {'Ireland': ['Irish', 'Irishman', 'Irishmen'],
                 'Germany': ['German', 'Germans'],
                 'Mexico': ['Mexican', 'Mexicans'],
                 'Italy': ['Italian', 'Italians'],
                 'England': ['Englishman', 'Englishmen'],
                 'Canada': ['Canadian', 'Canadians'],
                 'Russia': ['Russian', 'Russians'],
                 'Poland': ['Polish', 'Poles'],
                 'China': ['Chinese', 'Chinaman', 'Chinamen'],
                 'India': ['Indian', 'Indians'],
                 'Sweden': ['Swedish', 'Swedes'],
                 'Austria': ['Austrian', 'Austrians'],
                 'Philippines': ['Filipino', 'Filipinos', 'Filipina', 'Filipinas'],
                 'Cuba': ['Cuban', 'Cubans'],
                 'Hungary': ['Hungarian', 'Hungarians'],
                 'Norway': ['Norwegian', 'Norwegians'],
                 'Czechoslovakia': ['Czech', 'Czechs', 'Slovak', 'Slovaks', 'Slovakian', 'Slovakians', 'Czechoslovakian', 'Czechoslovakians'],
                 'Vietnam': ['Vietnamese'],
                 'Scotland': ['Scottish', 'Scots', 'Scotsman', 'Scotsmen'],
                 'El Salvador': ['Salvadoran', 'Salvadorans', 'Salvadorian', 'Salvadorians'],
                 'Korea': ['Korean', 'Koreans'],
                 'France': ['Frenchman', 'Frenchmen'],
                 'Dominican Republic': ['Dominican', 'Dominicans'],
                 'Guatemala': ['Guatemalan', 'Guatemalans'],
                 'Greece': ['Greek', 'Greeks'],
                 'Colombia': ['Colombian', 'Colombians'],
                 'Jamaica': ['Jamaican', 'Jamaicans'],
                 'Yugoslavia': ['Yugoslavian', 'Yugoslavians', 'Serbian', 'Serbians', 'Serb', 'Serbs', 'Croatian', 'Croatians', 'Croat', 'Croats', 'Macedonian', 'Macedonians', 'Bosnian', 'Bosnians'],
                 'Honduras': ['Honduran', 'Hondurans'],
                 'Japan': ['Japanese', 'Jap', 'Japs'],
                 'Haiti': ['Haitian', 'Haitians'],
                 'Portugal': ['Portuguese'],
                 'Denmark': ['Danish', 'Danes'],
                 'Lithuania': ['Lithuanian', 'Lithuanians'],
                 'Switzerland': ['Swiss'],
                 'Wales': ['Welsh', 'Welshman', 'Welshmen'],
                 'Taiwan': ['Taiwanese'],
                 'Netherlands': ['Dutch'],
                 'Brazil': ['Brazilian', 'Brazilians'],
                 'Finland': ['Finnish', 'Finn', 'Finns',],
                 'Iran': ['Iranian', 'Iranians'],
                 'Ecuador': ['Ecuadorian', 'Ecuadorians'],
                 'Venezuela': ['Venezuelan', 'Venezuelans'],
                 'Romania': ['Romanian', 'Romanians', 'Rumanian', 'Rumanians', 'Roumania', 'Roumanian', 'Roumanians'],
                 'Peru': ['Peruvian', 'Peruvians']
                 }

regionalities = {'Latin America': ['Mexican', 'Mexicans', 'Cuban', 'Cubans', 'Salvadoran', 'Salvadorans', 'Salvadorian', 'Salvadorians', 'Guatemalan', 'Guatemalans', 'Colombian', 'Colombians', 'Honduran', 'Hondurans', 'Ecuadorian', 'Ecuadorians', 'Venezuelan', 'Venezuelans', 'Peruvian', 'Peruvians', 'Hispanic', 'Hispanics', 'Latino', 'Latinos', 'Latina', 'Latinas'],
                 'Europe': ['Irish', 'Irishman', 'Irishmen', 'German', 'Germans', 'Italian', 'Italians', 'Russian', 'Russians', 'Polish', 'Poles', 'Swedish', 'Swedes', 'Austrian', 'Austrians', 'Hungarian', 'Hungarians', 'Norwegian', 'Norwegians', 'Czech', 'Czechs', 'Slovak', 'Slovaks', 'Slovakian', 'Slovakians', 'Czechoslovakian', 'Czechoslovakians', 'Greek', 'Greeks', 'Yugoslavian', 'Yugoslavians', 'Serbian', 'Serbians', 'Serb', 'Serbs', 'Croatian', 'Croatians', 'Croat', 'Croats', 'Macedonian', 'Macedonians', 'Bosnian', 'Bosnians', 'Portuguese', 'Danish', 'Danes', 'Lithuanian', 'Lithuanians', 'Swiss', 'Dutch', 'Finnish', 'Finn', 'Finns', 'Romanian', 'Romanians', 'Rumanian', 'Rumanians', 'Roumania', 'Roumanian', 'Roumanians'],
                 'Asia': ['Chinese', 'Chinaman', 'Chinamen', 'Filipino', 'Filipinos', 'Filipina', 'Filipinas', 'Vietnamese', 'Korean', 'Koreans', 'Japanese', 'Jap', 'Japs', 'Taiwanese']
                }

def get_nationalities():
    return nationalities


def get_regions_and_regionalities():
    return regions, regionalities


def add_american():
    # add American and Americans to appropriate terms, and create replacements
    american_terms = {}
    substitutions = {}
    for nationality, terms in nationalities.items():
        #american_terms[nationality] = terms.copy()
        american_terms[nationality] = [terms[0] + 'American', terms[0] + 'Americans']
        substitutions[terms[0] + ' ' + 'American'] = terms[0] + 'American'
        substitutions[terms[0] + ' ' + 'Americans'] = terms[0] + 'Americans'
        substitutions[terms[0] + '-American'] = terms[0] + 'American'
        substitutions[terms[0] + '-Americans'] = terms[0] + 'Americans'
        for term in terms[1:]:
            if term[-1] != 's' and term[-3:] != 'men' and term[-3:] != 'man':
                american_terms[nationality].extend([term + 'American', term + 'Americans'])
                substitutions[term + ' ' + 'American'] = term + 'American'
                substitutions[term + ' ' + 'Americans'] = term + 'Americans'
                substitutions[term + '-American'] = term + 'American'
                substitutions[term + '-Americans'] = term + 'Americans'
    return american_terms, substitutions


early_chinese_terms = {'china', 'chinese', 'chinamen', 'chinaman', 'indochina', 'indochinese',
                       'asia', 'asian', 'asians', 'asiatic', 'asiatics',
                       'orient', 'oriental', 'orientals', 'celestial', 'celestials',
                       'cooly', 'coolie', 'coolys', 'coolies',
                       'mongolian', 'mongolians', 'mongol', 'mongols'}

european_countries = {'Ireland',
                      'Germany',
                      'Italy',
                      'England',
                      'Poland',
                      'Sweden',
                      'Austria',
                      'Hungary',
                      'Norway',
                      'Czechoslovakia',
                      'Scotland',
                      'France',
                      'Greece',
                      'Yugoslavia',
                      'Portugal',
                      'Denmark',
                      'Lithuania',
                      'Switzerland',
                      'Wales',
                      'Netherlands',
                      'Finland',
                      'Romania'}

modern_mexican_terms = {'mexico',
                        'mexican',
                        'mexicans',
                        'wetback',
                        'wetbacks',
                        'bracero',
                        'braceros',
                        'mexicanamerican',
                        'mexicanamericans'}


modern_hispanic_terms = {'mexico',
                        'mexican',
                        'mexicans',
                        'wetback',
                        'wetbacks',
                        'bracero',
                        'braceros',
                        'mexicanamerican',
                        'mexicanamericans',
                        'hispanic',
                        'hispanics',
                        'latino',
                        'latinos',
                        'latina',
                        'latinas',
                        'hispanicamerican',
                        'hispanicamericans',
                        'cuba',
                        'cuban',
                        'cubanamerican',
                        'cubanamericans',
                        'cubans',
                        'salvador',
                        'salvadorans',
                        'salvadorian',
                        'salvadorians',
                        'guatemala',
                        'guatemalan',
                        'guatemalans',
                        'colombia',
                        'colombian',
                        'colombians',
                        'honduras',
                        'honduran',
                        'hondurans',
                        'ecuador',
                        'ecuadorian',
                        'ecuadorians',
                        'venezuela',
                        'venezuelan',
                        'venezuelans',
                        'peru',
                        'peruvian',
                        'peruvians'}


def get_subset_terms():
    return early_chinese_terms, european_countries, modern_mexican_terms


def get_modern_hispanic_terms():
    return modern_hispanic_terms