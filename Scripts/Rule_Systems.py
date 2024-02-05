import panphon



ft=panphon.FeatureTable()


def normalize_plural(plural):
    # normalizing plural forms by plural merges
    result = plural
    if plural == '-n':
        result = '-en'
    elif plural == '-':
        result = '-e'
    elif plural == '-se':
        result = '-e'
    elif plural == '-U':
        result = '-Ue'
    elif plural == '-er':
        result = '-Uer'
    elif plural == '-ten':
        result = '-en'
    return result


def rule_system_1(word, gender, genitive, phonetic_representation): #72.2%
    if gender in('m', 'nt'):
        return normalize_plural('-e')
    elif gender == 'f':
        return normalize_plural('-en')

def rule_system_2(word, gender, genitive, phonetic_representation):   #73.1
    if genitive == '-[e]s':
        return normalize_plural('-e')
    elif genitive == '-es':
        return normalize_plural('-e')
    elif genitive == '-s':
        return normalize_plural('-e')
    elif genitive == '-ns':
        return normalize_plural('-en')
    elif genitive == '-en':
        return normalize_plural('-en')
    elif genitive == '-n':
        return normalize_plural('-eb')
    elif genitive == '-':
        return normalize_plural('-en')

def rule_system_3(word, gender, genitive, phonetic_representation): #74.5%
    if gender == 'f':
        return normalize_plural('-en')
    elif genitive in('-', '-en', '-n'):
        return normalize_plural('-en')
    else:
        return normalize_plural('-e')

def rule_system_4(word, gender, genitive, phonetic_representation): #75.3
    if (ft.word_array(['back', 'delrel', 'lo'], phonetic_representation[-1]).flatten() == 0).any():
        return normalize_plural('-e')
    elif bool(ft.word_array(['lo'], phonetic_representation[-1]).flatten() == 1):
        return normalize_plural('-s')
    elif bool(ft.word_array(['tense'], phonetic_representation[-1]).flatten() == -1):
        return normalize_plural('-en')
    elif bool(ft.word_array(['delrel'], phonetic_representation[-2]).flatten() == 1):
        return normalize_plural('-e')
    else:
        if gender == 'f':
            return normalize_plural('-en')
        elif genitive in('-', '-en', '-n'):
            return normalize_plural('-en')
        else:
            return normalize_plural('-e')

def galac_rule_system(word, gender, genitive, phonetic_representation):
    #A
    if word.endswith(('heit', 'keit', 'ung', 'schaft', 'anz', 'enz', 'tät', 'ion', 'ik', 'ur', 'in', 'or')):
        return normalize_plural('-en')
    elif word.endswith(('ling', 'ig', 'ich', 'icht', 'nis', 'sal', 'an', 'ar', 'är', 'on', 'ett', 'eur', 'il', 'iv', 'in')):
        return normalize_plural('-e')
    elif word.endswith('tum'):
        return normalize_plural('-Uer')
    elif word.endswith('chen'):
        return normalize_plural('-')

    #B
    if word[-1] == 'e':
        if word == 'Käse' or (gender == 'nt' and word.lower().startswith('ge')):
            return normalize_plural('-')
        else:
            return normalize_plural('-en')

    #C
    if gender == 'nt' and word.endswith(('er', '-en')):
        return normalize_plural('-')
    elif gender == 'm' and word.endswith(('er', '-en')):
        return normalize_plural('-')

    #D
    if gender in('m', 'nt'):
        if word in('Muskel', 'Stachel', 'Pantoffel'):
            return normalize_plural('-n')
        elif word.endswith('el'):
            if gender == 'm':
                return normalize_plural('-')
            return normalize_plural('-')

    #E
    if gender == 'f' and word.endswith(('el', 'er')):
        if word in('Mutter', 'Tochter'):
            return '-U'
        return normalize_plural('-en')

    #a
    if gender == 'f':
        return normalize_plural('-en')

    #b
    if gender == 'm':
        return normalize_plural('-e')

    #c
    if gender == 'n':
        return normalize_plural('-e')

    #d
    if word[-1].lower() in ('a', 'e', 'i', 'u'):
        if (gender == 'f' and phonetic_representation[-1] in('e:', 'i:') or (gender == 'nt' and word.endswith('ma'))):
            return normalize_plural('-en')
        else:
            return  normalize_plural('-s')

    #if nothing returned yet
    return '-e'