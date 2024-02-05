import re, regex
import numpy as np
import panphon

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from itertools import product
import matplotlib.pyplot as plt

from Rule_Systems import rule_system_1, rule_system_2, rule_system_3, rule_system_4, galac_rule_system

ft=panphon.FeatureTable()


vowel2uml = {'a': 'ä', 'o': 'ö', 'u': 'ü', 'A': 'Ä', 'O': 'Ö', 'U': 'Ü'}

with open('../Data/Data_from_Scraping.csv', 'r', encoding='UTF-8') as file:
    lines = file.readlines()


def replace_sublist(original_list, sublist, replacement_item):
    try:
        index = None
        for i in range(len(original_list) - len(sublist) + 1):
            if original_list[i:i+len(sublist)] == sublist:
                index = i
                break

        if index is not None:
            result_list = original_list[:index] + [replacement_item] + original_list[index + len(sublist):]
            return result_list
        else:
            raise ValueError
    except ValueError:
        return original_list


def normalize_plural(plural):
        #normalizing plural forms by plural merges
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



def parse_line(line):
    stressed_syllable = line.split(';')[2].strip()
    line = line.split(';')[1]
    word_separated = re.sub(re.compile('[^a-zA-ZßäüöÄÜÖÉé·]+'), '', line.split(' <')[0].split(' ')[0].replace('(in)', ''))
    syllable_num = str(word_separated.count('·') + 1)
    word = word_separated.replace('·', '')
    phonetic_representation = [segment[1:-1] for segment in line.split(' ') if segment.startswith('[') and (segment.endswith(']') or segment.endswith(','))][0]
    phonetic_representation = phonetic_representation.replace('ˈ', '').replace(':', 'ː').replace('ʤ', 'd͡ɮ').replace('ç', 'ç').replace('g', 'ɡ').replace('n̯', 'n').replace('t̩', 't').replace('z̥', 'z')
    phonetic_representation = regex.findall(r'\X', phonetic_representation, flags=regex.UNICODE)
    phonetic_representation = replace_sublist(phonetic_representation, ['d͡', 'ɮ'], 'd͡ɮ')
    while 'ː' in phonetic_representation:
        phonetic_representation = replace_sublist(phonetic_representation, ['i', 'ː'], 'iː')
        phonetic_representation = replace_sublist(phonetic_representation, ['y', 'ː'], 'yː')
        phonetic_representation = replace_sublist(phonetic_representation, ['u', 'ː'], 'uː')
        phonetic_representation = replace_sublist(phonetic_representation, ['o', 'ː'], 'oː')
        phonetic_representation = replace_sublist(phonetic_representation, ['a', 'ː'], 'aː')
        phonetic_representation = replace_sublist(phonetic_representation, ['e', 'ː'], 'eː')
        phonetic_representation = replace_sublist(phonetic_representation, ['ɛ', 'ː'], 'ɛː')
        phonetic_representation = replace_sublist(phonetic_representation, ['ø', 'ː'], 'øː')
        phonetic_representation = replace_sublist(phonetic_representation, ['õ', 'ː'], 'õː')
        phonetic_representation = replace_sublist(phonetic_representation, ['ã', 'ː'], 'ãː')


    pattern = re.compile(r'<([^>]+)>')
    matches = re.findall(pattern, line)
    genitive = ''.join(''.join(matches).split(', ')[0])
    plural = '-'
    if len(' '.join(matches).split(', ')) > 1:
        plural = ''.join(''.join(matches).split(', ')[1:])
    for n, l in enumerate(word):
        umlaut = False
        if vowel2uml.get(l):
            if len(plural) > n:
                if plural[n] == vowel2uml.get(l):
                    umlaut = True
                    break
    if any([set(plural).intersection(vowel2uml.values())]) and not any([set(word).intersection(vowel2uml.values())]):
        if len(word) == len(plural):
            plural = '-U'
    if plural[0] != '-':
        result = f'-{plural[len(word):]}'
        if umlaut:
            result = '-U'+result[1:]
    else:
        result = plural
    plural = result

    
    reverse_word = word[::-1]
    if 'U' not in plural:
        surface_plural = word + plural.replace('-', '')
    else:
        new_word = str()
        binary = True
        for l in reverse_word:
            if l in vowel2uml.keys() and binary:
                new_word += vowel2uml[l]
                binary = False
            else:
                new_word += l
        surface_plural = new_word[::-1].replace('aü', 'äu') + plural.replace('-', '').replace('U', '')

    gender = ''
    x = 1
    while gender not in('f', 'm', 'nt'):
        gender = re.sub(r'\([^)]*\)', '', line.split(' ')[-x]).strip()
        x += 1

    plural = normalize_plural(plural)
    return word, syllable_num, phonetic_representation, plural, genitive, word_separated, surface_plural, gender, stressed_syllable


entries = list()
for line in lines:

    entries.append(parse_line(line))
words, words_separated = list(), list()
plural_forms, genitive_forms, phonetic_endings = [set() for times in range(3)]
scores = list()
signs= set('-')
word2plurals = {entry[0]: list() for entry in entries}

def get_phonetic_feature(feature, sign):
    if sign != '-':
        return ft.word_array([feature], sign)[0][0]
    else:
        return 0


for word, syllable_num, phonetic_representation, plural, genitive, word_separated, surface_plural, gender, stressed_syllable in entries:
    plural_forms.add(plural)
    words.append(word)
    words_separated.append(word_separated)
    genitive_forms.add(genitive)
    word2plurals[word].append(plural)
    for sign in phonetic_representation:
        signs.add(sign)
    while len(phonetic_representation) < 3:
        phonetic_representation = ['-'] + phonetic_representation


plural_forms = sorted(plural_forms)

signs = sorted(signs)

sign2id = {sign: id for id, sign in enumerate(signs)}

plural_forms_counts = {form: 0 for form in plural_forms}
genitive_forms_counts = {form: 0 for form in genitive_forms}

for word, syllable_num, phonetic_representation, plural, genitive, word_separated, surface_plural, gender, stressed_syllable in entries:
    plural_forms_counts[plural] += 1
    genitive_forms_counts[genitive] += 1


def get_phonetic_border_counts(n, reverse):
    if reverse:
        phonetic_endings = set([entry[2][-n:] for entry in entries])
        phonetic_endings_counts = {form: 0 for form in phonetic_endings}
        for entry in entries:
            phonetic_endings_counts[entry[2][-n:]] += 1
    else:
        phonetic_endings = set([entry[2][:n] for entry in entries])
        phonetic_endings_counts = {form: 0 for form in phonetic_endings}
        for entry in entries:
            phonetic_endings_counts[entry[2][:n]] += 1

    print(f"There are {len(phonetic_endings)} different segments, {round(len([ending for ending in phonetic_endings if phonetic_endings_counts[ending] == 1])/len(phonetic_endings)*100, 1)}% of them are unique.")

def get_phonetic_features(segment):
    if segment != '-':
        return ft.word_array(phonF, segment)
    else:
        return np.zeros(len(phonF))

genitF = ['-ns', '-[s]', '-[e]s', '-n[s]', '-n', '-', '-en', '-es', '-s']
plurF = ['-n', '-ten', '-U', '-Ue', '-', '-s', '-en', '-Uer', '-e', '-er', '-se']
phonF = ['syl', 'son', 'cons', 'cont', 'delrel', 'lat', 'nas', 'strid', 'voi', 'sg', 'cg', 'ant', 'cor', 'distr', 'lab', 'hi', 'lo', 'back', 'round', 'velaric', 'tense', 'long']
phonP = ['-1', '-2', '-3']
genderF = ['m', 'f', 'nt']
sylnumF = ['1', '2', '3', '4', '5', '6']
sylstreF = ['1', '2', '3', '4', '5']


versefootF = list(product(sylnumF, sylstreF))
phonPF = list(product(phonF, phonP))
genderF = ['m', 'f', 'nt']
gendergenitF = list(product(genitF, genderF))
nonplurF = versefootF + phonPF + genderF + gendergenitF
FEATURES = list(product(nonplurF, plurF))
feature2id = {feature:num for num, feature in enumerate(FEATURES)}



def encode_feature(value, feature_values):
    feature_array = np.zeros(len(feature_values))
    if value in feature_values:
        feature_array[feature_values.index(value)] = 1.
    return feature_array

def extract_features(entry, label):
    word, syllable_num, phonetic_representation, plural, genitive, word_separated, surface_plural, gender, stressed_syllable = entry


    while len(phonetic_representation) < 3:
        phonetic_representation = ['-'] + phonetic_representation

    last_phonetic_features = np.concatenate(tuple([get_phonetic_features(phonetic_representation[-n]).flatten() for n in range(1, 4)]))
    last_phonetic_features = OneHotEncoder().fit_transform(last_phonetic_features.reshape(-1, 1)).toarray().flatten()

    genitive_gender_feature = encode_feature((genitive, gender), gendergenitF)
    versefoot_feature = encode_feature((syllable_num, stressed_syllable), versefootF)



    nonplurF = np.concatenate((last_phonetic_features, genitive_gender_feature, versefoot_feature), axis=0)
    plural = encode_feature(plural, sorted(list(plural_forms)))

    return np.append(np.outer(plural, nonplurF).flatten(), np.array(int(label)))




feature_mappings = [feature+f'={value} at position -{position}' for position in range(1, 4)for feature in phonF for value in range(-1, 2)] + gendergenitF + versefootF
feature_mappings = [f'{text} with plural {plural}' for plural in sorted(plural_forms) for text in feature_mappings] + ['label']


genderXplural = product(genderF, plural_forms)
genderXplural = {e: 0 for e in genderXplural}
for word, syllable_num, phonetic_representation, plural, genitive, word_separated, surface_plural, gender, stressed_syllable in entries:
    for key in genderXplural:
        if key == (gender, plural):
            genderXplural[key] += 1

genitiveXplural = {e: 0 for e in product(plural_forms, genitive_forms)}

for x in genitiveXplural:
    for word, syllable_num, phonetic_representation, plural, genitive, word_separated, surface_plural, gender, stressed_syllable in entries:
        if genitive == x[1] and plural == x[0]:
            genitiveXplural[x] += 1


def measure_percentage(function):
    results = list()
    for word, syllable_num, phonetic_representation, plural, genitive, word_separated, surface_plural, gender, stressed_syllable in entries:
        results.append(plural == function(word, gender, genitive, phonetic_representation))
    return (sum(results) / len(results)) * 100


print(f"Rule System Accuracy {round(measure_percentage(rule_system_4), 1)}%")
quit()

"""
The ML operation begins here.
"""

#generating true instances
true_instances = list()
for entry in entries:
    true_instances.append(extract_features(entry, True))
true_instances = np.array(true_instances)


#generating false instances
false_instances = list()
completed_words = list()
for entry in entries:
    word, syllable_num, phonetic_representation, plural, genitive, word_separated, surface_plural, gender, stressed_syllable = entry
    if word not in completed_words:
        completed_words.append(entry[0])
        for alternative_plural in [item for item in plural_forms if item not in word2plurals[word]]:
            false_instances.append(extract_features((word, syllable_num, phonetic_representation, alternative_plural, genitive, word_separated, surface_plural, gender, stressed_syllable), False))
false_instances = np.array(false_instances)


instances = np.vstack((true_instances, false_instances)) #shape: (5058, 1276)
X, y = instances[:, :-1], instances[:, -1:].flatten()


#model = Perceptron(penalty='l1')
#model = GaussianNB()
#model = MultinomialNB()
#model = PassiveAggressiveClassifier()
#model = KNeighborsClassifier(n_neighbors=5)
#model = MLPRegressor()
model = DecisionTreeClassifier(random_state=42)
#model = SVC(kernel='linear') # kernel='linear': 93.6%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {round(model.score(X_test, y_test) * 100, 1)}%")

def get_word_info(query_word):
    query = words.index(query_word)

    csr = model.decision_path(true_instances[:, :-1][query].reshape(1, -1))
    print(f"Nodes for this path: {model.decision_path(instances[:, :-1][query].reshape(1, -1)).nnz}")
    print(f"Leave node id: {model.apply(true_instances[:, :-1][query].reshape(1, -1))[0]}")
    print(*[words[n] for n, x in enumerate(true_instances[:, :-1]) if model.apply(true_instances[:, :-1][n].reshape(1, -1))[0] == model.apply(true_instances[:, :-1][query].reshape(1, -1))[0]])
    feature_indices = model.tree_.feature
    thresholds = model.tree_.threshold
    feature_names_mapping = [feature_mappings[idx] if idx != -2 else "Leaf" for idx in feature_indices]

    for num in csr.nonzero()[1]:
        feature_name = feature_names_mapping[num]
        if feature_name == "Leaf":
            print(f"{num};{feature_name};")
        else:
            threshold = thresholds[num]
            feature_index = feature_indices[num]
            is_positive = "Left" if X_test[query, feature_index] > threshold else "Right"
            print(f"{num};{feature_name};{is_positive}")


export_graphviz(model, out_file='../Data/Tree.dot', feature_names=feature_mappings[:-1], label='root', filled=True, proportion=True, rounded=True, node_ids=True)

#get extreme coefficients
# coeffs = list()
# for n in range(5):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=n)
#     model = SVC(kernel='linear')
#     model.fit(X_train, y_train)
#     coeffs.extend(list(np.where(np.tanh(model.coef_) > 0.8)[1]))
# from collections import Counter
# indices = list()
# for k, v in Counter(coeffs).items():
#     if v == 5:
#         indices.append(k)
# for x in indices:
#     print(feature_mappings[x])
# print(len(indices))


