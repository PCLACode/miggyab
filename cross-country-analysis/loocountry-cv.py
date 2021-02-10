import os
import pandas as pd
import math
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from datetime import datetime


def loocountry_cv(countries, folder, classifier, outfile):
    outfile.write('test_country,auc\n')

    for test_country in countries:
        print('starting {} at {}'.format(test_country, datetime.now()))

        train_countries = list()
        for f in os.listdir(folder):
            train_country = f.split('.')[0]
            if train_country in countries and train_country != test_country:
                train_countries.append(pd.read_csv(os.path.join(folder, f), low_memory=False))

        X_train = pd.concat(train_countries, ignore_index=True)
        X_test = pd.read_csv(os.path.join(folder, '{}.csv'.format(test_country)))

        y_train = [1 - math.floor(x) for x in list(X_train['drop_percent'])]
        y_test = [1 - math.floor(x) for x in list(X_test['drop_percent'])]

        X_train.drop(['course', 'session', 'user_id', 'country', 'continent', 'drop_percent'], axis=1, inplace=True)
        X_test.drop(['course', 'session', 'user_id', 'country', 'continent', 'drop_percent'], axis=1, inplace=True)

        classifier.fit(X_train, y_train)
        predictions = classifier.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, predictions)

        outfile.write('{},{}\n'.format(test_country, auc_roc))


def main():
    included_countries = ['AE', 'AR', 'AT', 'AU', 'BD', 'BE', 'BF', 'BG', 'BR', 'CA', 'CH', 'CL', 'CN', 'CO', 'CZ',
                          'DE', 'DJ', 'DK', 'EE', 'EG', 'ES', 'FI', 'FR', 'GB', 'GH', 'GR', 'HK', 'HR', 'HU', 'ID',
                          'IE', 'IN', 'IQ', 'IR', 'IT', 'JP', 'KR', 'KW', 'LB', 'LR', 'LT', 'LU', 'LV', 'LY', 'MA',
                          'MT', 'MU', 'MX', 'MY', 'MZ', 'NG', 'NL', 'NO', 'NZ', 'PE', 'PH', 'PK', 'PL', 'PT', 'RO',
                          'RS', 'RU', 'RW', 'SA', 'SD', 'SE', 'SG', 'SI', 'SK', 'SN', 'SO', 'SV', 'TH', 'TR', 'TT',
                          'TZ', 'UG', 'US', 'UY', 'VN', 'ZA', 'ZM', 'ZW']
    classifier = XGBClassifier(learning_rate=0.5, n_estimators=200, min_samples_split=.1, min_samples_leaf=.05,
                               min_weight_fraction_leaf=.05, random_state=5, verbosity=0)
    with open('loocountry.csv', 'w+') as outfile:
        loocountry_cv(included_countries, 'countries/', classifier, outfile)


if __name__ == '__main__':
    main()
