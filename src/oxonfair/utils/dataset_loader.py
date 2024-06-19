import numpy as np
import pandas as pd
from oxonfair import DataDict
from sklearn.preprocessing import LabelEncoder


class resample:
    "Induce unfairness by artifically altering the base rate and dropping positive samples for one group"
    def __init__(self, drop_group, drop_label, proportion) -> None:
        self.drop_group = drop_group
        self.drop_label = drop_label
        self.proportion = proportion

    def __call__(self, groups, labels):
        mask = np.random.random_sample(size=labels.shape) < self.proportion
        mask += (groups != self.drop_group)
        mask += (labels != self.drop_label)
        return mask > 0


def uniform_partition(target, groups, train_prop, test_prop, seed=None):
    "sample datasets uniformly. Datasets are assigned 0 for train, 1 for test and 2 for validation"
    target = np.asarray(target).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    assert target.shape == groups.shape
    out = np.empty_like(target, dtype=int)
    if seed is not None:
        np.random.seed(seed)
    for g in np.unique(groups):
        for t in np.unique(target):
            mask = (target == t) * (groups == g)
            size = mask.sum()
            dummy = np.empty(size)
            train_size = round(size*train_prop)
            test_size = train_size+round(size*test_prop)
            dummy[:train_size] = 0
            dummy[train_size:test_size] = 1
            dummy[test_size:] = 2
            np.random.shuffle(dummy)
            out[mask] = dummy
    return out


class partition:
    "Generic dataset loader and partitioner"
    def __init__(self, get_data, default_groups=None, resample=None) -> None:
        self.get_data = get_data
        self.resample = resample
        self.default_groups = default_groups

    def __call__(self,  groups=None, train_proportion=0.5, test_proportion=0.25, *,
                 seed=None, discard_groups=False, replace_groups=False,
                 encoding='ordinal', resample=None):
        """Generic code for controling datapartitioning.
        groups: a specification of the column containing group information that can be understood by pandas
        train_proportion: number between 0 and 1 expressing the proportion of the dataset used for training
        test_proportion: as above but for test
        seed: random seed used to make results deterministic
        discard_groups: if True drop groups from the data
        replace_groups: A dict used to merge small groups e.g.
                    {'Hispanic':'Other', 'Native American':'Other', 'Asian':'Other'}
        encoding: if 'ordinal' or 'onehot' encode data accordingly. If None don't encode.
        resample: override existing resampling. This should be a Resample class.
        """
        assert groups is not None or self.default_groups is not None
        if groups is None:
            groups = self.default_groups
        if resample is None:
            resample = self.resample

        total_data, target, positive_target = self.get_data()

        if callable(groups):
            groups = groups(total_data)

        if isinstance(target, str):
            t_name = target
            target = total_data[target]
            total_data = total_data.drop(t_name, axis=1)
        target = np.asarray(target).reshape(-1)
        if positive_target:
            target = target == positive_target

        assert 0 < target.mean() < 1, 'Something is wrong with the dataset. Every target value is the same.'

        assert 0 < np.asarray(target).mean() < 1

        if isinstance(groups, str):
            g_name = groups
            if replace_groups:
                total_data[groups] = total_data[groups].replace(replace_groups)
            groups = total_data[groups]
            if discard_groups:
                total_data = total_data.drop(g_name, axis=1)
        else:
            if replace_groups:
                groups = groups.replace(replace_groups)

        if resample:
            mask = resample(groups, target)
            total_data = total_data[mask]
            groups = groups[mask]
            target = target[mask]

        total_data.reset_index(drop=True)
        groups.reset_index(drop=True)

        if encoding == 'onehot':
            total_data = total_data.get_dummies()
        elif encoding == 'ordinal':
            total_data = total_data.apply(LabelEncoder().fit_transform)
        elif encoding is not None:
            assert encoding is not None, "encoding must be 'onehot', 'ordinal', or 'None'"

        part = uniform_partition(target, groups, train_prop=train_proportion,
                                 test_prop=test_proportion, seed=seed)
        train = total_data.iloc[part == 0]
        train_groups = groups.iloc[part == 0]
        train_y = target[part == 0]

        val = total_data.iloc[part == 2]
        val_y = target[part == 2]
        val_groups = groups.iloc[part == 2]

        test = total_data.iloc[part == 1]
        test_y = target[part == 1]
        test_groups = groups.iloc[part == 1]

        train_dict = DataDict(train_y, train, train_groups)
        val_dict = DataDict(val_y, val, val_groups)
        test_dict = DataDict(test_y, test, test_groups)
        return train_dict, val_dict, test_dict


def adult_raw():
    train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
    test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
    return pd.concat([train_data, test_data]), 'class', ' >50K'


def compas_raw():
    all_data = pd.read_csv('https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv')
    condensed_data = all_data[['sex', 'race', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                               'priors_count', 'age_cat', 'c_charge_degree', 'two_year_recid']].copy()
    return condensed_data, 'two_year_recid', None


def compas_audit_raw():
    all_data = pd.read_csv('https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv')
    condensed_data = all_data[['sex', 'race', 'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                               'priors_count', 'age_cat', 'c_charge_degree', 'decile_score.1',
                               'v_score_text' 'two_year_recid']].copy()
    return condensed_data, 'two_year_recid', None


def german_col_names(X):
    X.columns = ['status', 'duration', 'history', 'purpose', 'amount', 'savings/bonds', 'employment since',
                 'percent disposable', 'marital status', 'debtors', 'residence since', 'property', 'age',
                 'other plans', 'housing', 'existing credits', 'job', 'people liable', 'telephone', 'foreign worker',]

    X = X.replace(german_dict)
    return X


def taiwan_col_names(X):
    X.columns = ['Limit balance', 'sex', 'education', 'marriage', 'age', 'pay1', 'pay2', 'pay3', 'pay4', 'pay5',
                 'pay6', 'bill amount1', 'bill amount2', 'bill amount3', 'bill amount4', 'bill amount5', 'bill amount6',
                 'pay amount1', 'pay amount2', 'pay amount3', 'pay amount4', 'pay amount5', 'pay amount6']
    return X


def german_sex(X):
    return pd.DataFrame(list(map(lambda x: x.startswith('female'), X['marital status'])))


class UCI_raw:
    def __init__(self, index, pos_y_val=None, fix_X=None, fix_y=None) -> None:
        """index is the UCI index used to specify the dataset
        fixX is any function that must be called on X to clean it.
        fixY is a function that must be called on y to clean it
        """
        self.index = index
        self.fix_X = fix_X
        self.fix_y = fix_y
        self.pos_y_val = pos_y_val

    def __call__(self):
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=self.index)

        # data (as pandas dataframes)
        X = data.data.features
        if self.fix_X is not None:
            X = self.fix_X(X)
        y = data.data.targets
        if self.fix_y is not None:
            y = self.fix_y(y)
        return X, y, self.pos_y_val


def replace_nan(X):
    X[X.isnull()] = -1
    return X


diabetes_raw = UCI_raw(891)
support2_raw = UCI_raw(880, fix_y=lambda y: y['death'])
german_raw = UCI_raw(144, pos_y_val=2, fix_X=german_col_names,)
taiwan_default_raw = UCI_raw(350, fix_X=taiwan_col_names)
bank_marketing_raw = UCI_raw(222, pos_y_val='yes')
student_raw = UCI_raw(856)
myocardial_infarction_raw = UCI_raw(579, fix_X=replace_nan, fix_y=lambda y: y['LET_IS'] > 0)

adult = partition(adult_raw, 'sex')
compas = partition(compas_raw, 'race')
compas_audit = partition(compas_audit_raw, 'race')
diabetes = partition(diabetes_raw, 'Sex')
support2 = partition(support2_raw, 'sex')
german = partition(german_raw, german_sex)
taiwan_default = partition(taiwan_default_raw, 'sex')
bank_marketing = partition(bank_marketing_raw, 'marital')
student = partition(student_raw, 'sex')
myocardial_infarction = partition(myocardial_infarction_raw, 'SEX')

german_dict = {
        'A11': '< 0 DM',
        'A12': '< 200 DM',
        'A13': '>= 200 DM / salary assignments for at least 1 year',
        'A14': 'no checking account',
        'A30': 'no credits taken/all credits paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account/ other credits existing (not at this bank)',
        'A40': 'car (new)',
        'A41': 'car (used)',
        'A42': 'furniture/equipment',
        'A43': 'radio/television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': '(vacation - does not exist?)',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others',
        'A61': '         ... <  100 DM',
        'A62': '  100 <= ... <  500 DM',
        'A63': '  500 <= ... < 1000 DM',
        'A64': '         .. >= 1000 DM',
        'A65': '  unknown/ no savings account',
        'A71': 'unemployed',
        'A72': '      ... < 1 year',
        'A73': '1  <= ... < 4 years',
        'A74': '4  <= ... < 7 years',
        'A75': '      .. >= 7 years',
        'A91': 'male   :divorced/separated',
        'A92': 'female :divorced/separated/married',
        'A93': 'male   :single',
        'A94': 'male   :married/widowed',
        'A95': 'female :single',
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor',
        'A121': 'real estate',
        'A122': 'if not A121:building society savings agreement/life insurance',
        'A123': 'if not A121/A122:car or other, not in attribute 6',
        'A124': 'unknown / no property',
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none',
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free',
        'A171': 'unemployed/ unskilled  - non-resident',
        'A172': 'unskilled - resident',
        'A173': 'skilled employee / official',
        'A174': 'management/ self-employed/highly qualified employee/ officer',
        'A191': 'none',
        'A192': 'yes, registered under the customers name',
        'A201': 'yes',
        'A202': 'no'
}
