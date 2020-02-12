import pandas as pd
from sklearn.preprocessing import LabelBinarizer, Normalizer, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import  datetime
from sklearn.decomposition import PCA

train_data_types = {
    "id": int,
    "amount_tsh": float,
    "date_recorded": str,
    "funder": str,
    "gps_height": float,
    "installer": str,
    "longitude": float,
    "latitude": float,
    "wpt_name": str,
    "num_private": float,
    "basin": str,
    "subvillage": str,
    "region": str,
    "region_code": str,
    "district_code": str,
    "lga": str,
    "ward": str,
    "population": float,
    "public_meeting": str,
    "recorded_by": str,
    "scheme_management": str,
    "scheme_name": str,
    "permit": str,
    "construction_year": int,
    "extraction_type": str,
    "extraction_type_group": str,
    "extraction_type_class": str,
    "management": str,
    "management_group": str,
    "payment": str,
    "payment_type": str,
    "water_quality": str,
    "quality_group": str,
    "quantity": str,
    "quantity_group": str,
    "source": str,
    "source_type": str,
    "source_class": str,
    "waterpoint_type": str,
    "waterpoint_type_group": str
}

labels_type = {
    "id": int,
    "status_group": str
}

text_variables = ["funder", "installer"]
logical_variables = ["permit", "public_meeting"]
numerical_variables = ["population", "longitude", "latitude", "gps_height", "amount_tsh"]
drop_variables = ["num_private", "recorded_by", "payment_type", "quantity_group", "waterpoint_type_group",
                  "management_group", "extraction_type_group", "extraction_type_class", "scheme_management"]

discretized_labels = ["subvillage", "lga", "ward", "wpt_name",
                      "basin", "region", "region_code", "district_code", "scheme_name",
                      "extraction_type"]

encoded_labels = [*discretized_labels, "funder", "installer", "permit", "public_meeting", "management",
                  "payment", "water_quality", "quality_group", "quantity", "source", "source_type", "source_class",
                  "waterpoint_type"]


def read_data(filename):
    df = pd.read_csv(filename, dtype=train_data_types)
    return df

class dataPreprocess:
    def __init__(self):
        self.text_variables = text_variables
        self.logical_variables = logical_variables
        self.high_labels_variables = discretized_labels
        self.numerical_variables = numerical_variables
        self.median_values = {}
        self.frequent_values = {}
        self.encoders = {}
        self.pca = None

    def fit(self, train, num_bins=35, num_chars=4):
        self.max_bins = num_bins
        self.num_chars = num_chars
        self.__compute_bins(train)
        self.__learn_missing(train)
        data = self.drop_data(train)
        data = self.handle_missing_values(data)
        data = self.discretize_text_variables(data)
        data = self.discretize_high_labels_variables(data)
        data = self.scale_data(data)
        self.__encode(data)
        data = self.encode_data(data)
        self.train_pca(data)
        return self

    def __encode(self, train):
        for variable in encoded_labels:
            encoder = LabelEncoder()
            encoder.fit(train[variable])
            self.encoders[variable] = encoder


    def __learn_missing(self, train):
        imputer = SimpleImputer(strategy="most_frequent")
        imputer.fit(train[self.logical_variables])
        for variable, value in zip(self.logical_variables, imputer.statistics_):
            self.frequent_values[variable] = value
        zero_variables = ["construction_year", "longitude", "latitude"]
        zero_inputer = SimpleImputer(missing_values=0, strategy="median")
        zero_inputer.fit(train[zero_variables])
        for variable, value in zip(zero_variables, zero_inputer.statistics_):
            self.median_values[variable] = value

    def train_pca(self, data):
        pca = PCA(n_components=5)
        pca.fit(data)
        self.pca = pca

    def apply_pca(self, data):
        columns = ['pca_%i' % i for i in range(5)]
        df_pca = pd.DataFrame(self.pca.transform(data), columns=columns, index=data.index)
        return df_pca

    def transform(self, data):
        data = self.drop_data(data)
        data = self.handle_missing_values(data)
        data = self.discretize_text_variables(data)
        data = self.discretize_high_labels_variables(data)
        data = self.scale_data(data)
        data = self.encode_data(data)
        return data

    def encode_data(self, data):
        data["date_recorded"] = data["date_recorded"].transform(
            func=lambda date: datetime.strptime(date, "%Y-%m-%d").timestamp())
        for variable in encoded_labels:
            data[variable] = self.encoders[variable].transform(data[variable])
        return data

    def __get_initial_chars(self, feature):
        feature_data = list(feature)
        chars_data = pd.Series([str(char).lower()[0:self.num_chars] for char in feature_data])
        return chars_data

    def __compute_bins(self, data):
        dict_features = {}
        for feature in [*self.text_variables, *self.high_labels_variables]:
            chars_data = self.__get_initial_chars(data[feature])
            series = chars_data.value_counts()
            series.sort_values(ascending=False, inplace=True)
            top_data = list(series[0:self.max_bins].index)
            dict_features[feature] = top_data
        self.bins = dict_features

    def handle_missing_values(self, data):
        categorical_features = list(data.columns[data.isnull().any()])
        categorical_features = [x for x in categorical_features if x not in logical_variables]
        data[categorical_features] = data[categorical_features].fillna("unknown")
        for variable, value in self.frequent_values.items():
            data[variable] = data[variable].fillna(value)
        for variable, value in self.median_values.items():
            df = data[variable].copy()
            df[df == 0] = value
            data[variable] = df
        return data

    def discretize_text_variables(self, data):
        for feature in self.text_variables:
            series = self.__get_initial_chars(data[feature])
            temp_data = [value if value in self.bins[feature] else "other" for value in series.values]
            data[feature] = temp_data
        return data

    def discretize_high_labels_variables(self, data):
        for feature in self.high_labels_variables:
            series = self.__get_initial_chars(data[feature])
            temp_data = [value if value in self.bins[feature] else "other" for value in series.values]
            data[feature] = temp_data
        return data

    def __bin_feature(self, data, feature):
        labelmodel = LabelBinarizer()
        bins = labelmodel.fit_transform(data[feature])
        classes = ["{}-{}".format(feature, clas) for clas in labelmodel.classes_]
        temp_df = pd.DataFrame(bins, columns=classes)
        for label in classes:
            data[label] = temp_df[label].to_numpy()
        data = data.drop([feature], axis=1)
        return data

    def drop_data(self, data):
        data = data.drop(drop_variables, axis=1)
        return data

    def scale_data(self, data):
        data[self.numerical_variables] = Normalizer().fit_transform(data[self.numerical_variables])
        return data
