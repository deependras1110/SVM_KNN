from pickletools import optimize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from collections import Counter
import matplotlib.pyplot as plt


class MLChoice:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def read_data_from_file(self):
        data = pd.read_csv(self.dataset, sep=",", header=None)
        return data

    def calculate_distance(self, p1, p2):
        dist = np.sqrt(np.sum((p1 - p2)**2))
        return dist

    def split_data(self, data, size):
        no_of_features = len(data.columns)
        X = data.iloc[:, 0:no_of_features - 1]
        y = data.iloc[:, no_of_features - 1]

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=size,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def knn(self, X_train, y_train, predict, k):
        if len(X_train) <= k:
            warnings.warn('K is set to a value less than total voting groups')

        eucledian_distances = []
        for feature, output in zip(X_train, y_train):
            ed = self.calculate_distance(feature, predict)
            eucledian_distances.append([ed, output])

        # for i in distances:
        #     print(i)

        votes = [i[1] for i in sorted(eucledian_distances)[:k]]
        votes_result = Counter(votes).most_common(1)[0][0]
        return votes_result

    def test_knn(self, X_train, y_train, X_test, y_test, k):
        result = []
        true = 0
        false = 0

        for feature, expected_output in zip(X_test, y_test):
            output = self.knn(X_train, y_train, feature, k)
            result.append([expected_output, output])

        for i in result:
            if i[0] == i[1]:
                true = true + 1
            else:
                false = false + 1

        accuracy = (true / (true + false)) * 100
        return accuracy

    def svm(self, X_train, y_train):

        opt_dict = {}
        # transform1 = X_train.shape[1]

        transforms = [[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1],
                      [1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, -1, -1]]

        all_data = []

        for featureset in X_train:
            for feature in featureset:
                all_data.append(feature)

        max_feature_value = max(all_data)
        min_feature_value = min(all_data)

        # print(max_feature_value)
        # print(min_feature_value)

        all_data = None

        step_sizes = [max_feature_value * 0.1]

        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = max_feature_value * 10

        # for x, y in zip(X_train, y_train):
        #     print(f'{x},{y}')

        for step in step_sizes:
            w = np.array([
                latest_optimum, latest_optimum, latest_optimum, latest_optimum
            ])
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (max_feature_value * b_range_multiple),
                                   max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transforamtion in transforms:
                        w_t = w * transforamtion
                        found_option = True

                        for x, y in zip(X_train, y_train):
                            yi = y
                            dot_product = np.dot(w_t, x)
                            mul = dot_product + b
                            # print(yi)
                            # print(dot_product)
                            # print(mul)
                            if not yi * mul >= 1:
                                found_option = False
                            if found_option:
                                opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                            # print(np.linalg.norm(w_t))
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')
                else:
                    w = w - step

        norms = sorted([n for n in opt_dict])
        opt_choice = opt_dict[norms[0]]
        w = opt_choice[0]
        b = opt_choice[1]
        latest_optimum = opt_choice[0][0] + step * 2

        print(w)
        print(b)

    def test_svm(self, features, w, b):
        #sign ((x.w) + b)
        classification = np.sign(np.dot(np.array(features), w) + b)
        print(classification)


def main():
    user_dataset_chosen = "banknote.txt"
    user_model_choice = "SVM"

    if user_model_choice == "KNN":
        model = MLChoice(user_model_choice, user_dataset_chosen)
        data = model.read_data_from_file()
        X_train, X_test, y_train, y_test = model.split_data(data, 0.2)
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()
        # knn(X_train, y_train, [1.5356, 9.1772, -2.2718, -0.73535], 3)
        if user_dataset_chosen == "banknote.txt":
            print(model.test_knn(X_train, y_train, X_test, y_test, 30))
        else:
            print(model.test_knn(X_train, y_train, X_test, y_test, 3))
    else:
        model = MLChoice(user_model_choice, user_dataset_chosen)
        data = model.read_data_from_file()
        X_train, X_test, y_train, y_test = model.split_data(data, 0.3)
        slk = SelectKBest(score_func=f_classif, k=4)
        X_train = slk.fit_transform(X_train, y_train)
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()
        y_new_train = []
        #M for Mine, #R for Rock
        #0 for authentic #1 for unauthentic
        # last_col_map = {'M': 1, 'R': -1, '0': 1, '1': -1}
        for i in range(0, len(y_train)):
            if y_train[i] == 'M':
                y_new_train.append(1)
            elif y_train[i] == 'R':
                y_new_train.append(-1)
            elif y_train[i] == 0:
                y_new_train
                
                .append(1)
            else:
                y_new_train.append(-1)

            # # data_to_train = []
            # for i, j in zip(X_train, y_train):
            #     data_to_train.append([i, j])

            model.svm(X_train, y_new_train)

            # for i in y_train:
            #     print(i)


if __name__ == "__main__":
    main()