import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def svm(data):
    opt_dict = {}

    transforms = [[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1],
                  [1, 1, 1, 1], [-1, -1, -1, -1]]

    all_data = []
    for yi in data:
        for featureset in data[yi]:
            for feature in featureset:
                all_data.append(feature)

    max_feature_value = max(all_data)
    min_feature_value = min(all_data)

    all_data = None

    #support vectors yi(xi.w + b) = 1

    step_sizes = [
        max_feature_value * 0.1, max_feature_value * 0.01,
        max_feature_value * 0.001
    ]

    b_range_multiple = 5
    b_multiple = 5

    latest_optimum = max_feature_value * 10

    for step in step_sizes:
        print(f'for step {step}')
        w = np.array(
            [latest_optimum, latest_optimum, latest_optimum, latest_optimum])
        optimized = False

        while not optimized:

            for b in np.arange(-1 * (max_feature_value * b_range_multiple),
                               max_feature_value * b_range_multiple,
                               step * b_multiple):
                for transformation in transforms:
                    w_t = w * transformation
                    found_option = True

                    for i in data:
                        for xi in data[i]:
                            yi = i
                            if not yi * (np.dot(w_t, xi) + b) >= 1:
                                found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
            if w[0] < 0:
                optimized = True
                print('Optimized a step')
            else:
                w = w - step

        # opt_dict[np.linalg.norm(w)] = [w, b]
        norms = sorted([n for n in opt_dict])
        # print(opt_dict)
        opt_choice = opt_dict[norms[0]]
        w = opt_choice[0]
        b = opt_choice[1]
        latest_optimum = opt_choice[0][0] + step * 2

    w_result = w[0]
    b_result = b

    return w_result, b_result


def test_svm(w, b, test_data, no_of_features):
    true = 0
    false = 0

    for i in test_data:
        featureset = i[0:no_of_features - 1]
        classification = np.sign(np.dot(np.array(featureset), w) + b)
        print(classification)


def main():

    #read the data
    data = pd.read_csv("banknote.txt", sep=",", header=None)
    no_of_features = len(data.columns)
    train, test = train_test_split(data, test_size=0.3, random_state=42)
    train = train.to_numpy()
    test = test.to_numpy()
    #0: -1, 1: 1
    data_dict = {-1: [], 1: []}
    test_data_dict = {-1: [], 1: []}
    count = 0
    for i in train:
        if (i[no_of_features - 1] == 0):
            data_to_append = i[0:no_of_features - 1]
            data_dict[-1].append(data_to_append)
        elif (i[no_of_features - 1] == 1):
            data_to_append = i[0:no_of_features - 1]
            data_dict[1].append(data_to_append)
        else:
            print("data not available")
    for i in test:
        if (i[no_of_features - 1] == 0):
            data_to_append = i[0:no_of_features - 1]
            test_data_dict[-1].append(data_to_append)
        elif (i[no_of_features - 1] == 1):
            data_to_append = i[0:no_of_features - 1]
            test_data_dict[1].append(data_to_append)
        else:
            print("data not available")

    # transformation = [-1, 1, 1, 1]
    # w = np.array([2, 2, 2, 2])
    # b = 2
    # w_t = w * transformation
    # for i in data_dict:
    #     for xi in data_dict[i]:
    #         yi = i
    #         print(yi * (np.dot(w_t, xi) + b))

    # for i in data_dict:
    #     for x in data_dict[i]:
    #         count = count + 1
    #         print(f'{i}: {x[0:no_of_features-1]}')

    # print(count)
    #split the data
    #
    # X = data.iloc[:, 0:no_of_features - 1]
    # y = data.iloc[:, no_of_features - 1]

    # X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                     y,
    #                                                     test_size=0.3,
    #
    #                                                random_state=42)

    # X_train.to_numpy()
    # y_train.to_numpy()

    w, b = svm(data_dict)
    print(w)
    print(b)

    test_svm(w, b, test, no_of_features)
    # classification = int(
    #     np.sign(np.dot(np.array([-3.3604, -0.32696, 2.1324, 0.6017]), w) + b))
    # print(classification[0])


if __name__ == "__main__":
    main()