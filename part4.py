"""Part 4: split gains, model votes, two boosting rounds and network sizes."""
import math
import numpy as np
from common import (classification_data, logistic_fit, sigmoid, knn_select,
                    knn_neighbours, metrics, write_results, write_csv)


def impurity(y, kind):
    if not len(y):
        return 0.0
    p = float(np.mean(y))
    probabilities = (p, 1-p)
    if kind == "entropy":
        return -sum(q * math.log2(q) for q in probabilities if q > 0)
    return 1 - sum(q*q for q in probabilities)


def split_masks(x):
    return [x[:, 5] == 1, x[:, 6] == 1, x[:, 2] > 0.5, x[:, 0] <= 24]


def split_gains(x, y):
    rows = []
    for i, mask in enumerate(split_masks(x), 1):
        gains = []
        for kind in ("entropy", "gini"):
            children = sum(len(side) * impurity(side, kind) for side in (y[mask], y[~mask])) / len(y)
            gains.append(impurity(y, kind) - children)
        rows.append([f"S{i}", int(mask.sum()), int(y[mask].sum()),
                     int((~mask).sum()), int(y[~mask].sum()), *gains])
    return rows


def adaboost():
    y = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    rules = [np.array([1, 1, -1, -1, -1, -1, -1, -1]),
             np.array([-1, 1, 1, -1, 1, -1, 1, -1])]
    weights = np.full(len(y), 1/len(y))
    result, rows, alphas = {}, [], []
    for t, h in enumerate(rules, 1):
        error = float(weights[h != y].sum())
        alpha = 0.5 * math.log((1-error)/error)
        unnormalised = weights * np.exp(-alpha*y*h)
        after = unnormalised / unnormalised.sum()
        result[f"round{t}_error"] = error
        result[f"round{t}_alpha"] = alpha
        for point in range(len(y)):
            rows.append([t, point+1, weights[point], unnormalised[point], after[point]])
        weights = after
        alphas.append(alpha)
    score = sum(alpha*h for alpha, h in zip(alphas, rules))
    result["ensemble_accuracy"] = float(np.mean(np.sign(score) == y))
    return result, rows


def architecture():
    def dense(inputs, outputs):
        return (inputs+1)*outputs
    def conv(kernel, depth, filters):
        return (kernel*kernel*depth+1)*filters
    def spatial(side, kernel, stride, padding):
        numerator = side-kernel+2*padding
        assert numerator % stride == 0
        return numerator//stride+1
    side1 = spatial(28, 5, 1, 2)
    pool1 = spatial(side1, 2, 2, 0)
    side2 = spatial(pool1, 5, 1, 0)
    pool2 = spatial(side2, 2, 2, 0)
    rows = [("mlp_hidden1", dense(28*28, 16)), ("mlp_hidden2", dense(16, 16)),
            ("mlp_output", dense(16, 10)), ("cnn_conv1", conv(5, 1, 6)),
            ("cnn_conv2", conv(5, 6, 16)), ("cnn_dense1", dense(pool2*pool2*16, 32)),
            ("cnn_dense2", dense(32, 10))]
    result = dict(mlp_total=sum(v for _, v in rows[:3]),
                  cnn_total=sum(v for _, v in rows[3:]),
                  fc_layer_params=dense(128*128*3, 1024),
                  conv_layer_params=conv(3, 3, 64),
                  rnn_total=128*64 + 128*128 + 16*128 + 128 + 16)
    shapes = [("input", "28 x 28 x 1", 0),
              ("conv1", f"{side1} x {side1} x 6", rows[3][1]),
              ("ReLU1", f"{side1} x {side1} x 6", 0),
              ("pool1", f"{pool1} x {pool1} x 6", 0),
              ("conv2", f"{side2} x {side2} x 16", rows[4][1]),
              ("ReLU2", f"{side2} x {side2} x 16", 0),
              ("pool2", f"{pool2} x {pool2} x 16", 0),
              ("flatten", str(pool2*pool2*16), 0),
              ("dense1", "32", rows[5][1]), ("ReLU3", "32", 0),
              ("dense2", "10", rows[6][1])]
    return result, rows, shapes


def run():
    x, y, tr, te, train, test = classification_data()
    a, b = y[tr], y[te]
    result = dict(root_entropy=impurity(a, "entropy"), root_gini=impurity(a, "gini"))
    splits = split_gains(x[tr], a)
    best_info = int(np.argmax([row[-2] for row in splits]))
    best_gini = int(np.argmax([row[-1] for row in splits]))
    result.update(best_split_by_info_gain=splits[best_info][0],
                  best_split_by_gini_gain=splits[best_gini][0])
    write_csv("part4_splits.csv", "split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain", splits)
    mask = split_masks(x[tr])[best_info]
    yes, no = float(a[mask].mean()), float(a[~mask].mean())
    result.update(stump_rate_yes=yes, stump_rate_no=no)
    stump = np.where(split_masks(x[te])[best_info], yes, no)
    w, bias, _ = logistic_fit(train, a)
    logistic = sigmoid(test @ w + bias)
    k, _ = knn_select(train, a)
    neighbours = knn_neighbours(train, a, test, k)
    knn_probability = neighbours.mean(axis=1)
    predictions = dict(logistic=logistic >= 0.3,
                       knn=neighbours.sum(axis=1) >= (k+1)//2,
                       stump=stump >= 0.3)
    predictions["hard_vote"] = np.sum(list(predictions.values()), axis=0) >= 2
    predictions["soft_vote"] = (logistic+knn_probability+stump)/3 >= 0.3
    voting = []
    for name, prediction in predictions.items():
        m = metrics(b, prediction)
        voting.append([name] + [m[key] for key in "tp fp tn fn precision recall f1".split()])
    write_csv("part4_voting.csv", "model,tp,fp,tn,fn,precision,recall,f1", voting)
    boosting, rounds = adaboost()
    result.update(boosting)
    write_csv("part4_adaboost.csv", "round,point,weight_before,weight_unnormalised,weight_after", rounds)
    result["oob_fraction"] = (1 - 1/len(tr))**len(tr)
    sizes, layers, _ = architecture()
    result.update(sizes)
    write_csv("part4_architecture.csv", "layer,parameters", layers)
    write_results(4, result)
    print(f"Part 4: {len(result)} keys; best split {splits[best_info][0]}")
    return result


if __name__ == "__main__":
    run()
