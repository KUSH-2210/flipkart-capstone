"""Part 2: pinned split, batch logistic regression, KNN and retention costs."""
import numpy as np
from common import (classification_data, logistic_fit, sigmoid, bce, metrics,
                    knn_select, knn_neighbours, cost, write_results, write_csv)


def run():
    _, y, tr, te, train, test = classification_data()
    a, b = y[tr], y[te]
    result = dict(train_rows=len(tr), train_positives=int(a.sum()),
                  test_rows=len(te), test_positives=int(b.sum()))
    w, bias, losses = logistic_fit(train, a)
    for epoch in (0, 1, 10, 100, 1000):
        result[f"loss_epoch_{epoch}"] = losses[epoch]
    result.update({f"weight_X{i+1}": v for i, v in enumerate(w)})
    result.update(bias=bias, test_bce=bce(test, b, w, bias))
    for eta in (0.01, 0.1, 1.0):
        _, _, history = logistic_fit(train, a, eta=eta, updates=200)
        result[f"loss_epoch_200_eta_{eta}"] = history[200]
    result.update({f"odds_ratio_X{i+1}": np.exp(v) for i, v in enumerate(w)})
    probability = sigmoid(test @ w + bias)
    fields = "tp fp tn fn accuracy precision recall f1 specificity".split()
    for tau in (0.5, 0.3):
        m = metrics(b, probability >= tau)
        result.update({f"tau_{tau}_{key}": m[key] for key in fields})
    result["blind_accuracy"] = float(np.mean(b == 0))
    sweep = []
    for k in range(1, 10):
        tau = k / 10
        m = metrics(b, probability >= tau)
        sweep.append([tau] + [m[key] for key in "tp fp tn fn tpr fpr".split()])
    write_csv("part2_thresholds.csv", "tau,tp,fp,tn,fn,tpr,fpr", sweep)
    best_k, cv = knn_select(train, a)
    result["knn_best_k"] = best_k
    neighbour_count = knn_neighbours(train, a, test, best_k).sum(axis=1)
    m = metrics(b, neighbour_count >= (best_k + 1)//2)
    result.update({f"knn_{key}": m[key] for key in fields[:-1]})
    write_csv("part2_knn_cv.csv", "K,mean_cv_f1", cv)
    train_probability = sigmoid(train @ w + bias)
    candidates = [(cost(a, train_probability, k / 20), k / 20) for k in range(1, 20)]
    train_cost, tau = min(candidates)
    result.update(cost_tau=tau, cost_train_at_tau=train_cost,
                  cost_test_at_tau=cost(b, probability, tau),
                  **{"cost_test_at_0.5": cost(b, probability, 0.5)},
                  cost_test_flag_nobody=400 * int(b.sum()))
    write_results(2, result)
    print(f"Part 2: {len(result)} keys; K={best_k}; training-selected tau={tau}")
    return result


if __name__ == "__main__":
    run()
