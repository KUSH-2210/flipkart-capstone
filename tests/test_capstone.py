"""Reproducibility, independent arithmetic, schemas and pinned conventions.

Expected model answers are never stored here. Compare calculated quantities,
mathematical invariants and fresh-script artifacts; only supplied constants,
schemas and the original generator fingerprint are fixed.
"""
from pathlib import Path
import csv
import hashlib
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common
import part1
import part3
import part4


def results(part):
    pairs = [line.split(" = ") for line in (ROOT / "outputs" / f"part{part}_results.txt").read_text().splitlines()]
    if len(dict(pairs)) != len(pairs):
        raise AssertionError("Duplicate results key")
    return dict(pairs)


def csv_rows(name):
    with (ROOT / "outputs" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


CSV_SCHEMAS = {
    "part2_thresholds.csv": ("tau,tp,fp,tn,fn,tpr,fpr", 9, "tp fp tn fn"),
    "part2_knn_cv.csv": ("K,mean_cv_f1", 11, "K"),
    "part3_segments.csv": ("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value", 3, "size"),
    "part3_hierarchical.csv": ("linkage,merge_no,cluster_a,cluster_b,height", 15, "merge_no"),
    "part4_splits.csv": ("split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain", 4,
                         "n_yes positives_yes n_no positives_no"),
    "part4_voting.csv": ("model,tp,fp,tn,fn,precision,recall,f1", 5, "tp fp tn fn"),
    "part4_adaboost.csv": ("round,point,weight_before,weight_unnormalised,weight_after", 16, "round point"),
    "part4_architecture.csv": ("layer,parameters", 7, "parameters"),
}


def required_keys():
    p1 = set(("n mean_value sd_value se_value ci95_low ci95_high z_stat reject_h0 n_required "
              "reg_slope reg_intercept tss ess rss tss_equals_ess_plus_rss r2 rmse mae "
              "gd_J0 gd_J1 gd_J10 gd_J100 gd_J500 gd_slope gd_intercept gd_slope_gap "
              "resid_sd_lower_raw resid_sd_upper_raw resid_sd_ratio_raw loglog_intercept loglog_slope "
              "loglog_r2 loglog_rmse resid_sd_lower_log resid_sd_upper_log resid_sd_ratio_log "
              "mr_r2 mr_adj_r2 loglog_adj_r2").split())
    p1.update(f"mr_beta_{i}" for i in range(8))
    p2 = set(("train_rows train_positives test_rows test_positives bias test_bce blind_accuracy "
              "knn_best_k cost_tau cost_train_at_tau cost_test_at_tau cost_test_at_0.5 cost_test_flag_nobody").split())
    p2.update(f"loss_epoch_{i}" for i in (0, 1, 10, 100, 1000))
    p2.update(f"loss_epoch_200_eta_{eta}" for eta in (0.01, 0.1, 1.0))
    p2.update(f"{prefix}_X{i}" for prefix in ("weight", "odds_ratio") for i in range(1, 8))
    fields = "tp fp tn fn accuracy precision recall f1".split()
    p2.update(f"tau_{tau}_{field}" for tau in (0.5, 0.3) for field in fields + ["specificity"])
    p2.update(f"knn_{field}" for field in fields)
    p3 = set(("elbow_k single_clusters_at_cut complete_clusters_at_cut average_clusters_at_cut "
              "eigenvalue_sum components_for_80pct").split())
    p3.update(f"wcss_k{k}" for k in range(2, 7))
    p3.update(f"{prefix}_{name}" for prefix in ("size_segment", "mean_pc1_segment") for name in "ABC")
    p3.update(f"{prefix}_{i}" for prefix in ("eigenvalue", "explained_ratio", "cumulative_ratio") for i in range(1, 6))
    p3.update(f"pc{pc}_loading_X{i}" for pc in (1, 2) for i in range(1, 6))
    p3.update(f"row{r}_pc{pc}" for r in (1, 2, 3) for pc in (1, 2))
    p4 = set(("root_entropy root_gini best_split_by_info_gain best_split_by_gini_gain "
              "stump_rate_yes stump_rate_no round1_error round1_alpha round2_error round2_alpha "
              "ensemble_accuracy oob_fraction mlp_total cnn_total fc_layer_params conv_layer_params rnn_total").split())
    return {1: p1, 2: p2, 3: p3, 4: p4}


class CapstoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y, cls.value = common.load_data()

    def close_output(self, actual, printed):
        self.assertAlmostEqual(float(actual), float(printed), delta=0.00005001)

    def test_dataset_and_unmodified_generator(self):
        # Fingerprint recorded before any edits and compared with the official script.
        digest = hashlib.sha256((ROOT / "generate_data.py").read_bytes()).hexdigest()
        self.assertEqual(digest, "1578e51eb70624bf9a897ecc3dc20d6584d6e38ac4ffacfead1149a46de8dd63")
        with (ROOT / "data" / "records.csv").open(newline="") as f:
            reader = csv.reader(f)
            self.assertEqual(next(reader), ["record_id", *common.FEATURES, "churned", "order_value_inr_hundreds"])
            rows = list(reader)
        self.assertEqual(len(rows), 800)
        self.assertEqual([r[0] for r in rows], [f"E{i+1:04d}" for i in range(len(rows))])
        self.assertTrue(all(len(r) == 10 for r in rows))
        self.assertTrue(np.isfinite(self.x).all())
        self.assertTrue(np.all(self.value > 0))
        self.assertEqual(set(np.unique(self.y)), {0, 1})
        self.assertEqual(int(self.y.sum()), 170)

    def test_exact_result_keys_and_print_formats(self):
        integer_keys = {"n", "n_required", "knn_best_k", "elbow_k", "components_for_80pct",
                        "mlp_total", "cnn_total", "fc_layer_params", "conv_layer_params", "rnn_total"}
        boolean_keys = {"reject_h0", "tss_equals_ess_plus_rss"}
        text_keys = {"best_split_by_info_gain", "best_split_by_gini_gain"}
        for part, keys in required_keys().items():
            r = results(part)
            self.assertEqual(set(r), keys)
            for key, value in r.items():
                if key in boolean_keys:
                    self.assertIn(value, ("yes", "no"))
                elif key in text_keys:
                    self.assertIn(value, ("S1", "S2", "S3", "S4"))
                elif (key in integer_keys or key.startswith(("train_", "test_rows", "test_positives", "size_segment_", "cost_train_", "cost_test_"))
                      or key.endswith(("_tp", "_fp", "_tn", "_fn", "_clusters_at_cut"))):
                    self.assertRegex(value, r"^-?\d+$")
                else:
                    self.assertRegex(value, r"^-?\d+\.\d{4}$")

    def test_csv_schemas_rows_and_formats(self):
        text = {"segment", "linkage", "cluster_a", "cluster_b", "split", "model", "layer"}
        for name, (header, count, integers) in CSV_SCHEMAS.items():
            rows = csv_rows(name)
            self.assertEqual(list(rows[0]), header.split(","))
            self.assertEqual(len(rows), count)
            for row in rows:
                for key, value in row.items():
                    if key in text:
                        self.assertTrue(value)
                    elif key in integers.split():
                        self.assertRegex(value, r"^\d+$")
                    else:
                        self.assertRegex(value, r"^-?\d+\.\d{4}$")

    def test_part1_independent_scalar_formulas(self):
        r = results(1)
        x = self.x[:, 3].tolist()
        y = self.value.tolist()
        n = len(y)
        mean, sd = statistics.mean(y), statistics.stdev(y)
        se = sd/math.sqrt(n)
        expected = dict(n=n, mean_value=mean, sd_value=sd, se_value=se,
                        ci95_low=mean-1.96*se, ci95_high=mean+1.96*se,
                        z_stat=(mean-8)/se, n_required=math.ceil((1.96*sd/0.5)**2))
        self.assertEqual(r["reject_h0"], "yes" if abs(expected["z_stat"]) > 1.96 else "no")
        def line(a, b):
            am, bm = statistics.mean(a), statistics.mean(b)
            slope = sum((u-am)*(v-bm) for u, v in zip(a, b))/sum((u-am)**2 for u in a)
            intercept = bm-slope*am
            fitted = [intercept+slope*u for u in a]
            residual = [v-p for v, p in zip(b, fitted)]
            rss = sum(e*e for e in residual)
            tss = sum((v-bm)**2 for v in b)
            return intercept, slope, fitted, residual, rss, tss
        intercept, slope, fitted, residual, rss, tss = line(x, y)
        ess = sum((p-mean)**2 for p in fitted)
        expected.update(reg_slope=slope, reg_intercept=intercept, tss=tss, rss=rss, ess=ess,
                        r2=1-rss/tss, rmse=math.sqrt(rss/n), mae=statistics.mean(map(abs, residual)))
        self.assertEqual(r["tss_equals_ess_plus_rss"], "yes" if abs(tss-ess-rss) < 1e-6*tss else "no")
        for suffix, a, b in (("raw", x, y), ("log", list(map(math.log, x)), list(map(math.log, y)))):
            c, m, pred, errors, ss, total = line(a, b)
            order = sorted(range(n), key=lambda i: (pred[i], i))
            lower = statistics.stdev(errors[i] for i in order[:n//2])
            upper = statistics.stdev(errors[i] for i in order[n//2:])
            expected.update({f"resid_sd_lower_{suffix}": lower, f"resid_sd_upper_{suffix}": upper,
                             f"resid_sd_ratio_{suffix}": upper/lower})
            if suffix == "log":
                expected.update(loglog_intercept=c, loglog_slope=m, loglog_r2=1-ss/total,
                                loglog_rmse=math.sqrt(ss/n), loglog_adj_r2=1-(ss/total)*(n-1)/(n-2))
        z = [(u-statistics.mean(x))/statistics.stdev(x) for u in x]
        m, c = 0.0, 0.0
        expected["gd_J0"] = statistics.mean(v*v for v in y)
        for epoch in range(1, 501):
            errors = [v-m*u-c for u, v in zip(z, y)]
            m += 0.2*statistics.mean(u*e for u, e in zip(z, errors))
            c += 0.2*statistics.mean(errors)
            if epoch in (1, 10, 100, 500):
                expected[f"gd_J{epoch}"] = statistics.mean((v-m*u-c)**2 for u, v in zip(z, y))
        expected.update(gd_slope=m/statistics.stdev(x), gd_intercept=c-m*statistics.mean(x)/statistics.stdev(x),
                        gd_slope_gap=abs(m/statistics.stdev(x)-slope))
        design = np.column_stack((np.ones(n), np.log(x), self.x[:, 0:3], self.x[:, 4:7]))
        beta = np.linalg.solve(design.T@design, design.T@np.log(y))
        errors = np.log(y)-design@beta
        np.testing.assert_allclose(design.T@errors, 0, atol=1e-8)
        for i, coefficient in enumerate(beta):
            expected[f"mr_beta_{i}"] = coefficient
        mr_r2 = 1-float(errors@errors)/sum((math.log(v)-statistics.mean(map(math.log, y)))**2 for v in y)
        expected.update(mr_r2=mr_r2, mr_adj_r2=1-(1-mr_r2)*(n-1)/(n-8))
        self.assertEqual(set(expected), set(r)-{"reject_h0", "tss_equals_ess_plus_rss"})
        for key, actual in expected.items():
            self.close_output(actual, r[key])

    def test_split_and_training_only_scaling(self):
        _, y, tr, te, train, test = common.classification_data()
        chosen = []
        for label in (0, 1):
            members = [i for i, v in enumerate(y) if v == label]
            chosen.extend(members[:len(members)*7//10])
        self.assertEqual(tr.tolist(), sorted(chosen))
        self.assertEqual(sorted(tr.tolist()+te.tolist()), list(range(len(y))))
        self.assertFalse(set(tr) & set(te))
        for j in range(7):
            mean = statistics.mean(self.x[tr, j])
            sd = statistics.stdev(self.x[tr, j])
            np.testing.assert_allclose(train[:, j], (self.x[tr, j]-mean)/sd, atol=1e-14)
            np.testing.assert_allclose(test[:, j], (self.x[te, j]-mean)/sd, atol=1e-14)
        np.testing.assert_allclose(train.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(train.std(axis=0, ddof=1), 1, atol=1e-14)

    def test_logistic_gradient_and_losses_independently(self):
        _, y, tr, te, x, test = common.classification_data()
        r = results(2)
        def reference(eta, updates):
            weights = np.zeros(7)
            bias = 0.0
            def loss(matrix, labels):
                probabilities = 1/(1+np.exp(-(matrix@weights+bias)))
                return float(-np.mean(labels*np.log(probabilities)+(1-labels)*np.log(1-probabilities)))
            history = {0: loss(x, y[tr])}
            for epoch in range(1, updates+1):
                errors = 1/(1+np.exp(-(x@weights+bias)))-y[tr]
                gradient = np.array([np.mean(errors*x[:, j]) for j in range(7)])
                weights = weights-eta*gradient
                bias = bias-eta*float(np.mean(errors))
                if epoch in (1, 10, 100, updates):
                    history[epoch] = loss(x, y[tr])
            return weights, bias, history, loss(test, y[te])
        w, b, history, test_loss = reference(0.1, 1000)
        for epoch, v in history.items():
            self.close_output(v, r[f"loss_epoch_{epoch}"])
        for i, v in enumerate(w, 1):
            self.close_output(v, r[f"weight_X{i}"])
            self.close_output(math.exp(v), r[f"odds_ratio_X{i}"])
        self.close_output(b, r["bias"])
        self.close_output(test_loss, r["test_bce"])
        for eta in (0.01, 0.1, 1.0):
            self.close_output(reference(eta, 200)[2][200], r[f"loss_epoch_200_eta_{eta}"])

    def test_metrics_thresholds_and_training_cost_choice(self):
        _, y, tr, te, train, test = common.classification_data()
        r = results(2)
        w, b, _ = common.logistic_fit(train, y[tr])
        pt, pv = common.sigmoid(train@w+b), common.sigmoid(test@w+b)
        for row in csv_rows("part2_thresholds.csv"):
            m = common.metrics(y[te], pv >= float(row["tau"]))
            for key in ("tp", "fp", "tn", "fn", "tpr", "fpr"):
                self.close_output(m[key], row[key])
        for tau in (0.3, 0.5):
            for key in "tp fp tn fn accuracy precision recall f1 specificity".split():
                self.close_output(common.metrics(y[te], pv >= tau)[key], r[f"tau_{tau}_{key}"])
        def money(labels, probability, tau):
            return sum(50 if label == 0 and p >= tau else 400 if label == 1 and p < tau else 0
                       for label, p in zip(labels, probability))
        choices = [(money(y[tr], pt, k/20), k/20) for k in range(1, 20)]
        train_cost, tau = min(choices)
        self.close_output(tau, r["cost_tau"])
        self.assertEqual(train_cost, int(r["cost_train_at_tau"]))
        self.assertEqual(money(y[te], pv, tau), int(r["cost_test_at_tau"]))
        self.assertEqual(money(y[te], pv, 0.5), int(r["cost_test_at_0.5"]))
        self.assertEqual(400*int(y[te].sum()), int(r["cost_test_flag_nobody"]))
        self.close_output(float(np.mean(y[te] == 0)), r["blind_accuracy"])
        zero = common.metrics(np.zeros(3), np.zeros(3))
        self.assertEqual(zero["precision"], 0)
        self.assertEqual(zero["recall"], 0)
        self.assertEqual(zero["f1"], 0)
        self.assertTrue(np.isfinite(common.sigmoid(np.array([-1000, 1000]))).all())

    def test_knn_reference_folds_and_distance_ties(self):
        _, y, tr, te, train, test = common.classification_data()
        labels = y[tr]
        ks = list(range(1, 22, 2))
        scores = {k: [] for k in ks}
        # Independent row-by-row ordering uses Python's explicit tuple tie break.
        def neighbours(reference, targets, queries):
            answer = []
            for query in queries:
                distances = np.sum((reference-query)**2, axis=1)
                order = sorted(range(len(reference)), key=lambda j: (float(distances[j]), j))
                answer.append(targets[order[:max(ks)]] )
            return np.array(answer)
        for fold in range(5):
            held = np.arange(len(labels)) % 5 == fold
            ranked = neighbours(train[~held], labels[~held], train[held])
            for k in ks:
                predicted = ranked[:, :k].sum(axis=1) > k/2
                tp = int(np.sum(predicted & (labels[held] == 1)))
                fp = int(np.sum(predicted & (labels[held] == 0)))
                fn = int(np.sum(~predicted & (labels[held] == 1)))
                scores[k].append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
        means = {k: statistics.mean(scores[k]) for k in ks}
        for row in csv_rows("part2_knn_cv.csv"):
            self.close_output(means[int(row["K"])], row["mean_cv_f1"])
        chosen = min(ks, key=lambda k: (-means[k], k))
        r = results(2)
        self.assertEqual(chosen, int(r["knn_best_k"]))
        prediction = neighbours(train, labels, test)[:, :chosen].sum(axis=1) > chosen/2
        for key in "tp fp tn fn accuracy precision recall f1".split():
            self.close_output(common.metrics(y[te], prediction)[key], r[f"knn_{key}"])
        tied = common.knn_neighbours(np.array([[-1.], [1.]]), np.array([1, 0]), np.array([[0.]]), 1)
        self.assertEqual(tied.tolist(), [[1]])

    def test_kmeans_segments_and_pca_invariants(self):
        x = self.x[:, :5]
        z, _, _ = common.standardise(x)
        r = results(3)
        wcss = {}
        for k in range(2, 7):
            labels, centres, ss, passes = part3.kmeans(z, k)
            assigned = np.argmin(np.sum((z[:, None]-centres[None, :])**2, axis=2), axis=1)
            np.testing.assert_array_equal(labels, assigned)
            for c in range(k):
                if np.any(labels == c):
                    np.testing.assert_allclose(centres[c], z[labels == c].mean(axis=0))
            self.close_output(ss, r[f"wcss_k{k}"])
            wcss[k] = ss
            self.assertGreater(passes, 1)
        elbow = min(k for k in (3, 4, 5) if wcss[k]-wcss[k+1] < (wcss[k-1]-wcss[k])/2)
        self.assertEqual(elbow, int(r["elbow_k"]))
        labels = part3.kmeans(z, 3)[0]
        ordered = sorted(range(3), key=lambda c: x[labels == c, 2].mean())
        eigenvalues, vectors, scores = part3.pca(z)
        covariance = (z-z.mean(axis=0)).T@(z-z.mean(axis=0))/(len(z)-1)
        np.testing.assert_allclose(vectors.T@vectors, np.eye(5), atol=1e-14)
        np.testing.assert_allclose(covariance@vectors, vectors*eigenvalues, atol=1e-14)
        np.testing.assert_allclose(vectors@np.diag(eigenvalues)@vectors.T, covariance, atol=1e-14)
        self.assertTrue(np.all(eigenvalues[:-1] >= eigenvalues[1:]))
        for j in range(5):
            self.assertGreater(vectors[np.argmax(np.abs(vectors[:, j])), j], 0)
            self.close_output(eigenvalues[j], r[f"eigenvalue_{j+1}"])
            self.close_output(eigenvalues[j]/eigenvalues.sum(), r[f"explained_ratio_{j+1}"])
            self.close_output(eigenvalues[:j+1].sum()/eigenvalues.sum(), r[f"cumulative_ratio_{j+1}"])
        self.close_output(np.trace(covariance), r["eigenvalue_sum"])
        self.assertEqual(int(np.searchsorted(np.cumsum(eigenvalues)/eigenvalues.sum(), 0.8))+1,
                         int(r["components_for_80pct"]))
        for pc in range(2):
            for j in range(5):
                self.close_output(vectors[j, pc], r[f"pc{pc+1}_loading_X{j+1}"])
            for row in range(3):
                self.close_output(scores[row, pc], r[f"row{row+1}_pc{pc+1}"])
        rows = csv_rows("part3_segments.csv")
        for name, c, row in zip("ABC", ordered, rows):
            mask = labels == c
            self.assertEqual(row["segment"], name)
            self.assertEqual(int(row["size"]), int(mask.sum()))
            self.assertEqual(int(r[f"size_segment_{name}"]), int(mask.sum()))
            for j in range(5):
                self.close_output(x[mask, j].mean(), row[f"mean_X{j+1}"])
            self.close_output(self.y[mask].mean(), row["outcome_rate"])
            self.close_output(self.value[mask].mean(), row["mean_value"])
            self.close_output(scores[mask, 0].mean(), r[f"mean_pc1_segment_{name}"])
        self.assertEqual(sum(int(row["size"]) for row in rows), len(x))
        # Duplicate initial centres exercise the empty-cluster retention rule.
        _, centres, _, _ = part3.kmeans(np.zeros((6, 2)), 3)
        np.testing.assert_array_equal(centres, np.zeros((3, 2)))

    def test_hierarchical_member_distances_and_cuts(self):
        r = results(3)
        for linkage in ("single", "complete", "average"):
            partition = {frozenset([i]) for i in range(6)}
            heights, partitions = [], []
            def members(text):
                return frozenset(int(v[1:])-1 for v in text.split("+"))
            reduce = {"single": min, "complete": max, "average": statistics.mean}[linkage]
            def distance(a, b):
                return reduce(float(part3.DISSIMILARITY[i, j]) for i in a for j in b)
            for row in [row for row in csv_rows("part3_hierarchical.csv") if row["linkage"] == linkage]:
                a, b = members(row["cluster_a"]), members(row["cluster_b"])
                self.assertIn(a, partition)
                self.assertIn(b, partition)
                self.assertLess(min(a), min(b))
                smallest = min(distance(u, v) for u in partition for v in partition if u != v)
                self.assertAlmostEqual(distance(a, b), smallest)
                self.close_output(smallest, row["height"])
                heights.append(smallest)
                partition = (partition-{a, b}) | {a | b}
                partitions.append(partition.copy())
            largest_gap = max(range(4), key=lambda i: heights[i+1]-heights[i])
            self.assertEqual(len(partitions[largest_gap]), int(r[f"{linkage}_clusters_at_cut"]))

    def test_tree_gains_votes_and_architecture(self):
        x, y, tr, te, train, test = common.classification_data()
        r = results(4)
        for kind, key in (("entropy", "root_entropy"), ("gini", "root_gini")):
            self.close_output(part4.impurity(y[tr], kind), r[key])
        splits = part4.split_gains(x[tr], y[tr])
        for calculated, stored in zip(splits, csv_rows("part4_splits.csv")):
            for value, key in zip(calculated[1:], list(stored)[1:]):
                self.close_output(value, stored[key])
        info = max(range(4), key=lambda i: splits[i][-2])
        gini = max(range(4), key=lambda i: splits[i][-1])
        self.assertEqual(r["best_split_by_info_gain"], f"S{info+1}")
        self.assertEqual(r["best_split_by_gini_gain"], f"S{gini+1}")
        mask = part4.split_masks(x[tr])[info]
        yes, no = y[tr][mask].mean(), y[tr][~mask].mean()
        self.close_output(yes, r["stump_rate_yes"])
        self.close_output(no, r["stump_rate_no"])
        w, b, _ = common.logistic_fit(train, y[tr])
        logistic = common.sigmoid(test@w+b)
        k, _ = common.knn_select(train, y[tr])
        knn = common.knn_neighbours(train, y[tr], test, k).mean(axis=1)
        stump = np.where(part4.split_masks(x[te])[info], yes, no)
        base = np.array([logistic >= 0.3, knn >= 0.5, stump >= 0.3])
        predictions = [base[0], base[1], base[2], base.sum(axis=0) >= 2, (logistic+knn+stump)/3 >= 0.3]
        for row, name, pred in zip(csv_rows("part4_voting.csv"), ["logistic", "knn", "stump", "hard_vote", "soft_vote"], predictions):
            self.assertEqual(row["model"], name)
            for key in "tp fp tn fn precision recall f1".split():
                self.close_output(common.metrics(y[te], pred)[key], row[key])
        self.close_output(math.exp(len(tr)*math.log1p(-1/len(tr))), r["oob_fraction"])
        sizes, layers, shapes = part4.architecture()
        for key, value in sizes.items():
            self.assertEqual(value, int(r[key]))
        for (name, parameters), row in zip(layers, csv_rows("part4_architecture.csv")):
            self.assertEqual(row["layer"], name)
            self.assertEqual(int(row["parameters"]), parameters)
        self.assertEqual(sum(row[2] for row in shapes), sizes["cnn_total"])

    def test_adaboost_independent_scalar_updates(self):
        r = results(4)
        y = [1, 1, 1, -1, -1, -1, 1, -1]
        rules = [[1, 1, -1, -1, -1, -1, -1, -1], [-1, 1, 1, -1, 1, -1, 1, -1]]
        weights = [1/len(y)]*len(y)
        rows = csv_rows("part4_adaboost.csv")
        alphas = []
        for t, h in enumerate(rules, 1):
            error = sum(w for w, target, prediction in zip(weights, y, h) if target != prediction)
            alpha = math.log((1-error)/error)/2
            raw = [w*math.exp(-alpha*target*prediction) for w, target, prediction in zip(weights, y, h)]
            after = [w/sum(raw) for w in raw]
            self.close_output(error, r[f"round{t}_error"])
            self.close_output(alpha, r[f"round{t}_alpha"])
            for i in range(len(y)):
                row = rows[(t-1)*len(y)+i]
                self.assertEqual((int(row["round"]), int(row["point"])), (t, i+1))
                for v, key in zip((weights[i], raw[i], after[i]), ("weight_before", "weight_unnormalised", "weight_after")):
                    self.close_output(v, row[key])
            self.assertAlmostEqual(sum(after), 1)
            weights = after
            alphas.append(alpha)
        correct = sum((1 if sum(a*h[i] for a, h in zip(alphas, rules)) > 0 else -1) == y[i] for i in range(len(y)))
        self.close_output(correct/len(y), r["ensemble_accuracy"])

    def test_readme_structure_memo_and_links(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[0], "Domain: E-commerce")
        self.assertEqual(re.findall(r"^## (.+)$", text, re.M),
                         ["How to run", "Part 1 findings", "Part 2 findings", "Part 3 findings",
                          "Part 4 findings", "Model-choice memo", "Limitations"])
        memo = text.split("## Model-choice memo\n", 1)[1].split("## Limitations", 1)[0]
        self.assertLessEqual(len(memo.split()), 300)
        self.assertNotRegex(text, r"https?://|\$\d")
        limitations = text.split("## Limitations", 1)[1]
        self.assertGreaterEqual(len(re.findall(r"^- ", limitations, re.M)), 3)

    def test_clean_rerun_matches_every_committed_artifact(self):
        names = [f"part{p}_results.txt" for p in range(1, 5)] + list(CSV_SCHEMAS)
        expected = {name: (ROOT/"outputs"/name).read_bytes() for name in names}
        self.assertEqual(set(p.name for p in (ROOT/"outputs").iterdir()), set(names))
        with tempfile.TemporaryDirectory(prefix="flipkart-repro-") as directory:
            work = Path(directory)
            for source in ROOT.glob("*.py"):
                shutil.copyfile(source, work/source.name)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            # Importing all helpers/Parts must create no data or output artifacts.
            subprocess.run([sys.executable, "-c", "import common,part1,part2,part3,part4"],
                           cwd=work, env=env, check=True, capture_output=True, text=True)
            self.assertFalse((work/"outputs").exists())
            for script in ("generate_data.py", "part1.py", "part2.py", "part3.py", "part4.py"):
                completed = subprocess.run([sys.executable, script], cwd=work, env=env,
                                           check=True, capture_output=True, text=True)
                self.assertFalse(completed.stderr, completed.stderr)
            generated = (work/"data"/"records.csv").read_bytes()
            self.assertEqual(hashlib.md5(generated).hexdigest(), common.DATA_MD5)
            self.assertEqual(generated, (ROOT/"data"/"records.csv").read_bytes())
            self.assertEqual((work/"generate_data.py").read_bytes(), (ROOT/"generate_data.py").read_bytes())
            for name, original in expected.items():
                self.assertEqual((work/"outputs"/name).read_bytes(), original, name)


if __name__ == "__main__":
    unittest.main()
