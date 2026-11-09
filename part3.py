"""Part 3: deterministic K-means, member-pair linkage and seven-step PCA."""
import numpy as np
from common import load_data, standardise, write_results, write_csv

# Supplied dissimilarities, not model results.
DISSIMILARITY = np.array([
    [0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0]])


def kmeans(x, k):
    centroids = x[np.arange(k) * (len(x)//k)].copy()
    previous = np.full(len(x), -1)
    passes = 0
    while True:
        squared = np.sum((x[:, None, :] - centroids[None, :, :])**2, axis=2)
        labels = np.argmin(squared, axis=1)  # first minimum wins ties
        passes += 1
        if np.array_equal(labels, previous):
            break
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
        previous = labels.copy()
    wcss = float(np.sum((x - centroids[labels])**2))
    return labels, centroids, wcss, passes


def cluster_name(members):
    return "+".join(f"E{i+1}" for i in sorted(members))


def hierarchical(table, linkage):
    clusters = [(i,) for i in range(len(table))]
    rows, partitions = [], []
    reduce = {"single": np.min, "complete": np.max, "average": np.mean}[linkage]
    while len(clusters) > 1:
        candidates = []
        for i, a in enumerate(clusters):
            for j in range(i+1, len(clusters)):
                b = clusters[j]
                height = float(reduce(table[np.ix_(a, b)]))
                candidates.append((height, i, j))
        height, i, j = min(candidates)
        a, b = sorted((clusters[i], clusters[j]), key=min)
        rows.append([linkage, len(rows)+1, cluster_name(a), cluster_name(b), height])
        merged = tuple(sorted(a + b))
        clusters = [c for index, c in enumerate(clusters) if index not in (i, j)] + [merged]
        clusters.sort(key=min)
        partitions.append(clusters.copy())
    heights = np.array([row[-1] for row in rows])
    merges_below = int(np.argmax(np.diff(heights))) + 1
    return rows, partitions[merges_below-1]


def pca(x):
    # 1: mean; 2: centre; 3: sample covariance; 4: eigendecomposition.
    mean = x.mean(axis=0)
    centred = x - mean
    covariance = centred.T @ centred / (len(x)-1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # 5: descending order and pinned sign; 6: retain two; 7: project.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    for col in range(eigenvectors.shape[1]):
        largest = np.argmax(np.abs(eigenvectors[:, col]))
        if eigenvectors[largest, col] < 0:
            eigenvectors[:, col] *= -1
    projection = centred @ eigenvectors[:, :2]
    return eigenvalues, eigenvectors, projection


def run():
    x, outcome, value = load_data()
    x = x[:, :5]
    z, _, _ = standardise(x)
    result, solutions = {}, {}
    for k in range(2, 7):
        solutions[k] = kmeans(z, k)
        result[f"wcss_k{k}"] = solutions[k][2]
        print(f"K={k}: {solutions[k][3]} assignment passes")
    result["elbow_k"] = next(k for k in (3, 4, 5)
                             if result[f"wcss_k{k}"]-result[f"wcss_k{k+1}"] <
                             0.5*(result[f"wcss_k{k-1}"]-result[f"wcss_k{k}"]))
    labels = solutions[3][0]
    ordered = sorted(range(3), key=lambda c: x[labels == c, 2].mean())
    masks, segments = {}, []
    for name, cluster in zip("ABC", ordered):
        mask = labels == cluster
        masks[name] = mask
        size = int(mask.sum())
        result[f"size_segment_{name}"] = size
        segments.append([name, size, *x[mask].mean(axis=0),
                         float(outcome[mask].mean()), float(value[mask].mean())])
    write_csv("part3_segments.csv",
              "segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value", segments)
    merges = []
    for linkage in ("single", "complete", "average"):
        rows, partition = hierarchical(DISSIMILARITY, linkage)
        merges.extend(rows)
        result[f"{linkage}_clusters_at_cut"] = len(partition)
    write_csv("part3_hierarchical.csv", "linkage,merge_no,cluster_a,cluster_b,height", merges)
    eigenvalues, vectors, projection = pca(z)
    total = float(eigenvalues.sum())
    ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)
    result["eigenvalue_sum"] = total
    for prefix, values in (("eigenvalue", eigenvalues), ("explained_ratio", ratios),
                           ("cumulative_ratio", cumulative)):
        for i, v in enumerate(values, 1):
            result[f"{prefix}_{i}"] = v
    result["components_for_80pct"] = int(np.flatnonzero(cumulative >= 0.8)[0]) + 1
    for pc in range(2):
        for j in range(5):
            result[f"pc{pc+1}_loading_X{j+1}"] = vectors[j, pc]
    for row in range(3):
        for pc in range(2):
            result[f"row{row+1}_pc{pc+1}"] = projection[row, pc]
    for name, mask in masks.items():
        result[f"mean_pc1_segment_{name}"] = float(projection[mask, 0].mean())
    write_results(3, result)
    return result


if __name__ == "__main__":
    run()
