import hashlib
from fractions import Fraction
from ..values import integer, fraction, rational, reject, fingerprint

DISTANCE_METRIC = "squared_euclidean_on_unit_total_shares"


class SeededRandom:
    """Deterministic SHA-256 based PRNG so a fixed seed always replays identically."""

    def __init__(self, seed):
        integer(seed, 0, 2**63 - 1)
        self.state = str(seed).encode("ascii")
        self.counter = 0

    def next_uint(self):
        self.counter += 1
        digest = hashlib.sha256(
            self.state + b"#" + str(self.counter).encode("ascii")
        ).digest()
        return int.from_bytes(digest[:8], "big")

    def uniform(self):
        return Fraction(self.next_uint(), 1 << 64)

    def weighted_choice(self, weights):
        total = sum(weights)
        if total <= 0:
            # All points coincide with existing centers; choose deterministically.
            return 0
        target = self.uniform() * total
        cursor = Fraction(0)
        for index, weight in enumerate(weights):
            cursor += weight
            if target < cursor:
                return index
        return len(weights) - 1


def euclid_squared(left, right):
    """Exact squared Euclidean distance; ordering matches Euclidean distance."""
    if len(left) != len(right):
        reject("input", "vectors must share the same grid")
    return sum(
        (fraction(a) - fraction(b)) ** 2 for a, b in zip(left, right)
    )


# Public callers use the squared metric: it stays an exact Fraction and gives
# identical nearest-neighbor ordering to Euclidean distance.
euclid = euclid_squared


def mean_vector(rows):
    if not rows:
        reject("input", "at least one vector is required")
    cells = len(rows[0])
    count = len(rows)
    return [
        sum((fraction(row[index]) for row in rows), Fraction(0)) / count
        for index in range(cells)
    ]


def nearest(point, centers):
    distances = [euclid_squared(point, center) for center in centers]
    best = min(range(len(centers)), key=lambda i: (distances[i], i))
    return best, distances[best]


def kmeans_plus_plus(points, k, seed):
    rng = SeededRandom(seed)
    first = rng.next_uint() % len(points)
    centers = [list(points[first])]
    nearest_sq = [euclid_squared(point, centers[0]) for point in points]
    while len(centers) < k:
        index = rng.weighted_choice(nearest_sq)
        centers.append(list(points[index]))
        nearest_sq = [
            min(previous, euclid_squared(point, centers[-1]))
            for previous, point in zip(nearest_sq, points)
        ]
    return centers


def run_kmeans(points, k, seed, max_iterations=100):
    integer(k, 1, len(points))
    integer(max_iterations, 1, 10000)
    centers = kmeans_plus_plus(points, k, seed)
    labels = None
    for iteration in range(max_iterations):
        labels = []
        groups = [[] for _ in range(k)]
        for index, point in enumerate(points):
            label, _ = nearest(point, centers)
            labels.append(label)
            groups[label].append(index)
        moved = False
        for label, members in enumerate(groups):
            if not members:
                continue
            updated = mean_vector([points[i] for i in members])
            if updated != centers[label]:
                centers[label] = updated
                moved = True
        if not moved:
            break
    # Final assignment against converged centers with deterministic tie breaking.
    labels = [nearest(point, centers)[0] for point in points]
    return centers, labels, iteration + 1


def silhouette(points, labels, k):
    """Exact silhouette score using squared Euclidean distances.

    Squared distance preserves every nearest/nearest-neighbor ordering used in
    the silhouette definition, and avoids floating point square roots.
    """
    if len(points) < 2 or k < 2:
        return None
    scores = []
    cache = {}
    for i, point in enumerate(points):
        same = [j for j in label_indices(labels, labels[i]) if j != i]
        if not same:
            scores.append(Fraction(0))
            continue
        own = sum(distance(cache, points, i, j) for j in same) / len(same)
        neighbour = None
        for label in range(k):
            if label == labels[i]:
                continue
            members = label_indices(labels, label)
            if not members:
                continue
            value = sum(
                distance(cache, points, i, j) for j in members
            ) / len(members)
            neighbour = value if neighbour is None else min(neighbour, value)
        if neighbour is None:
            scores.append(Fraction(0))
            continue
        span = max(own, neighbour)
        scores.append(
            (neighbour - own) / span if span > 0 else Fraction(0)
        )
    return sum(scores) / len(scores)


def label_indices(labels, label):
    return [index for index, value in enumerate(labels) if value == label]


def distance(cache, points, i, j):
    if i == j:
        return Fraction(0)
    key = (i, j) if i < j else (j, i)
    if key not in cache:
        cache[key] = euclid(points[key[0]], points[key[1]])
    return cache[key]


class Clusterer:
    def fit(self, normalized_days, k, seed, max_iterations=100):
        if not normalized_days:
            reject("input", "at least one eligible day is required to fit")
        dates = [row["date"] for row in normalized_days]
        if len(dates) != len(set(dates)):
            reject("input", "duplicate production days in clustering input")
        cells = {len(row["shares"]) for row in normalized_days}
        if len(cells) != 1:
            reject("input", "all curves must share the same grid width")
        for row in normalized_days:
            total = sum((fraction(v) for v in row["shares"]), Fraction(0))
            if total != 1:
                reject("input", "curves must be normalized to unit total")
        points = [[fraction(v) for v in row["shares"]] for row in normalized_days]
        k = min(integer(k, 1, len(points)), len(points))
        centers, labels, iterations = run_kmeans(
            points, k, seed, max_iterations
        )
        clusters = []
        for label in range(k):
            members = [i for i, value in enumerate(labels) if value == label]
            if not members:
                continue
            # Representative day is the real day closest to the cluster center.
            rep_index = min(
                members,
                key=lambda i: (
                    euclid_squared(points[i], centers[label]),
                    dates[i],
                ),
            )
            member_distances = [
                rational(euclid(points[i], centers[label])) for i in members
            ]
            within = [
                euclid_squared(points[i], centers[label]) for i in members
            ]
            clusters.append(
                {
                    "label": "C" + str(label + 1),
                    "size": len(members),
                    "dates": [dates[i] for i in sorted(members, key=lambda i: dates[i])],
                    "center": [rational(value) for value in centers[label]],
                    "representative_date": dates[rep_index],
                    "representative_shares": normalized_days[rep_index]["shares"],
                    "max_distance": rational(max(
                        euclid(points[i], centers[label]) for i in members
                    )),
                    "mean_distance": rational(
                        sum(
                            (euclid(points[i], centers[label]) for i in members),
                            Fraction(0),
                        )
                        / len(members)
                    ),
                    "within_inertia": rational(sum(within)),
                    "member_distances": dict(
                        sorted(
                            (
                                (dates[i], rational(euclid(points[i], centers[label])))
                                for i in members
                            )
                        )
                    ),
                }
            )
        score = silhouette(points, labels, k)
        inertia = sum(
            euclid_squared(points[i], centers[labels[i]]) for i in range(len(points))
        )
        return {
            "k": k,
            "seed": integer(seed, 0, 2**63 - 1),
            "iterations": iterations,
            "distance_metric": DISTANCE_METRIC,
            "dates": dates,
            "assignments": {
                dates[i]: "C" + str(labels[i] + 1) for i in range(len(dates))
            },
            "clusters": clusters,
            "inertia": rational(inertia),
            "silhouette": rational(score) if score is not None else None,
            "input_digest": fingerprint(
                [
                    {
                        "date": row["date"],
                        "shares": row["shares"],
                        "source_digest": row["source_digest"],
                    }
                    for row in normalized_days
                ]
            ),
        }

    def select_k(
        self, normalized_days, seed, min_k=2, max_k=6, max_iterations=100
    ):
        """Run every admissible k and rank by exact silhouette (higher is better)."""
        integer(min_k, 1)
        integer(max_k, min_k)
        upper = min(max_k, len(normalized_days))
        if min_k > upper:
            reject("input", "minimum cluster count exceeds eligible days")
        candidates = []
        for value in range(min_k, upper + 1):
            result = self.fit(normalized_days, value, seed, max_iterations)
            candidates.append(
                {
                    "k": value,
                    "silhouette": result["silhouette"],
                    "inertia": result["inertia"],
                    "iterations": result["iterations"],
                }
            )
        ranked = sorted(
            candidates,
            key=lambda row: (
                -(fraction(row["silhouette"]) if row["silhouette"] is not None else -1),
                row["k"],
            ),
        )
        return {
            "candidates": candidates,
            "best_k": ranked[0]["k"],
            "seed": integer(seed, 0, 2**63 - 1),
        }

    def assign(self, normalized_day, model, novelty_threshold):
        """Classify one later day against frozen centers; never mutates labels."""
        threshold = fraction(novelty_threshold)
        if threshold < 0:
            reject("input", "novelty threshold cannot be negative")
        point = [fraction(value) for value in normalized_day["shares"]]
        grid = len(model["clusters"][0]["center"])
        if len(point) != grid:
            reject("input", "curve grid differs from frozen model grid")
        scored = []
        for cluster in model["clusters"]:
            center = [fraction(value) for value in cluster["center"]]
            distance_value = euclid(point, center)
            scored.append(
                {
                    "label": cluster["label"],
                    "distance": rational(distance_value),
                    "threshold": rational(
                        fraction(cluster["max_distance"]) + threshold
                    ),
                }
            )
        best = min(scored, key=lambda row: (fraction(row["distance"]), row["label"]))
        novel = fraction(best["distance"]) > fraction(best["threshold"])
        return {
            "date": normalized_day["date"],
            "label": None if novel else best["label"],
            "state": "new_pattern" if novel else "assigned",
            "distance_metric": DISTANCE_METRIC,
            "distance": best["distance"],
            "threshold": best["threshold"],
            "candidates": scored,
            "source_digest": normalized_day["source_digest"],
        }
