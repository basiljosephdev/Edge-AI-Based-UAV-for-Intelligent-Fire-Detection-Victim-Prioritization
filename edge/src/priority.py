"""
priority.py — Computes the rescue priority score for each detected victim,
combining posture vulnerability, proximity to the nearest fire, and that
fire's severity, per the weighting scheme described in the README:

    P = w1 * posture_vulnerability
      + w2 * (1 - min(distance_to_fire, d_max) / d_max)
      + w3 * fire_severity_normalized

Victims are then ranked and bucketed into the same five severity classes
shown in the README's posture table.
"""

from .gps_projector import haversine_distance_m

POSTURE_VULNERABILITY = {
    "Lying": 1.0,
    "Sitting": 0.75,
    "Standing": 0.5,
    "Walking": 0.25,
    "Running": 0.1,
}

# Priority-score thresholds -> severity class label (mirrors the posture
# table's vulnerability tiers, since posture is the dominant term by default
# weighting). Tune these if you change priority_weights.
SEVERITY_THRESHOLDS = [
    (0.85, "CRITICAL"),
    (0.65, "HIGH"),
    (0.45, "MEDIUM"),
    (0.2, "LOW"),
    (0.0, "MINIMAL"),
]


def severity_class_for_score(score):
    for threshold, label in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "MINIMAL"


def nearest_fire(victim_gps, fire_list):
    """Return (distance_m, fire) for the closest fire detection with a valid
    GPS estimate, or (None, None) if there are no fires this frame."""
    best_dist, best_fire = None, None
    for fire in fire_list:
        if fire.get("gps") is None:
            continue
        dist = haversine_distance_m(
            victim_gps["lat"], victim_gps["lon"], fire["gps"]["lat"], fire["gps"]["lon"]
        )
        if best_dist is None or dist < best_dist:
            best_dist, best_fire = dist, fire
    return best_dist, best_fire


def compute_priorities(victims, fires, weights):
    """victims/fires: lists of detection dicts that already have a 'gps' key
    (see gps_projector.project_detection_to_gps). Mutates each victim dict in
    place with distance/severity/priority_score/severity_class, then returns
    the list sorted by descending priority with a 1-based 'rescue_rank'."""
    w1 = weights.get("posture", 0.5)
    w2 = weights.get("proximity", 0.3)
    w3 = weights.get("fire_severity", 0.2)
    d_max = weights.get("d_max_meters", 30.0)

    scored = []
    for v in victims:
        if v.get("gps") is None:
            continue

        posture_score = POSTURE_VULNERABILITY.get(v["class"], 0.5)

        dist_m, fire = nearest_fire(v["gps"], fires)
        if dist_m is None:
            # No fire detected this frame yet: proximity/fire terms drop out,
            # priority is driven by posture alone.
            proximity_score = 0.0
            fire_severity = 0.0
        else:
            proximity_score = max(0.0, 1.0 - min(dist_m, d_max) / d_max)
            fire_severity = fire.get("severity_normalized", 0.0)

        priority = w1 * posture_score + w2 * proximity_score + w3 * fire_severity

        v["distance_to_nearest_fire_m"] = round(dist_m, 2) if dist_m is not None else None
        v["fire_severity_normalized"] = round(fire_severity, 2)
        v["priority_score"] = round(priority, 3)
        v["severity_class"] = severity_class_for_score(priority)
        scored.append(v)

    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    for i, v in enumerate(scored, start=1):
        v["rescue_rank"] = i

    return scored
