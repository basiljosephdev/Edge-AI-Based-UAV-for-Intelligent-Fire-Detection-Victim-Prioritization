"""
gps_projector.py — Estimates the ground GPS coordinate of a detected
bounding box using the drone's own GPS position, altitude, heading and the
camera's field of view.

This uses a simplified flat-ground, nadir-camera pinhole model: it assumes
the camera points straight down (or a fixed known tilt) and the terrain
under the drone is roughly flat. This is the same approximation approach
implied by the README's "±3-5 meters at 15m altitude" accuracy figure —
it is NOT survey-grade and will degrade at low altitude, on sloped terrain,
or with a gimbal that isn't pointing straight down. For a real deployment,
feed the camera's actual gimbal tilt angle in here instead of assuming nadir.
"""

import math

EARTH_RADIUS_M = 6371000.0


def pixel_to_ground_offset(
    center_px, frame_w, frame_h, altitude_m, h_fov_deg, v_fov_deg
):
    """Convert a pixel location to a forward/right ground offset (meters)
    from the point directly beneath the drone, assuming a nadir-pointing
    camera on a level airframe."""
    px_x, px_y = center_px

    # Normalized offset from frame center, range [-0.5, 0.5]
    norm_x = (px_x / frame_w) - 0.5
    norm_y = (px_y / frame_h) - 0.5

    ground_width_m = 2 * altitude_m * math.tan(math.radians(h_fov_deg / 2))
    ground_height_m = 2 * altitude_m * math.tan(math.radians(v_fov_deg / 2))

    right_m = norm_x * ground_width_m
    forward_m = -norm_y * ground_height_m  # image y grows downward -> forward is negative norm_y
    return forward_m, right_m


def offset_to_gps(lat, lon, heading_deg, forward_m, right_m):
    """Rotate a forward/right ground offset by the drone's heading and apply
    it to a lat/lon origin, returning the resulting (lat, lon)."""
    heading_rad = math.radians(heading_deg or 0.0)

    # Rotate forward/right into North/East components.
    north_m = forward_m * math.cos(heading_rad) - right_m * math.sin(heading_rad)
    east_m = forward_m * math.sin(heading_rad) + right_m * math.cos(heading_rad)

    d_lat = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    d_lon = (east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180.0 / math.pi)

    return lat + d_lat, lon + d_lon


def project_detection_to_gps(center_px, frame_w, frame_h, telemetry, cam_cfg):
    """Given a detection's pixel center and current flight telemetry, return
    an estimated {lat, lon, alt_m} dict, or None if telemetry isn't valid."""
    if not telemetry.is_valid():
        return None

    altitude_m = telemetry.relative_alt_m or 0.0
    if altitude_m <= 0:
        return None

    forward_m, right_m = pixel_to_ground_offset(
        center_px,
        frame_w,
        frame_h,
        altitude_m,
        cam_cfg["horizontal_fov_deg"],
        cam_cfg["vertical_fov_deg"],
    )
    lat, lon = offset_to_gps(
        telemetry.lat, telemetry.lon, telemetry.heading_deg, forward_m, right_m
    )
    return {"lat": lat, "lon": lon, "alt_m": altitude_m}


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
