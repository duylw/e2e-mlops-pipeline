from src.utils.geo import calculate_haversine


def test_haversine_returns_positive_distance_for_different_points():
    distance = calculate_haversine(40.785091, -73.968285, 40.758896, -73.98513)

    assert distance > 0
