import pytest

from .workingAreaChecking import WorkingAreaRobotChecking

# filepath: /home/marta/Projects/SecurityModule/test_unitTestsWorkingArea.py

# Define test cases from setUp
test_cases = [
    # Test case 1: Points inside the upper hemisphere
    {
        "test_points": {
            1: {"x": 0.1, "y": 0.1, "z": 0.1},  # Inside
            2: {"x": -0.2, "y": -0.2, "z": 0.2},  # Inside
            3: {"x": 0.3, "y": -0.3, "z": 0.3},  # Outside (y < 0)
            4: {"x": -0.3, "y": 0.3, "z": -0.3},  # Outside (z < 0)
            5: {"x": 0.6, "y": 0.6, "z": 0.6},  # Outside (exceeds radius)
            6: {"x": 0.2, "y": 0.2, "z": 0.2},  # Inside
        },
        "expected_results": {
            6: False,
        }
    },

    # Test case 3: Points on the boundary (edge of the sphere)
    {
        "test_points": {
            1: {"x": 0.5, "y": 0, "z": 0},  # On the boundary
            2: {"x": -0.5, "y": 0, "z": 0},  # On the boundary
            3: {"x": 0, "y": 0.5, "z": 0},  # On the boundary
            4: {"x": 0, "y": -0.5, "z": 0},  # On the boundary
            5: {"x": 0, "y": 0, "z": 0.5},  # On the boundary
            6: {"x": 0, "y": 0, "z": -0.5},  # On the boundary (z < 0)
        },
        "expected_results": {

            6: False,  # Bottom hemisphere, not inside the upper hemisphere
        }
    },
    # Test case 4: Random scattered points
    {
        "test_points": {
            1: {"x": -0.3, "y": 0.4, "z": 0.1},  # Inside
            2: {"x": 0.2, "y": 0.5, "z": -0.3},  # Outside (z < 0)
            3: {"x": -0.7, "y": -0.2, "z": 0.2},  # Outside (exceeds radius)
            4: {"x": 0.2, "y": -0.1, "z": 0.3},  # Inside
            5: {"x": 0.1, "y": 0.1, "z": 0.4},  # Inside
            6: {"x": -0.1, "y": -0.2, "z": -0.3},  # Outside (z < 0)
        },
        "expected_results": {
            6: False,
        }
    },
    # Test case 5: Extreme points (further from the origin)
    {
        "test_points": {
            1: {"x": 1000, "y": 0, "z": 0},  # Outside (exceeds radius)
            2: {"x": 0, "y": 1000, "z": 0},  # Outside (exceeds radius)
            3: {"x": 0, "y": 0, "z": 1000},  # Outside (exceeds radius)
            4: {"x": -0.5, "y": 0, "z": 0},  # Inside
            5: {"x": 0, "y": 0, "z": -0.5},  # Inside
            6: {"x": 0, "y": -0.5, "z": 0},  # Inside
        },
        "expected_results": {
            6: True,   # Inside
        }
    },
]

@pytest.mark.parametrize("case", test_cases)
def test_points_in_half_sphere(case):
    test_points = case["test_points"]
    expected_results = case["expected_results"]
    angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Dummy angles for testing

    # Create WorkingAreaRobotChecking instance
    sphere = WorkingAreaRobotChecking(0, 0, 0, 0.5, angles, True, test_points)

    # Check if the computed results match the expected results
    results = sphere.checkPointsInHalfOfSphere()
    assert results == expected_results