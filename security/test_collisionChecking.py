import pytest

from .collisionChecking import RobotCollisionCheck 

# Define multiple sets of angles for testing
test_angles = [
    {
        "angles": [0.9509, -1.6623, 1.8353, -0.5976, -1.5722, 0.0],
        "expected": False,  # collision with itself
    },
    {"angles": [0.9509, -1.6623, 0.6353, -0.5976, -1.5722, 0.0], "expected": True},
    {
        "angles": [0.9509, -1.6623, 2.6353, 0.5976, 1.5722, 0.0],
        "expected": False,  # collision with itself
    },
    {"angles": [0.9509, -1.6623, 0.6353, -0.5976, -1.5722, 0.0], "expected": True},
    {
        "angles": [0.0, -3.14, 3.14, 0.0, 0.0, 0.0],
        "expected": False,  # collision with itself
    },
    {
        "angles": [0.0, -3.14, 2.7, 0.0, 3.14, 0.0],
        "expected": False,  # collision with itself
    },
    {
        "angles": [0.9509, -1.6623, 0.6353, 0.5, -1.5722, 0.0],
        "expected": False,  # collision with itself
    },
    {
        "angles": [0.00001, 0.000001, 0.00001, 0.00001, 0.00001, 0.000001],
        "expected": True,
    },
    {
        "angles": [0.0, -3.14, 3.14, 0.0, 3.14, 0.0],
        "expected": False,  # collision with itself
    },
    {"angles": [-0.7493, -1.0, 1.0478, -1.0997, -0.2, -2.3201], "expected": True},
    {
        "angles": [0.9019, -1.1795, 1.9326, -2.3253, -1.5697, -0.6689],
        "expected": False,  # collision with itself
    },
    {
        "angles": [0.0, -3.14, 2.2, 0.6, 3.0, 0.0],
        "expected": False,  # collision with itself
    },
    {
        "angles": [0.9509, -1.6623, 1.8353, -0.5976, -1.5722, 0.0],
        "expected": False,  # collision with itself
    },
    {
        "angles": [0.0, -3.14, 3.14, 0.0, 0.0, 0.0],
        "expected": False,  # collision with the ground
    },
    {
        "angles": [0.0, -3.14, 2.7, 0.0, 3.14, 0.0],
        "expected": False,  # collision with the ground
    },
    {"angles": [-0.7493, -1.0, 1.0478, -1.0997, -0.2, -2.3201], "expected": True},
    {
        "angles": [0.9019, -1.1795, 1.9326, -2.3253, -1.5697, -0.6689],
        "expected": False,  # collision with the ground
    },
    {
        "angles": [0.0, -3.14, 2.2, 0.6, 3.0, 0.0],
        "expected": False,  # collision with the ground
    },
    {
        "angles": [3.141, 2.094, 0.785, 4.712, 6.283, 9.425],
        "expected": False,  # collision with the ground,
    },
    {
        "angles": [0.9019, -0.28, 1.7852, -3.0774, -1.5697, -0.6689],
        "expected": False,  # collision with the ground
    },
    {
        "angles": [0.5, -1.0, 1.2, -0.8, 1.0, 0.0],  # Inside working area
        "expected": True,
    },
    {
        "angles": [0.3, -0.5, 0.7, -0.4, 0.6, 0.2],  # Inside working area
        "expected": True,
    },
    {
        "angles": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],  # Inside working area
        "expected": True,
    },
    {
        "angles": [-0.5, -1.2, 1.0, -0.6, 0.8, -0.2],  # Inside working area
        "expected": True,
    },
    {
        "angles": [0.1, -0.8, 0.9, -0.3, 0.4, 0.1],  # Inside working area
        "expected": True,
    },
    {
        "angles": [1.5, -2.0, 2.5, -1.0, 1.0, 0.0],  # Pen position likely exceeds x_max
        "expected": True,
    },
    {
        "angles": [0.0, -3.14, 3.14, 0.0, 0.0, 0.0],  # Pen position likely below z_min
        "expected": False,
    },
    {
        "angles": [0.0, -3.0, 3.0, 0.0, 0.0, 0.0],  # Pen position likely exceeds z_max
        "expected": False,
    },
]
    
robot = RobotCollisionCheck(False,True)
@pytest.mark.parametrize("test_case", test_angles)
def test_checkingCollisonWithItself(test_case):
    angles = test_case["angles"]
    expected = test_case["expected"]

    result = robot.isValidConfiguration(angles)
    print("Failed for angles : ", angles)
    assert result == expected

    
