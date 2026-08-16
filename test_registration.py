import numpy as np
from main import register, transform_points

# Tibia points
LANDMARKS = np.array([
    [0, 0],
    [-20, 100],
    [20, 200],
    [-30, 300],
    [30, 375],
    [0, 450],
])

# Round Trip Test
def test_recovers_known_transform():
    OR = transform_points(LANDMARKS, 30, (200, 300))
    angle, translation = register(LANDMARKS, OR)
    assert np.isclose(angle, 30)
    assert np.allclose(translation, [200, 300])

# Isolated Tests and Identity (registered against itself)
def test_identity():
    angle, translation = register(LANDMARKS, LANDMARKS)
    assert np.isclose(angle, 0)
    assert np.allclose(translation, [0, 0])

def test_pure_translation():
    OR = transform_points(LANDMARKS, 0, (100, -50))
    angle, translation = register(LANDMARKS, OR)
    assert np.isclose(angle, 0)
    assert np.allclose(translation, [100, -50])

def test_pure_rotation():
    OR = transform_points(LANDMARKS, 45, (0, 0))
    angle, translation = register(LANDMARKS, OR)
    assert np.isclose(angle, 45)
    assert np.allclose(translation, [0, 0])

# Randomized Tests
def test_random_transforms():
    rng = np.random.default_rng(0)
    for _ in range(500):
        true_angle = rng.uniform(-180, 180)
        true_t = rng.uniform(-500, 500, size=2)
        OR = transform_points(LANDMARKS, true_angle, true_t)
        angle, transformation = register(LANDMARKS, OR)
        assert np.isclose(angle, true_angle)
        assert np.allclose(transformation, true_t)