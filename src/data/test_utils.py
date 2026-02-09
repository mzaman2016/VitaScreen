import numpy as np

from data.utils import nctd_transform


def test_nctd_transform():
    # Test with a simple case
    x = np.array([1, 2, 3, 4, 5])
    n_features = len(x)
    transformed_x = nctd_transform(x, n_features)

    # Check the shape of the transformed data
    assert transformed_x.shape == (2 * n_features, 2 * n_features)

    # Check if the values are scaled correctly
    expected_values = 255 * np.array(
        [
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
            [3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
            [4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
            [5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
            [3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
            [4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
            [5, 1, 2, 3, 4, 5, 1, 2, 3, 4],
        ]
    )
    assert np.allclose(transformed_x, expected_values)
