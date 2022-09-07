import pytest
import numpy as np

from core.grad import *


@pytest.fixture(scope="function", params=[
    (np.array([2.0]), 1e-6, 32.0, 5),
    (np.array([2.0]), 0.01, 32.0, 3),
    (np.array([0.0]), 1e-6, 12.0, 5),
    (np.array([0.0]), 1e-7, 12.0, 6),
    (np.array([-2.0]), 1e-6, 16.0, 5)])
def non_smoothed_single_variable_params(request):
    return request.param

@pytest.fixture(scope="function", params=[
    (np.array([2.0]), 1e-6, 32.0, 5, 1),
    (np.array([2.0]), 1e-6, 32.0, 5, 3),
    (np.array([2.0]), 1e-6, 32.0, 5, 5),
    (np.array([2.0]), 1e-6, 32.0, 5, 10),
    (np.array([2.0]), 1e-6, 32.0, 5, 250)])
def smoothed_single_variable_params(request):
    return request.param

@pytest.fixture(scope="function", params=[
    (np.array([2.0, -1.0]), 1e-6, np.array([3.0, 0.0]), 5),
    (np.array([2.0, -1.0]), 0.01, np.array([3.0, 0.0]), 3),
    (np.array([0.0, 0.0]), 1e-6, np.array([0.0, 0.0]), 5),
    (np.array([0.0, 0.0]), 1e-7, np.array([0.0, 0.0]), 6),
    (np.array([-5.0, 10.0]), 1e-6, np.array([0.0, 15.0]), 5)])
def non_smoothed_multiple_variable_params(request):
    return request.param

def test_non_smoothed_should_calculate_grad_vector_single_variable(non_smoothed_single_variable_params):
    F = lambda x: x ** 3 + 2 * x ** 2 + 12 * x + 100
    (x0, h, g0, dec) = non_smoothed_single_variable_params # input variable, grid step, expected gradient value, decimal accuracy

    np.testing.assert_almost_equal(grad_left(F, x0, h=h), g0, decimal=dec)
    np.testing.assert_almost_equal(grad_center(F, x0, h=h), g0, decimal=dec)
    np.testing.assert_almost_equal(grad_right(F, x0, h=h), g0, decimal=dec)

def test_smoothed_should_calculate_grad_vector_single_variable(smoothed_single_variable_params):
    np.random.seed(0)

    F = lambda x: x ** 3 + 2 * x ** 2 + 12 * x + 100
    (x0, h, g0, dec, k) = smoothed_single_variable_params # input variable, grid step, expected gradient value, decimal accuracy, smoothing term

    np.testing.assert_almost_equal(grad_smoothed(F, x0, h=h, k=k), g0, decimal=dec)

def test_non_smoothed_should_calculate_grad_vector_multiple_variable(non_smoothed_multiple_variable_params):
    F = lambda x: x[0] ** 2 + x[0] * x[1] + x[1] ** 2
    (x0, h, g0, dec) = non_smoothed_multiple_variable_params

    np.testing.assert_almost_equal(grad_left(F, x0, h=h), g0, decimal=dec)
    np.testing.assert_almost_equal(grad_center(F, x0, h=h), g0, decimal=dec)
    np.testing.assert_almost_equal(grad_right(F, x0, h=h), g0, decimal=dec)
