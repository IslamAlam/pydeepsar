"""Tests for Models module."""

# %%
import sys

from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from scipy.integrate import quad

from pydeepsar.models.coherene import CoherenceIceModel


def complex_quadrature(
    func: Callable[[np.float64], np.complex128],
    a: float,
    b: float,
    **kwargs: Any,
) -> Tuple[np.complex128, np.complex128]:
    """Numerically compute the complex quadrature of a complex-valued function.

    This function computes the complex quadrature of a \
        complex-valued function using numerical integration.

    Parameters
    ----------
    func : Callable[[np.float64], np.complex128]
        The complex-valued function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    **kwargs : keyword arguments, optional
        Additional options for the scipy.integrate.quad function.

    Returns
    -------
    Tuple[np.complex128, np.complex128]
        A tuple containing the complex integral value, \
            real integral errors, and imaginary integral errors.

    Examples
    --------
    >>> import numpy as np
    >>> def func(x):
    ...     return np.exp(1j * x)
    >>> complex_quadrature(func, 0, np.pi)
    (1.5707963267948966j, (5.921189464667501e-14,), (5.921189464667501e-14,))
    """

    def real_func(x: np.float64) -> np.float64:
        """Real part of the complex-valued function."""
        return np.real(func(x))

    def imag_func(x: np.float64) -> np.float64:
        """Imaginary part of the complex-valued function."""
        return np.imag(func(x))

    real_integral: Tuple[float, float] = quad(real_func, a, b, **kwargs)
    imag_integral: Tuple[float, float] = quad(imag_func, a, b, **kwargs)
    integral = np.complex128(real_integral[0] + 1j * imag_integral[0])
    error = np.complex128(real_integral[1] + 1j * imag_integral[1])
    return integral, error


class ComplexCoherenceEstimator:
    """Estimate the complex coherence parameter gamma.

    This class computes the complex coherence parameter \
        gamma given a function f(z),
    along with parameters kappa_z, kappa_z_vol, and z0.

    Parameters
    ----------
    f : Callable[[np.float64], np.complex128]
        The function f(z).
    kappa_z : np.float64
        Parameter kappa_z.
    kappa_z_vol : np.float64
        Parameter kappa_z_vol.
    z0 : np.float64
        Parameter z0.
    """

    def __init__(
        self,
        f: Callable[[np.float64], np.complex128],
        kappa_z: np.float64,
        kappa_z_vol: np.float64,
        z0: np.float64,
    ) -> None:
        """Initialize the ComplexCoherenceEstimator.

        Parameters
        ----------
        f : Callable[[np.float64], np.complex128]
            The function f(z).
        kappa_z : np.float64
            Parameter kappa_z.
        kappa_z_vol : np.float64
            Parameter kappa_z_vol.
        z0 : np.float64
            Parameter z0.
        """
        self.f = f
        self.kappa_z = kappa_z
        self.kappa_z_vol = kappa_z_vol
        self.z0 = z0

    def integrand_num(self, z: np.float64) -> np.complex128:
        """Define the integrand for the numerator of gamma.

        Parameters
        ----------
        z : np.float64
            Integration variable.

        Returns
        -------
        np.complex128
            Value of the integrand for the numerator.
        """
        return np.complex128(self.f(z) * np.exp(1j * self.kappa_z_vol * z))

    def integrand_den(self, z: np.float64) -> np.complex128:
        """Define the integrand for the denominator of gamma.

        Parameters
        ----------
        z : np.float64
            Integration variable.

        Returns
        -------
        np.complex128
            Value of the integrand for the denominator.
        """
        return np.complex128(self.f(z))

    def calculate_gamma(
        self, a: float = -np.inf, b: float = 0.0
    ) -> np.complex128:
        """Calculate the complex coherence parameter gamma numerically.

        Parameters
        ----------
        a : float, optional
            Lower limit of integration. Defaults to -np.inf.
        b : float, optional
            Upper limit of integration. Defaults to 0.

        Returns
        -------
        np.complex128
            The complex coherence parameter gamma.
        """
        # Numerically compute the integrals
        integral_num, _ = complex_quadrature(self.integrand_num, a, b)
        integral_den, _ = complex_quadrature(self.integrand_den, a, b)

        # Calculate gamma
        epsilon = np.complex128(sys.float_info.epsilon)
        gamma = (
            np.exp(1j * self.kappa_z * self.z0)
            * integral_num
            / (integral_den + epsilon)
        )
        return np.complex128(gamma)


def f_UV(
    z: np.float64, m1: np.float64, kappa_e: np.float64, theta_r: np.float64
) -> np.complex128:
    """Define the function f_UV(z).

    Parameters
    ----------
    z : np.float64
        Variable z.
    m1 : np.float64
        Parameter m1.
    kappa_e : np.float64
        Parameter kappa_e.
    theta_r : np.float64
        Parameter theta_r.

    Returns
    -------
    np.complex128
        Value of the function f_UV(z).
    """
    return np.complex128(m1 * np.exp((2 * kappa_e / np.cos(theta_r)) * z))


def estimate_kappa_e(theta_r: np.float64, d_pen: np.float64) -> np.float64:
    """Estimate kappa_e for the Uniform Volume model.

    Parameters
    ----------
    theta_r : float
        Angle of refraction in radians.
    d_pen : float
        Penetration depth.

    Returns
    -------
    float
        Estimated value of kappa_e.
    """
    return np.float64(np.abs(np.cos(theta_r) / d_pen))


# %%


@pytest.fixture
def response_coherence() -> bool:
    """Sample pytest fixture."""
    # Example usage:
    # Initialize variables as numpy arrays
    d_pen: npt.NDArray[np.float64] = np.array([30])
    theta_r_value: npt.NDArray[np.float64] = np.array([np.deg2rad(40)])
    m1_value: npt.NDArray[np.float64] = np.array([1.0])
    kappa_z_value: npt.NDArray[np.float64] = np.array([0.16])
    kappa_z_vol_value: npt.NDArray[np.float64] = np.array([0.15])
    z0_value: npt.NDArray[np.float64] = np.array([0])

    # Estimate kappa_e
    kappa_e_value = estimate_kappa_e(theta_r_value[0], d_pen[0])

    # Calculate the complex coherence
    gamma = ComplexCoherenceEstimator(
        lambda z: f_UV(z, m1_value[0], kappa_e_value, theta_r_value[0]),
        kappa_z_value[0],
        kappa_z_vol_value[0],
        z0_value[0],
    ).calculate_gamma()

    # Define z values
    a_input = -500.0
    b_input = 0.0
    num_intervals_input = 1000
    z_value = np.linspace(a_input, b_input, num_intervals_input + 1)

    X = {
        "d_pen": d_pen[:, np.newaxis],
        "z": z_value[
            np.newaxis,
            :,
        ],
        "kappa_z": kappa_z_value[:, np.newaxis],
        "z0": z0_value[:, np.newaxis],
        "kappa_z_vol": kappa_z_vol_value[:, np.newaxis],
    }

    ice_model = CoherenceIceModel()

    if ice_model is not None:
        gamma_tf = ice_model.model.predict(X)  # type: ignore[union-attr]
        assert np.allclose(gamma_tf, gamma)
    else:
        raise ValueError("Failed to create CoherenceIceModel instance")

    return np.allclose(gamma_tf, gamma)
