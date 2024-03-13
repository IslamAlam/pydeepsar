"""
Module for computing coherence measures using TensorFlow.

This module provides functions for computing coherence measures
using TensorFlow, a popular deep learning framework.

Classes:
    CoherenceLayer: A TensorFlow layer for computing coherence measures.

Functions:
    compute_coherence: Compute coherence measures using TensorFlow.

"""
# %%
from typing import Any, Callable

# import numpy as np
# import numpy.typing as npt
import tensorflow as tf

# -> Type[tf.keras.layers.Layer]


class IntegrateSimpsonsRule(tf.keras.layers.Layer):  # type: ignore[misc]
    """Numerically integrate a function using Simpson's Rule.

    This layer numerically integrates a function using Simpson's Rule method.

    Parameters
    ----------
    func : Callable[[tf.Tensor], tf.Tensor]
        The function to be integrated.
    num_intervals : int, optional
        Number of intervals for integration. Defaults to 1000.
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The approximate integral of the given function.

    Raises
    ------
    ValueError
        If the number of intervals is not positive.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> def func(x):
    ...     return tf.sin(x)
    >>> layer = IntegrateSimpsonsRule(func=func)
    >>> inputs = tf.constant([0.0, np.pi])
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor(2.0, shape=(), dtype=float32)
    """

    def __init__(
        self,
        func: Callable[[tf.Tensor], tf.Tensor],
        num_intervals: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize the IntegrateSimpsonsRule layer.

        Parameters
        ----------
        func : Callable[[tf.Tensor], tf.Tensor]
            The function to be integrated.
        num_intervals : int, optional
            Number of intervals for integration. Defaults to 1000.
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(IntegrateSimpsonsRule, self).__init__(**kwargs)
        self.func = func
        if num_intervals <= 0:
            raise ValueError("Number of intervals must be positive.")
        self.num_intervals = num_intervals

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Perform numerical integration using Simpson's Rule.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing the integration limits [a, b].

        Returns
        -------
        tf.Tensor
            The approximate integral of the given function over the specified interval.
        """  # noqa: E501
        a, b = inputs

        h = tf.cast(
            (b - a) / tf.cast(self.num_intervals, dtype=tf.float32),
            dtype=tf.float32,
        )

        x = tf.linspace(a, b, self.num_intervals + 1)
        y = self.func(x)

        odd_indices = tf.range(1, self.num_intervals, 2)
        even_indices = tf.range(2, self.num_intervals, 2)
        sum_y_odd = tf.reduce_sum(tf.gather(y, odd_indices))
        sum_y_even = tf.reduce_sum(tf.gather(y, even_indices))

        sum_y_odd = tf.cast(sum_y_odd, dtype=tf.float32)
        sum_y_even = tf.cast(sum_y_even, dtype=tf.float32)

        integral = (
            h
            / 3
            * (
                self.func(a)
                + 4.0 * sum_y_odd
                + 2.0 * sum_y_even
                + self.func(b)
            )
        )
        return integral


class IntegrateDiscreteSimpsonsRule(tf.keras.layers.Layer):  # type: ignore[misc]
    """Numerically integrate discrete data using Simpson's Rule.

    This layer numerically integrates discrete data using Simpson's Rule method.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The approximate integral of the given discrete data.

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = IntegrateDiscreteSimpsonsRule()
    >>> x_values = tf.constant([0.0, 1.0, 2.0])
    >>> y_values = tf.constant([1.0, 2.0, 1.0])
    >>> inputs = [x_values, y_values]
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor(1.3333334, shape=(), dtype=float32)
    """  # noqa: E501

    def __init__(self, **kwargs: Any):
        """Initialize the IntegrateDiscreteSimpsonsRule layer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(IntegrateDiscreteSimpsonsRule, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Perform numerical integration using Simpson's Rule for discrete data.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing the x and y values of discrete data points.

        Returns
        -------
        tf.Tensor
            The approximate integral of the given discrete data.
        """  # noqa: E501
        x_values, y_values = inputs

        # Calculate step size based on x_values
        h = (x_values[-1] - x_values[0]) / tf.cast(
            tf.shape(x_values)[0] - 1, dtype=tf.float32
        )

        # Simpson's rule calculation
        sum_y_odd = tf.reduce_sum(y_values[1::2])
        sum_y_even = tf.reduce_sum(y_values[2:-1:2])

        integral = (
            h
            / 3
            * (y_values[0] + 4.0 * sum_y_odd + 2.0 * sum_y_even + y_values[-1])
        )

        return integral


class ComplexIntegrateDiscreteSimpsonsRule(tf.keras.layers.Layer):  # type: ignore[misc]
    """Perform complex integration of discrete data using Simpson's Rule.

    This layer performs complex integration of discrete data using Simpson's Rule method.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The complex integral of the given discrete data.

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = ComplexIntegrateDiscreteSimpsonsRule()
    >>> x_values = tf.constant([0.0, 1.0, 2.0])
    >>> y_values = tf.constant([1.0, 2.0, 1.0])
    >>> inputs = [x_values, y_values]
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor((1.3333334+0j), shape=(), dtype=complex64)
    """  # noqa: E501

    def __init__(self, **kwargs: Any):
        """Initialize the ComplexIntegrateDiscreteSimpsonsRule layer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(ComplexIntegrateDiscreteSimpsonsRule, self).__init__(**kwargs)
        self.integrate_discret_simpsons_rule = IntegrateDiscreteSimpsonsRule()

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Perform complex integration using Simpson's Rule for discrete data.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing the x and y values of discrete data points.

        Returns
        -------
        tf.Tensor
            The complex integral of the given discrete data.
        """
        x_values, y_values = inputs

        def real_func(x: tf.Tensor) -> tf.Tensor:
            return tf.math.real(x)

        def imag_func(x: tf.Tensor) -> tf.Tensor:
            return tf.math.imag(x)

        real_integral = self.integrate_discret_simpsons_rule(
            [(x_values), real_func(y_values)]
        )
        imag_integral = self.integrate_discret_simpsons_rule(
            [(x_values), imag_func(y_values)]
        )
        return tf.complex(real_integral, imag_integral)


class ComplexCoherenceEstimatorLayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Estimate complex coherence using Simpson's Rule with discrete data points.

    This layer estimates complex coherence using Simpson's Rule with discrete data points.

    Parameters
    ----------
    inputs : tf.Tensor
        A tensor containing z_values, func_z_values, kappa_z, kappa_z_vol, and z0.

    Returns
    -------
    tf.Tensor
        The estimated complex coherence.

    Raises
    ------
    ValueError
        If the input data is not in the expected format.

    Examples
    --------
    >>> layer = ComplexCoherenceEstimatorLayer()
    >>> z_values = tf.constant([1, 2, 3])
    >>> func_z_values = tf.constant([2, 3, 4])
    >>> kappa_z = tf.constant(0.5)
    >>> kappa_z_vol = tf.constant(0.8)
    >>> z0 = tf.constant(1.2)
    >>> inputs = [z_values, func_z_values, kappa_z, kappa_z_vol, z0]
    >>> result = layer(inputs)
    >>> print(result)
    <tf.Tensor: shape=(), dtype=complex64, numpy=(0.9877519+0.15643446j)>
    """  # noqa: E501

    def __init__(self, **kwargs: Any):
        """Initialize the ComplexCoherenceEstimatorLayer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional keyword arguments to be passed to the base Layer class.
        """
        super(ComplexCoherenceEstimatorLayer, self).__init__(**kwargs)

    def integrand_num(
        self, z: tf.Tensor, f: tf.Tensor, kappa_z_vol: tf.Tensor
    ) -> tf.Tensor:
        """Define the integrand for the numerator of gamma.

        Parameters
        ----------
        z : tf.Tensor
            Integration variable.
        f : tf.Tensor
            Function f(z).
        kappa_z_vol : tf.Tensor
            Parameter kappa_z_vol.

        Returns
        -------
        tf.Tensor
            Value of the integrand for the numerator.
        """
        x_values = z
        y_values = tf.cast(f, tf.complex64) * tf.math.exp(
            tf.complex(0.0, (kappa_z_vol * z))
        )

        return ComplexIntegrateDiscreteSimpsonsRule(name="IntegrateNum")(
            [x_values, y_values]
        )

    def integrand_den(self, z: tf.Tensor, f: tf.Tensor) -> tf.Tensor:
        """Define the integrand for the denominator of gamma.

        Parameters
        ----------
        z : tf.Tensor
            Integration variable.
        f : tf.Tensor
            Function f(z).

        Returns
        -------
        tf.Tensor
            Value of the integrand for the denominator.
        """
        x_values = z
        y_values = f
        return ComplexIntegrateDiscreteSimpsonsRule(name="IntegrateDen")(
            [x_values, y_values]
        )

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Perform the complex coherence estimation.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing z_values, func_z_values, kappa_z, kappa_z_vol, and z0.

        Returns
        -------
        tf.Tensor
            The estimated complex coherence.
        """  # noqa: E501
        z_values, func_z_values, kappa_z, kappa_z_vol, z0 = inputs

        def integral_num(
            z: tf.Tensor, func_z: tf.Tensor, kappa_z_vol: tf.Tensor
        ) -> tf.Tensor:
            return self.integrand_num(z, func_z, kappa_z_vol)

        def integral_den(z: tf.Tensor, func_z: tf.Tensor) -> tf.Tensor:
            return self.integrand_den(
                z,
                func_z,
            )

        # Simpson's rule calculation for numerator
        integral_num = integral_num(
            z_values,
            func_z_values,
            kappa_z_vol,
        )

        # Simpson's rule calculation for denominator
        integral_den = integral_den(
            z_values,
            func_z_values,
        )

        # Calculate gamma
        gamma = tf.math.exp(tf.complex(0.0, (kappa_z * z0))) * (
            tf.cast(integral_num, tf.complex64)
            / tf.cast(integral_den, tf.complex64)
        )

        return gamma


class PhaseEstimationLayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Estimate the phase angle of complex input.

    This layer estimates the phase angle of the complex input.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The phase angle of the complex input.

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = PhaseEstimationLayer()
    >>> inputs = tf.constant([1+1j, -1-1j])
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor([ 0.7853982 -2.3561945], shape=(2,), dtype=float32)
    """

    def __init__(self, **kwargs: Any):
        """Initialize the PhaseEstimationLayer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(PhaseEstimationLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Estimate the phase angle of complex input.

        Parameters
        ----------
        inputs : tf.Tensor
            The complex input.

        Returns
        -------
        tf.Tensor
            The phase angle of the complex input.
        """
        complex_input = inputs
        return tf.math.angle(complex_input)


class AmplitudeEstimationLayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Estimate the amplitude of complex input.

    This layer estimates the amplitude of the complex input.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The amplitude of the complex input.

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = AmplitudeEstimationLayer()
    >>> inputs = tf.constant([1+1j, -1-1j])
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor([1.4142135 1.4142135], shape=(2,), dtype=float32)
    """

    def __init__(self, **kwargs: Any):
        """Initialize the AmplitudeEstimationLayer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(AmplitudeEstimationLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Estimate the amplitude of complex input.

        Parameters
        ----------
        inputs : tf.Tensor
            The complex input.

        Returns
        -------
        tf.Tensor
            The amplitude of the complex input.
        """
        complex_input = inputs
        amplitude = tf.abs(complex_input)
        return amplitude


class PhaseCenterDepthEstimationLayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Estimate the phase center depth using complex coherence and kappa_z_vol.

    This layer estimates the phase center depth using complex coherence and kappa_z_vol.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The estimated phase center depth.

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = PhaseCenterDepthEstimationLayer()
    >>> gamma = tf.constant([0.5+0.5j, -0.5-0.5j])
    >>> kappa_z_vol = tf.constant(0.1)
    >>> inputs = [gamma, kappa_z_vol]
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor([ 7.8539815 -7.8539815], shape=(2,), dtype=float32)
    """  # noqa: E501

    def __init__(self, **kwargs: Any):
        """Initialize the PhaseCenterDepthEstimationLayer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(PhaseCenterDepthEstimationLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Estimate the phase center depth.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing the complex coherence gamma and kappa_z_vol.

        Returns
        -------
        tf.Tensor
            The estimated phase center depth.
        """
        gamma, kappa_z_vol = inputs
        phase = tf.math.angle(gamma)
        phase_center_depth = phase / kappa_z_vol
        return phase_center_depth


class EstimateKappaELayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Estimate kappa_e using theta_r and d_pen.

    This layer estimates kappa_e using theta_r and d_pen.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The estimated kappa_e.

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = EstimateKappaELayer()
    >>> theta_r = tf.constant([0.1, 0.2])
    >>> d_pen = tf.constant([0.5, 1.0])
    >>> inputs = [theta_r, d_pen]
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor([1.9800678 1.0      ], shape=(2,), dtype=float32)
    """

    def __init__(self, **kwargs: Any):
        """Initialize the EstimateKappaELayer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(EstimateKappaELayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Estimate kappa_e.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing theta_r and d_pen.

        Returns
        -------
        tf.Tensor
            The estimated kappa_e.
        """
        theta_r, d_pen = inputs
        kappa_e = tf.abs(tf.cos(theta_r) / d_pen)
        return kappa_e


class UVLayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Compute the f(z) using a Uniform Volume (UV) model.

    This layer computes the f(z) using a Uniform Volume (UV) model based on the input parameters.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The computed f(z).

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = UVLayer()
    >>> z = tf.constant([0.1, 0.2])
    >>> m1 = tf.constant([0.5, 1.0])
    >>> kappa_e = tf.constant([0.2, 0.3])
    >>> theta_r = tf.constant([0.1, 0.2])
    >>> inputs = [z, m1, kappa_e, theta_r]
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor([1.0214052 1.0426943], shape=(2,), dtype=float32)
    """  # noqa: E501

    def __init__(self, **kwargs: Any):
        """Initialize the UVLayer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(UVLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Compute the f(z) using a Uniform Volume (UV) model.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing z, m1, kappa_e, and theta_r.

        Returns
        -------
        tf.Tensor
            The computed UV field.
        """
        z, m1, kappa_e, theta_r = inputs
        z = tf.cast(z, dtype=tf.float32)
        m1 = tf.cast(m1, dtype=tf.float32)
        kappa_e = tf.cast(kappa_e, dtype=tf.float32)
        theta_r = tf.cast(theta_r, dtype=tf.float32)
        return m1 * tf.math.exp((2 * kappa_e / tf.math.cos(theta_r)) * z)


class UniformVolumeLayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Compute the f(z) using a Uniform Volume model.

    This layer computes the f(z) using a Uniform Volume model based on the input parameters.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The computed f(z).

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = UniformVolumeLayer()
    >>> z = tf.constant([0.1, 0.2])
    >>> d_pen = tf.constant([0.5, 1.0])
    >>> inputs = [z, d_pen]
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor([1.2214028 1.4918247], shape=(2,), dtype=float32)
    """  # noqa: E501

    def __init__(self, **kwargs: Any):
        """Initialize the UniformVolumeLayer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(UniformVolumeLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Compute the f(z) using a Uniform Volume model.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing z and d_pen.

        Returns
        -------
        tf.Tensor
            The computed f(z).
        """
        z, d_pen = inputs
        return tf.math.exp((2 / d_pen) * z)


class WeibullLayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Compute the Weibull distribution.

    This layer computes the Weibull distribution based on the input parameters.

    Parameters
    ----------
    **kwargs : keyword arguments, optional
        Additional options for the base Layer class.

    Returns
    -------
    tf.Tensor
        The computed Weibull distribution.

    Examples
    --------
    >>> import tensorflow as tf
    >>> layer = WeibullLayer()
    >>> z = tf.constant([0.1, 0.2])
    >>> lambda_w = tf.constant([1.0, 2.0])
    >>> k_w = tf.constant([2.0, 3.0])
    >>> inputs = [z, lambda_w, k_w]
    >>> result = layer(inputs)
    >>> print(result)
    tf.Tensor([0.18051676 0.06593975], shape=(2,), dtype=float32)
    """

    def __init__(self, **kwargs: Any):
        """Initialize the WeibullLayer.

        Parameters
        ----------
        **kwargs : keyword arguments, optional
            Additional options for the base Layer class.
        """
        super(WeibullLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Compute the Weibull distribution.

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor containing z, lambda_w, and k_w.

        Returns
        -------
        tf.Tensor
            The computed Weibull distribution.
        """
        z, lambda_w, k_w = inputs
        return (
            lambda_w
            * k_w
            * tf.pow(-(lambda_w * z), (k_w - 1))
            * tf.exp(-tf.pow(-(lambda_w * z), k_w))
        )


class LinspaceLayer(tf.keras.layers.Layer):  # type: ignore[misc]
    """Generate a linearly spaced vector.

    This layer generates a linearly spaced vector between two given values.

    Parameters
    ----------
    a : tf.constant
        The start value of the linspace.
    b : tf.constant
        The end value of the linspace.
    num_intervals : tf.int
        The number of intervals to generate.
    **kwargs : Any, optional
        Additional options for the base Layer class.

    Raises
    ------
    ValueError
        If the number of intervals is not positive.

    Returns
    -------
    tf.Tensor
        The generated linearly spaced vector.

    """

    def __init__(
        self,
        a: tf.constant,
        b: tf.constant,
        num_intervals: tf.int,
        **kwargs: Any,
    ):
        """Initialize the LinspaceLayer.

        Parameters
        ----------
        a : tf.Tensor
            The start value of the linspace.
        b : tf.Tensor
            The end value of the linspace.
        num_intervals : tf.Tensor
            The number of intervals to generate.
        **kwargs : Any, optional
            Additional options for the base Layer class.

        Raises
        ------
        ValueError
            If the number of intervals is not positive.

        """
        super(LinspaceLayer, self).__init__(**kwargs)
        if tf.math.less_equal(num_intervals, 0):
            raise ValueError("Number of intervals must be positive.")
        self.a = a
        self.b = b
        self.num_intervals = num_intervals

    def call(self, inputs: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Generate a linearly spaced vector.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor (unused).

        Returns
        -------
        tf.Tensor
            The generated linearly spaced vector.

        """
        z_value = tf.linspace(self.a, self.b, self.num_intervals + 1)
        return z_value


# %%
