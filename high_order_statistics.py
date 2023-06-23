import numpy
import pywt

def cumulants(signal:numpy.ndarray):
    """
    计算4阶以下所有累积量
    signal: [Batch, 2], [[I1,Q1], [I2, Q2], ...]

    return cumulant_value: Cumulant of the specified order.
    """

    # 计算动量矩
    x = signal[:, :, 0] + 1j * signal[:, :, 1]
    e_x = numpy.mean(x, axis=-1)
    i = numpy.conjugate(x)
    xx = numpy.mean(x * x, axis=-1)
    xi = numpy.mean(x * i, axis=-1)
    xxx = numpy.mean(x * x * x, axis=-1)
    xxi = numpy.mean(x * x * i, axis=-1)
    xii = numpy.mean(x * i * i, axis=-1)
    xxxx = numpy.mean(x * x * x * x, axis=-1)
    xxxi = numpy.mean(x * x * x * i, axis=-1)
    xxii = numpy.mean(x * x * i * i, axis=-1)
    xiii = numpy.mean(x * i * i * i, axis=-1)

    # 计算累积量
    c_20 = xx
    c_21 = xi
    c_40 = xxxx - 3 * c_20 * c_20
    c_41 = xxxi - 3 * c_20 * c_21
    c_42 = xxii - c_20 * numpy.conjugate(c_20) + 2 * c_21 * c_21
    energy = c_21
    energy_2 = c_21 * c_21

    return numpy.abs(c_20 / energy), numpy.abs(c_40 / energy_2), numpy.abs(c_41 / energy_2), numpy.abs(c_42 / energy_2)

def wavelet_transform(signal:numpy.ndarray):
    """
    Parameters:
        signal (ndarray): Complex-valued signal.
    Returns:
        coeffs: Coefficients of the wavelet transform.
    """
    x = signal[:, 0] + 1j * signal[:, 1]
    coeffs = pywt.wavedec(x, 'db4', level=4)

    return coeffs

if __name__=="__main__":
    # A Unit Test
    # Generate a complex-valued stationary random process
    N = 1000  # Number of samples
    signal = numpy.random.normal(0, 1, size=(1, N, 2))

    # Perform the wavelet transform
    cumulants = cumulants(signal)
    coeffs = wavelet_transform(signal)

    print(cumulants)
    print(len(coeffs), numpy.shape(coeffs[0]), numpy.shape(coeffs[1]), numpy.shape(coeffs[2]), numpy.shape(coeffs[3]), numpy.shape(coeffs[4]))
