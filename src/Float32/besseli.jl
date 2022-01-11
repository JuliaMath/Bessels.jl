#	Modified Bessel function of order zero
#	Modified Bessel function of order zero,
#	exponentially scaled
#
#	Modified Bessel function of order one
#	Modified Bessel function of order one,
#	exponentially scaled
#=
Ported to Julia from:
Cephes Math Library Release 2.2:  June, 1992
Copyright 1984, 1987, 1992 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
https://github.com/jeremybarnes/cephes/blob/master/single/i0f.c
https://github.com/jeremybarnes/cephes/blob/master/single/i1f.c
=#
function besseli0(x::Float32)
    T = Float32
    if x < zero(x)
        x = -x
    end
    if x <= 8.0f0
        y = 0.5f0 * x - 2.0f0
        return exp(x) * chbevl(y, A_i0(T))
    else
        return exp(x) * chbevl(32.0f0 / x - 2.0f0, B_i0(T)) / sqrt(x)
    end
end
function besseli0x(x::Float32)
    T = Float32
    if x < zero(x)
        x = -x
    end
    if x <= 8.0f0
        y = 0.5f0 * x - 2.0f0
        return chbevl(y, A_i0(T))
    else
        return chbevl(32.0f0 / x - 2.0f0, B_i0(T)) / sqrt(x)
    end
end
function besseli1(x::Float32)
    T = Float32
    z = abs(x)
    if z <= 8.0f0
        y = 0.5f0 * z - 2.0f0
        z = chbevl(y, A_i1(T)) * z * exp(z)
    else
        z = exp(z) * chbevl(32.0f0 / z - 2.0f0, B_i1(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
function besseli1x(x::Float32)
    T = Float32
    z = abs(x)
    if z <= 8.0f0
        y = 0.5f0 * z - 2.0f0
        z = chbevl(y, A_i1(T)) * z
    else
        z = chbevl(32.0f0 / z - 2.0f0, B_i1(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
