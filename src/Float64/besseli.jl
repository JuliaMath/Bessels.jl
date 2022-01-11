#=
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 2000 by Stephen L. Moshier
https://github.com/jeremybarnes/cephes/blob/master/bessel/i0.c
https://github.com/jeremybarnes/cephes/blob/master/bessel/i1.c
=#
function besseli0(x::Float64)
    T = Float64
    x = abs(x)
    if x <= 8.0
        y = x / 2.0 - 2.0
        return exp(x) * chbevl(y, A_i0(T))
    else
        return exp(x) * chbevl(32.0 / x - 2.0, B_i0(T)) / sqrt(x)
    end
end
function besseli0x(x::Float64)
    T = Float64
    x = abs(x)
    if x <= 8.0
        y = x / 2.0 - 2.0
        return chbevl(y, A_i0(T))
    else
        return chbevl(32.0 / x - 2.0, B_i0(T)) / sqrt(x)
    end
end
function besseli1(x::Float64)
    T = Float64
    z = abs(x)
    if z <= 8.0
        y = z / 2.0 - 2.0
        z = chbevl(y, A_i1(T)) * z * exp(z)
    else
        z = exp(z) * chbevl(32.0 / z - 2.0, B_i1(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
function besseli1x(x::Float64)
    T = Float64
    z = abs(x)
    if z <= 8.0
        y = z / 2.0 - 2.0
        z = chbevl(y, A_i1(T)) * z
    else
        z = chbevl(32.0 / z - 2.0, B_i1(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
