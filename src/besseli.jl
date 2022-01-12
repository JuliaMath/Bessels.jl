#=
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 2000 by Stephen L. Moshier
https://github.com/jeremybarnes/cephes/blob/master/bessel/i0.c
https://github.com/jeremybarnes/cephes/blob/master/bessel/i1.c
=#
function besseli0(x::T) where T <: Union{Float32, Float64}
    if x <= 8
        y = muladd(x, T(.5), T(-2))
        return exp(x) * chbevl(y, A_i0(T))
    else
        return exp(x) * chbevl(T(32) / x - T(-2), B_i0(T)) / sqrt(x)
    end
end
function besseli0x(x::T) where T <: Union{Float32, Float64}
    x = abs(x)
    if x <= 8
        y = muladd(x, T(.5), T(-2))
        return chbevl(y, A_i0(T))
    else
        return chbevl(T(32) / x - T(-2), B_i0(T)) / sqrt(x)
    end
end
function besseli1(x::T) where T <: Union{Float32, Float64}
    z = abs(x)
    if x <= 8
        y = muladd(z, T(.5), T(-2))
        z = chbevl(y, A_i1(T)) * z * exp(z)
    else
        z = exp(z) * chbevl(T(32) / z - T(-2), B_i1(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
function besseli1x(x::T) where T <: Union{Float32, Float64}
    z = abs(x)
    if z <= 8
        y = muladd(z, T(.5), T(-2))
        z = chbevl(y, A_i1(T)) * z
    else
        z = chbevl(T(32) / z - T(-2), B_i1(T)) / sqrt(z)
    end
    if x < zero(x)
        z = -z
    end
    return z
end
