#=
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 2000 by Stephen L. Moshier
https://github.com/jeremybarnes/cephes/blob/master/bessel/k0.c
https://github.com/jeremybarnes/cephes/blob/master/bessel/k1.c
=#
function besselk0(x::T) where T <: Union{Float32, Float64}
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end

    if x <= 2
        y = muladd(x, x, T(-2))
        y = chbevl(y, A_k0(T)) - log(T(.5) * x) * besseli0(x)
        return y
    else
        z = T(8) / x - T(2)
        return exp(-x) * chbevl(z, B_k0(T)) / sqrt(x)
    end
end

function besselk0x(x::T) where T <: Union{Float32, Float64}
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end
    if x <= 2
        y =  muladd(x, x, T(-2))
        y = chbevl(y, A_k0(T)) - log(T(.5) * x) * besseli0(x)
        return y * exp(x)
    else
        z = T(8) / x - T(2)
        return chbevl(z, B_k0(T)) / sqrt(x)
    end
end
function besselk1(x::T) where T <: Union{Float32, Float64}
    z = T(.5) * x
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end
    if x <= 2
        y = muladd(x, x, T(-2))
        y = log(z) * besseli1(x) + chbevl(y, A_k1(T)) / x
        return y
    else
        return exp(-x) * chbevl(T(8) / x - T(2), B_k1(T)) / sqrt(x)
    end
end
function besselk1x(x::T) where T <: Union{Float32, Float64}
    z = T(.5) * x
    if x <= zero(x)
        return throw(DomainError(x, "NaN result for non-NaN input."))
    end
    if x <= 2
        y = muladd(x, x, T(-2))
        y = log(z) * besseli1(x) + chbevl(y, A_k1(T)) / x
        return y * exp(x)
    else
        return chbevl(T(8) / x - T(2), B_k1(T)) / sqrt(x)
    end
end
