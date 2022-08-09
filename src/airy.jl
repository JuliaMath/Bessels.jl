#                           Airy functions
#
#                       airyai(x), airybi(nu, x)
#                       airyaiprime(x), airybiprime(nu, x)
#
#    A numerical routine to compute the airy functions and their derivatives.
#    These routines use their relations to other special functions using https://dlmf.nist.gov/9.6.
#    Specifically see (NIST 9.6.E1 - 9.6.E9) for computation from the defined bessel functions.
#    For negative arguments these definitions are prone to some cancellation leading to higher errors.
#    In the future, these could be replaced with more custom routines as they depend on a single variable.
#
    
"""
    airyai(x)
Airy function of the first kind ``\\operatorname{Ai}(x)``.
"""
airyai(x::Real) = _airyai(float(x))

_airyai(x::Float16) = Float16(_airyai(Float32(x)))

function _airyai(x::T) where T <: Union{Float32, Float64}
    isnan(x) && return x
    x_abs = abs(x)
    z = 2 * x_abs^(T(3)/2) / 3

    if x > zero(T)
        return isinf(z) ? zero(T) : sqrt(x / 3) * besselk(one(T)/3, z) / π
    elseif x < zero(T)
        Jv, Yv = besseljy_positive_args(one(T)/3, z)
        Jmv = (Jv - sqrt(T(3)) * Yv) / 2
        return isinf(z) ? throw(DomainError(x, "airyai(x) is not defined.")) : sqrt(x_abs) * (Jmv + Jv) / 3
    elseif iszero(x)
        return T(0.3550280538878172)
    end
end

"""
    airyaiprime(x)
Derivative of the Airy function of the first kind ``\\operatorname{Ai}'(x)``.
"""
airyaiprime(x::Real) = _airyaiprime(float(x))

_airyaiprime(x::Float16) = Float16(_airyaiprime(Float32(x)))

function _airyaiprime(x::T) where T <: Union{Float32, Float64}
    isnan(x) && return x
    x_abs = abs(x)
    z = 2 * x_abs^(T(3)/2) / 3

    if x > zero(T)
        return isinf(z) ? zero(T) : -x * besselk(T(2)/3, z) / (sqrt(T(3)) * π)
    elseif x < zero(T)
        Jv, Yv = besseljy_positive_args(T(2)/3, z)
        Jmv = -(Jv + sqrt(T(3))*Yv) / 2
        return isinf(z) ? throw(DomainError(x, "airyaiprime(x) is not defined.")) : x_abs * (Jv - Jmv) / 3
    elseif iszero(x)
        return T(-0.2588194037928068)
    end
end

"""
    airybi(x)
Airy function of the second kind ``\\operatorname{Bi}(x)``.
"""
airybi(x::Real) = _airybi(float(x))

_airybi(x::Float16) = Float16(_airybi(Float32(x)))

function _airybi(x::T) where T <: Union{Float32, Float64}
    isnan(x) && return x
    x_abs = abs(x)
    z = 2 * x_abs^(T(3)/2) / 3

    if x > zero(T)
        return isinf(z) ? z : sqrt(x / 3) * (besseli(-one(T)/3, z) + besseli(one(T)/3, z))
    elseif x < zero(T)
        Jv, Yv = besseljy_positive_args(one(T)/3, z)
        Jmv = (Jv - sqrt(T(3)) * Yv) / 2
        return isinf(z) ? throw(DomainError(x, "airybi(x) is not defined.")) : sqrt(x_abs/3) * (Jmv - Jv)
    elseif iszero(x)
        return T(0.6149266274460007)
    end
end

"""
    airybiprime(x)
Derivative of the Airy function of the second kind ``\\operatorname{Bi}'(x)``.
"""
airybiprime(x::Real) = _airybiprime(float(x))

_airybiprime(x::Float16) = Float16(_airybiprime(Float32(x)))

function _airybiprime(x::T) where T <: Union{Float32, Float64}
    isnan(x) && return x
    x_abs = abs(x)
    z = 2 * x_abs^(T(3)/2) / 3

    if x > zero(T)
        return isinf(z) ? z : x * (besseli(T(2)/3, z) + besseli(-T(2)/3, z)) / sqrt(T(3))
    elseif x < zero(T)
        Jv, Yv = besseljy_positive_args(T(2)/3, z)
        Jmv = -(Jv + sqrt(T(3))*Yv) / 2
        return isinf(z) ? throw(DomainError(x, "airybiprime(x) is not defined.")) : x_abs * (Jv + Jmv) / sqrt(T(3))
    elseif iszero(x)
        return T(0.4482883573538264)
    end
end
