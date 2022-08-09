"""
    airyai(x)
Airy function of the first kind ``\\operatorname{Ai}(x)``.
External links: [DLMF](https://dlmf.nist.gov/9.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)
See also: [`airyaix`](@ref), [`airyaiprime`](@ref), [`airybi`](@ref)
"""
function airyai(x::T) where T
    if x > zero(T)
        z = 2 * x^(T(3)/2) / 3
        return sqrt(x / 3) * besselk(one(T)/3, z) / π
    elseif x < zero(T)
        x_abs = abs(x)
        z = 2 * x_abs^(T(3)/2) / 3
        Jv, Yv = besseljy_positive_args(one(T)/3, z)
        Jmv = (Jv - sqrt(T(3)) * Yv) / 2
        return sqrt(x_abs) * (Jmv + Jv) / 3
    elseif iszero(x)
        return inv(3^(T(2)/3) * GAMMA_TWO_THIRDS(T))
    end
end

"""
    airyaiprime(x)
Derivative of the Airy function of the first kind ``\\operatorname{Ai}'(x)``.
External links: [DLMF](https://dlmf.nist.gov/9.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)
See also: [`airyaiprimex`](@ref), [`airyai`](@ref), [`airybi`](@ref)
"""
function airyaiprime(x::T) where T
    if x > zero(T)
        z = 2 * x^(T(3)/2) / 3
        return -x * besselk(T(2)/3, z) / (sqrt(T(3)) * π)
    elseif x < zero(T)
        x_abs = abs(x)
        z = 2 * x_abs^(T(3)/2) / 3
        Jv, Yv = besseljy_positive_args(T(2)/3, z)
        Jmv = -(Jv + sqrt(T(3))*Yv) / 2
        return x_abs * (Jv - Jmv) / 3
    elseif iszero(x)
        return T(-0.2588194037928068)
    end
end
"""
    airybi(x)
Airy function of the second kind ``\\operatorname{Bi}(x)``.
External links: [DLMF](https://dlmf.nist.gov/9.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)
See also: [`airybix`](@ref), [`airybiprime`](@ref),  [`airyai`](@ref)
"""
function airybi(x::T) where T
    if x > zero(T)
        z = 2 * x^(T(3)/2) / 3
        return sqrt(x / 3) * (besseli(-one(T)/3, z) + besseli(one(T)/3, z))
    elseif x < zero(T)
        x_abs = abs(x)
        z = 2 * x_abs^(T(3)/2) / 3
        Jv, Yv = besseljy_positive_args(one(T)/3, z)
        Jmv = (Jv - sqrt(T(3)) * Yv) / 2
        return sqrt(x_abs/3) * (Jmv - Jv)
    elseif iszero(x)
        return inv(3^(T(1)/6) * GAMMA_TWO_THIRDS(T))
    end
end

"""
    airybiprime(x)
Derivative of the Airy function of the second kind ``\\operatorname{Bi}'(x)``.
External links: [DLMF](https://dlmf.nist.gov/9.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)
See also: [`airybiprimex`](@ref), [`airybi`](@ref), [`airyai`](@ref)
"""
function airybiprime(x::T) where T
    if x > zero(T)
        z = 2 * x^(T(3)/2) / 3
        return x * (besseli(T(2)/3, z) + besseli(-T(2)/3, z)) / sqrt(T(3))
    elseif x < zero(T)
        x_abs = abs(x)
        z = 2 * x_abs^(T(3)/2) / 3
        Jv, Yv = besseljy_positive_args(T(2)/3, z)
        Jmv = -(Jv + sqrt(T(3))*Yv) / 2
        return x_abs * (Jv + Jmv) / sqrt(T(3))
    elseif iszero(x)
        return T(0.4482883573538264)
    end
end

const GAMMA_TWO_THIRDS(::Type{Float64}) = 1.3541179394264005
const GAMMA_TWO_THIRDS(::Type{Float32}) = 1.3541179394264005f0

const GAMMA_ONE_THIRD(::Type{Float64}) = 2.6789385347077475
const GAMMA_ONE_THIRD(::Type{Float64}) = 2.6789385347077475f0