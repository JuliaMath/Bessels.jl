#                           Airy functions
#
#                       airyai(z), airybi(nu, z)
#                   airyaiprime(z), airybiprime(nu, z)
#
#    A numerical routine to compute the airy functions and their derivatives in the entire complex plane.
#    These routines are based on the methods reported in [1] which use a combination of the power series
#    for small arguments and a large argument expansion for (x > ~10). The primary difference between [1]
#    and what is used here is that the regions where the power series and large argument expansions
#    do not provide good results they are filled by relation to other special functions (besselk and besseli)
#    using https://dlmf.nist.gov/9.6 (NIST 9.6.E1 - 9.6.E9). In this case the power series of besseli is used and then besselk 
#    is calculated using the continued fraction approach. This method is described in more detail in src/besselk.jl.
#    However, care must be taken when computing besseli because when the imaginary component is much larger than the real part
#    cancellation will occur. This can be overcome by shifting the order of besseli to be much larger and then using the power series
#    and downward recurrence to get besseli(1/3, x). Another difficult region is when -10<x<-5 and the imaginary part is close to zero.
#    In this region we use rotation (see connection formulas http://dlmf.nist.gov/9.2.v) to shift to different region of complex plane
#    where algorithms show good convergence. If imag(z) == zero then we use the reflection identities to compute in terms of bessel functions.
#    In general, the cutoff regions compared to [1] are different to provide full double precision accuracy and to prioritize using the power series
#    and asymptotic expansion compared to other approaches.
#
# [1] Jentschura, Ulrich David, and E. Lötstedt. "Numerical calculation of Bessel, Hankel and Airy functions." 
#     Computer Physics Communications 183.3 (2012): 506-519.

"""
    airyai(z)

Returns the Airy function of the first kind, ``\\operatorname{Ai}(z)``, which is the solution to the Airy differential equation ``f''(z) - z f(z) = 0``.

```math
\\operatorname{Ai}(z) = \\frac{\\sqrt{3}}{2 \\pi} \\int_{0}^^{\\infty} \\exp{-\\frac{t^3}{3} - \\frac{z^3}{3t^3}} dt
```

Routine supports single and double precision (e.g., `Float32`,  `Float64`, `ComplexF64`) for real and complex arguments.

# Examples

```
julia> airyai(1.2)
0.10612576226331255

julia> airyai(1.2 + 1.4im)
-0.03254458873613304 - 0.14708163733976673im
```

External links: [DLMF](https://dlmf.nist.gov/9.2.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)

See also: [`airyaiprime`](@ref), [`airybi`](@ref)
"""
airyai(z::Number) = _airyai(float(z))

_airyai(x::Float16) = Float16(_airyai(Float32(x)))
_airyai(z::ComplexF16) = ComplexF16(_airyai(ComplexF32(z)))

function _airyai(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2*T(π)/3
            return exp(-z)
        else
            return 1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airyai_large_argument(z)
    airyai_power_series_cutoff(x, y) && return airyai_power_series(z)

    if x > zero(T)
        # use relation to besselk (http://dlmf.nist.gov/9.6.E1)
        zz = 2 * z * sqrt(z) / 3
        return sqrt(z / 3) * besselk_continued_fraction_shift(one(T)/3, zz) / T(π)
    else
        # z is close to the negative real axis
        # for imag(z) == 0 use reflection to compute in terms of bessel functions of first kind (http://dlmf.nist.gov/9.6.E5)
        # use computation for real numbers then convert to input type for stability
        # for imag(z) != 0 use rotation identity (http://dlmf.nist.gov/9.2.E14)
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs * sqrt(xabs) / 3
            Jv, Yv = besseljy_positive_args(one(T)/3, xx)
            Jmv = (Jv - sqrt(T(3)) * Yv) / 2
            return convert(eltype(z), sqrt(xabs) * (Jmv + Jv) / 3)
        else
            return cispi(one(T)/3) * _airyai(-z*cispi(one(T)/3))  + cispi(-one(T)/3) * _airyai(-z*cispi(-one(T)/3))
        end
    end
end

"""
    airyaiprime(z)

Returns the derivative of the Airy function of the first kind, ``\\operatorname{Ai}'(z)``.
Routine supports single and double precision (e.g., `Float32`,  `Float64`, `ComplexF64`) for real and complex arguments.

# Examples

```
julia> airyaiprime(1.2)
-0.13278537855722622

julia> airyaiprime(1.2 + 1.4im)
-0.02884977394212135 + 0.21117856532576215im
```

External links: [DLMF](https://dlmf.nist.gov/9.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)

See also: [`airyai`](@ref), [`airybi`](@ref)
"""
airyaiprime(z::Number) = _airyaiprime(float(z))

_airyaiprime(x::Float16) = Float16(_airyaiprime(Float32(x)))
_airyaiprime(z::ComplexF16) = ComplexF16(_airyaiprime(ComplexF32(z)))

function _airyaiprime(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2*T(π)/3
            return -exp(-z)
        else
            return 1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airyaiprime_large_argument(z)
    airyai_power_series_cutoff(x, y) && return airyaiprime_power_series(z)

    if x > zero(T)
        # use relation to besselk (http://dlmf.nist.gov/9.6.E2)
        zz = 2 * z * sqrt(z) / 3
        return -z * besselk_continued_fraction_shift(T(2)/3, zz) / (T(π) * sqrt(T(3)))
    else
        # z is close to the negative real axis
        # for imag(z) == 0 use reflection to compute in terms of bessel functions of first kind (http://dlmf.nist.gov/9.6.E5)
        # use computation for real numbers then convert to input type for stability
        # for imag(z) != 0 use rotation identity (http://dlmf.nist.gov/9.2.E14)
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs * sqrt(xabs) / 3
            Jv, Yv = besseljy_positive_args(T(2)/3, xx)
            Jmv = -(Jv + sqrt(T(3))*Yv) / 2
            return convert(eltype(z), xabs * (Jv - Jmv) / 3)
        else
            return -(cispi(T(2)/3) * _airyaiprime(-z * cispi(one(T)/3)) + cispi(-T(2)/3) * _airyaiprime(-z * cispi(-one(T)/3)))
        end
    end
end

"""
    airybi(z)

Returns the Airy function of the second kind, ``\\operatorname{Bi}(z)``, which is the second solution to the Airy differential equation ``f''(z) - z f(z) = 0``.
Routine supports single and double precision (e.g., `Float32`,  `Float64`, `ComplexF64`) for real and complex arguments.

# Examples

```
julia> airybi(1.2)
1.4211336756103483

julia> airybi(1.2 + 1.4im)
0.3150484065220768 + 0.7138432162853446im
```

External links: [DLMF](https://dlmf.nist.gov/9.2.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)

See also: [`airybiprime`](@ref), [`airyai`](@ref)
"""
airybi(z::Number) = _airybi(float(z))

_airybi(x::Float16) = Float16(_airybi(Float32(x)))
_airybi(z::ComplexF16) = ComplexF16(_airybi(ComplexF32(z)))

function _airybi(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2π/3
            return exp(z)
        else
            return 1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airybi_large_argument(z)
    airybi_power_series_cutoff(x, y) && return airybi_power_series(z)

    if x > zero(T)
        zz = 2 * z * sqrt(z) / 3
        shift = 20
        order = one(T)/3
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return sqrt(z/3) * (inu + inu2)
    else
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs * sqrt(xabs) / 3
            Jv, Yv = besseljy_positive_args(one(T)/3, xx)
            Jmv = (Jv - sqrt(T(3)) * Yv) / 2
            return convert(eltype(z), sqrt(xabs/3) * (Jmv - Jv))
        else
            return cispi(one(T)/3) * _airybi(-z * cispi(one(T)/3))  + cispi(-one(T)/3) * _airybi(-z*cispi(-one(T)/3))
        end
    end
end

"""
    airybiprime(z)

Returns the derivative of the Airy function of the second kind, ``\\operatorname{Bi}'(z)``.
Routine supports single and double precision (e.g., `Float32`,  `Float64`, `ComplexF64`) for real and complex arguments.

# Examples

```
julia> airybiprime(1.2)
1.221231398704895

julia> airybiprime(1.2 + 1.4im)
-0.5250248310153249 + 0.9612736841097036im
```

External links: [DLMF](https://dlmf.nist.gov/9.2), [Wikipedia](https://en.wikipedia.org/wiki/Airy_function)

See also: [`airybi`](@ref), [`airyai`](@ref)
"""
airybiprime(z::Number) = _airybiprime(float(z))

_airybiprime(x::Float16) = Float16(_airybiprime(Float32(x)))
_airybiprime(z::ComplexF16) = ComplexF16(_airybiprime(ComplexF32(z)))

function _airybiprime(z::ComplexOrReal{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2*T(π)/3
            return exp(z)
        else
            return -1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airybiprime_large_argument(z)
    airybi_power_series_cutoff(x, y) && return airybiprime_power_series(z)

    if x > zero(T)
        zz = 2 * z * sqrt(z) / 3
        shift = 20
        order = T(2)/3
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return z / sqrt(3) * (inu + inu2)
    else
        if iszero(y)
            xabs = abs(x)
            xx = 2 * xabs * sqrt(xabs) / 3
            Jv, Yv = besseljy_positive_args(T(2)/3, xx)
            Jmv = -(Jv + sqrt(T(3))*Yv) / 2
            return convert(eltype(z), xabs * (Jv + Jmv) / sqrt(T(3)))
        else
            return -(cispi(T(2)/3) * _airybiprime(-z*cispi(one(T)/3)) + cispi(-T(2)/3) * _airybiprime(-z*cispi(-one(T)/3)))
        end
    end
end

#####
##### Power series for airyai(x)
#####

# cutoffs for power series valid for both airyai and airyaiprime
airyai_power_series_cutoff(x::T, y::T) where T <: Float64 = x < 2 && abs(y) > -1.4*(x + 5.5)
airyai_power_series_cutoff(x::T, y::T) where T <: Float32 = x < 4.5f0 && abs(y) > -1.4f0*(x + 9.5f0)

function airyai_power_series(x::ComplexOrReal{T}; tol=eps(T)) where T
    S = eltype(x)
    iszero(x) && return S(0.3550280538878172)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_TWO_THIRDS(T)
    t2 = 3*x / GAMMA_ONE_THIRD(T)
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < tol * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(2)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(4)/3))
    end
    return (ai1*3^(-T(2)/3) - ai2*3^(-T(4)/3))
end
airyai_power_series(x::Float32) = Float32(airyai_power_series(Float64(x), tol=eps(Float32)))
airyai_power_series(x::ComplexF32) = ComplexF32(airyai_power_series(ComplexF64(x), tol=eps(Float32)))

#####
##### Power series for airyaiprime(x)
#####

function airyaiprime_power_series(x::ComplexOrReal{T}; tol=eps(T)) where T
    S = eltype(x)
    iszero(x) && return S(-0.2588194037928068)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_ONE_THIRD(T)
    t2 = 3*x2 / (2*GAMMA_TWO_THIRDS(T))
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < tol * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(1)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(5)/3))
    end
    return -ai1*3^(-T(1)/3) + ai2*3^(-T(5)/3)
end
airyaiprime_power_series(x::Float32) = Float32(airyaiprime_power_series(Float64(x), tol=eps(Float32)))
airyaiprime_power_series(x::ComplexF32) = ComplexF32(airyaiprime_power_series(ComplexF64(x), tol=eps(Float32)))

#####
##### Power series for airybi(x)
#####

# cutoffs for power series valid for both airybi and airybiprime
# has a more complicated validity as it works well close to positive real line and for small negative arguments also works for angle(z) ~ 2pi/3
# the statements are somewhat complicated but we want to hit this branch when we can as the other algorithms are 10x slower
# the Float32 cutoff can be simplified because it overlaps with the large argument expansion so there isn't a need for more complicated statements
airybi_power_series_cutoff(x::T, y::T) where T <: Float64 = (abs(y) > -1.4*(x + 5.5) && abs(y) < -2.2*(x - 4)) || (x > zero(T) && abs(y) < 3)
airybi_power_series_cutoff(x::T, y::T) where T <: Float32 = abs(complex(x, y)) < 9

function airybi_power_series(x::ComplexOrReal{T}; tol=eps(T)) where T
    S = eltype(x)
    iszero(x) && return S(0.6149266274460007)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_TWO_THIRDS(T)
    t2 = 3*x / GAMMA_ONE_THIRD(T)
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < tol * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(2)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(4)/3))
    end
    return (ai1*3^(-T(1)/6) + ai2*3^(-T(5)/6))
end
airybi_power_series(x::Float32) = Float32(airybi_power_series(Float64(x), tol=eps(Float32)))
airybi_power_series(x::ComplexF32) = ComplexF32(airybi_power_series(ComplexF64(x), tol=eps(Float32)))

#####
##### Power series for airybiprime(x)
#####

function airybiprime_power_series(x::ComplexOrReal{T}; tol=eps(T)) where T
    S = eltype(x)
    iszero(x) && return S(0.4482883573538264)
    MaxIter = 3000
    ai1 = zero(S)
    ai2 = zero(S)
    x2 = x*x
    x3 = x2*x
    t = one(S) / GAMMA_ONE_THIRD(T)
    t2 = 3*x2 / (2*GAMMA_TWO_THIRDS(T))
    
    for i in 0:MaxIter
        ai1 += t
        ai2 += t2
        abs(t) < tol * abs(ai1) && break
        t *= x3 * inv(9*(i + one(T))*(i + T(1)/3))
        t2 *= x3 * inv(9*(i + one(T))*(i + T(5)/3))
    end
    return (ai1*3^(T(1)/6) + ai2*3^(-T(7)/6))
end
airybiprime_power_series(x::Float32) = Float32(airybiprime_power_series(Float64(x), tol=eps(Float32)))
airybiprime_power_series(x::ComplexF32) = ComplexF32(airybiprime_power_series(ComplexF64(x), tol=eps(Float32)))

#####
#####  Large argument expansion for airy functions
#####
airy_large_argument_cutoff(z::ComplexOrReal{Float64}) = abs(z) > 8
airy_large_argument_cutoff(z::ComplexOrReal{Float32}) = abs(z) > 4

function airyai_large_argument(x::Real)
    x < zero(x) && return real(airyai_large_argument(complex(x)))
    return airy_large_arg_a(abs(x))
end

function airyai_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    a = airy_large_arg_a(z)
    if x < zero(T) && abs(y) < 5.5
        b = airy_large_arg_b(z)
        y >= zero(T) ? (return a + im*b) : (return a - im*b)
    end
    return a
end

function airyaiprime_large_argument(x::Real)
    x < zero(x) && return real(airyaiprime_large_argument(complex(x)))
    return airy_large_arg_c(abs(x))
end

function airyaiprime_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    c = airy_large_arg_c(z)
    if x < zero(T) && abs(y) < 5.5
        d = airy_large_arg_d(z)
        y >= zero(T) ? (return c + im*d) : (return c - im*d)
    end
    return c
end

function airybi_large_argument(x::Real)
    if x < zero(x)
        return 2*real(airy_large_arg_b(complex(x)))
    else
        return 2*(airy_large_arg_b(x))
    end
end

function airybi_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    b = airy_large_arg_b(z)
    abs(y) <= 1.7*(x - 6) && return 2*b

    check_conj = false
    if y < zero(T)
        z = conj(z)
        b = conj(b)
        y = abs(y)
        check_conj = true
    end

    a = airy_large_arg_a(z)
    if x < zero(T) && y < 5
        out = b + im*a
        check_conj && (out = conj(out))
        return out
    else
        out = 2*b + im*a
        check_conj && (out = conj(out))
        return out
    end
end

function airybiprime_large_argument(x::Real)
    if x < zero(x)
        return 2*real(airy_large_arg_d(complex(x)))
    else
        return 2*(airy_large_arg_d(x))
    end
end

function airybiprime_large_argument(z::Complex{T}) where T
    x, y = real(z), imag(z)
    d = airy_large_arg_d(z)
    abs(y) <= 1.7*(x - 6) && return 2*d

    check_conj = false
    if y < zero(T)
        z = conj(z)
        d = conj(d)
        y = abs(y)
        check_conj = true
    end

    c = airy_large_arg_c(z)
    if x < zero(T) && y < 5
        out = d + im*c
        check_conj && (out = conj(out))
        return out
    else
        out = 2*d + im*c
        check_conj && (out = conj(out))
        return out
    end
end

# see equations 24 and relations using eq 25 and 26 in [1]
function airy_large_arg_a(x::ComplexOrReal{T}; tol=eps(T)*40) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    t = GAMMA_ONE_SIXTH(T) * GAMMA_FIVE_SIXTHS(T) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < tol*abs(out) && break
        t *= -3*(i + one(T)/6) * (i + T(5)/6) / (a*(i + one(T)))
    end
    return out * exp(-a / 6) / (sqrt(T(π)^3) * sqrt(xsqr))
end

function airy_large_arg_b(x::ComplexOrReal{T}; tol=eps(T)*40) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    t = GAMMA_ONE_SIXTH(T) * GAMMA_FIVE_SIXTHS(T) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < tol*abs(out) && break
        t *= 3*(i + one(T)/6) * (i + T(5)/6) / (a*(i + one(T)))
    end
    return out * exp(a / 6) / (sqrt(T(π)^3) * sqrt(xsqr))
end

function airy_large_arg_c(x::ComplexOrReal{T}; tol=eps(T)*40) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    # use identities of gamma to relate to defined constants
    # t = gamma(-one(T) / 6) * gamma(T(7) / 6) / 4
    t = -GAMMA_FIVE_SIXTHS(T) * GAMMA_ONE_SIXTH(T) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < tol*abs(out) && break
        t *= -3*(i - one(T)/6) * (i + T(7)/6) / (a*(i + one(T)))
    end
    return out * exp(-a / 6) * sqrt(xsqr) / sqrt(T(π)^3)
end

function airy_large_arg_d(x::ComplexOrReal{T}; tol=eps(T)*40) where T
    S = eltype(x)
    MaxIter = 3000
    xsqr = sqrt(x)

    out = zero(S)
    # use identities of gamma to relate to defined constants
    # t = gamma(-one(T) / 6) * gamma(T(7) / 6) / 4
    t = -GAMMA_FIVE_SIXTHS(T) * GAMMA_ONE_SIXTH(T) / 4
    a = 4*xsqr*x
    for i in 0:MaxIter
        out += t
        abs(t) < tol*abs(out) && break
        t *= 3*(i - one(T)/6) * (i + T(7)/6) / (a*(i + one(T)))
    end
    return -out * exp(a / 6) * sqrt(xsqr) / sqrt(T(π)^3)
end

# negative arguments of airy functions oscillate around zero so as x -> -Inf it is more prone to cancellation
# to give best accuracy it is best to promote to Float64 numbers until the Float32 tolerance
airy_large_arg_a(x::Float32) = (airy_large_arg_a(Float64(x), tol=eps(Float32)))
airy_large_arg_a(x::ComplexF32) = (airy_large_arg_a(ComplexF64(x), tol=eps(Float32)))

airy_large_arg_b(x::Float32) = Float32(airy_large_arg_b(Float64(x), tol=eps(Float32)))
airy_large_arg_b(x::ComplexF32) = ComplexF32(airy_large_arg_b(ComplexF64(x), tol=eps(Float32)))

airy_large_arg_c(x::Float32) = Float32(airy_large_arg_c(Float64(x), tol=eps(Float32)))
airy_large_arg_c(x::ComplexF32) = ComplexF32(airy_large_arg_c(ComplexF64(x), tol=eps(Float32)))

airy_large_arg_d(x::Float32) = Float32(airy_large_arg_d(Float64(x), tol=eps(Float32)))
airy_large_arg_d(x::ComplexF32) = ComplexF32(airy_large_arg_d(ComplexF64(x), tol=eps(Float32)))
# calculates besselk from the power series of besseli using the continued fraction and wronskian
# this shift the order higher first to avoid cancellation in the power series of besseli along the imaginary axis
# for real arguments this is not needed because besseli can be computed stably over the entire real axis
function besselk_continued_fraction_shift(nu, x)
    shift = 20
    inu, inum1 = besseli_power_series_inu_inum1(nu + shift, x)
    inu, inum1 = besselk_down_recurrence(x, inum1, inu, nu + shift - 1, nu)
    H_knu = besselk_ratio_knu_knup1(nu-1, x)
    return 1 / (x * (inum1 + inu / H_knu))
end
