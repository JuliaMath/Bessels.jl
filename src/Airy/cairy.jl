#                           Airy functions
#
#                       airyai(z), airybi(z)
#                   airyaiprime(z), airybiprime(z)
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

_airyai(z::ComplexF16) = ComplexF16(_airyai(ComplexF32(z)))

function _airyai(z::Complex{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < 2*T(π)/3
            return exp(-z)
        else
            return 1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airyai_large_args(z)[1]
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

Returns the derivative of the Airy function of the first kind, 1`\\operatorname{Ai}'(z)``.
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
    airy_large_argument_cutoff(z) && return airyai_large_args(z)[2]
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
    airy_large_argument_cutoff(z) && return airybi_large_args(z)[1]
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
    airy_large_argument_cutoff(z) && return airybi_large_args(z)[2]
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

#####
#####  Large argument expansion for airy functions
#####
airy_large_argument_cutoff(z::ComplexOrReal{Float64}) = abs(z) > 8.3
airy_large_argument_cutoff(z::ComplexOrReal{Float32}) = abs(z) > 4

function airyai_large_args(z::Complex{T}) where T
    if imag(z) < zero(T)
        out = conj.(airyaix_large_args(conj(z)))
    else
        out = airyaix_large_args(z)
    end
    return out .* exp(-2/3 * z * sqrt(z))
end
function airybi_large_args(z::Complex{T}) where T
    if imag(z) < zero(T)
        out = conj.(airybix_large_args(conj(z)))
    else
        out = airybix_large_args(z)
    end
    return out .* exp(2/3 * z * sqrt(z))
end

@inline function airyaix_large_args(z::Complex{T}) where T
    xsqr = sqrt(z)
    xsqrx =  Base.FastMath.inv_fast(z * xsqr)
    A, B, C, D = compute_airy_asy_coef(z, xsqrx)
    
    if (real(z) < 0.0) && abs(imag(z)) < sqrt(3)*abs(real(z))
        e = exp(4/3 * z * xsqr)
        ai = muladd(B*im, e, A)
        aip = muladd(-D*im, e, C)
    else
        ai = A
        aip = C
    end

    xsqr = sqrt(xsqr)
    return ai * Base.FastMath.inv_fast(xsqr) * inv(PIPOW3O2(T)), aip * xsqr * inv(PIPOW3O2(T))
end

# valid in 0 <= angle(z) <= pi
# use conjugation for bottom half plane
@inline function airybix_large_args(z::Complex{T}) where T
    xsqr = sqrt(z)
    xsqrx = Base.FastMath.inv_fast(z * xsqr)
    A, B, C, D = compute_airy_asy_coef(z, xsqrx)
    
    if (real(z) > 0.0) || abs(imag(z)) > sqrt(3)*abs(real(z))
        B *= 2
        D *= 2
    end

    e = exp(-4/3 * z * xsqr)
    xsqr = sqrt(xsqr)

    bi = muladd(A*im, e, B) * Base.FastMath.inv_fast(xsqr) * inv(PIPOW3O2(T))
    bip = muladd(C*im, e, -D) * xsqr * inv(PIPOW3O2(T))
    return bi, bip
end

@inline function compute_airy_asy_coef(z, xsqrx)
    invx3 = @fastmath inv(z^3)
    p = SIMDMath.horner_simd(invx3, pack_AIRY_ASYM_COEF)

    zvec = SIMDMath.ComplexVec{2, Float64}((xsqrx.re, xsqrx.re), (xsqrx.im, xsqrx.im))
    zvecn = SIMDMath.ComplexVec{2, Float64}((-xsqrx.re, -xsqrx.re), (-xsqrx.im, -xsqrx.im))

    pvec1 = SIMDMath.ComplexVec{2, Float64}((p.re[1], p.re[3]), (p.im[1], p.im[3]))
    pvec2 = SIMDMath.ComplexVec{2, Float64}((p.re[2], p.re[4]), (p.im[2], p.im[4]))

    a = SIMDMath.fmadd(zvec, pvec2, pvec1)
    b = SIMDMath.fmadd(zvecn, pvec2, pvec1)

    A = complex(b.re[1].value, b.im[1].value)
    B = complex(a.re[1].value, a.im[1].value)
    C = complex(b.re[2].value, b.im[2].value)
    D = complex(a.re[2].value, a.im[2].value)
    return A, B, C, D
end

# to generate asymptotic expansions then split into even and odd coefficients
# tuple(Float64.([3^k * gamma(k + 1//6) * gamma(k + 5//6) / (2^(2k+2) * gamma(k+1)) for k in big"0":big"24"])...)
# tuple(Float64.([(3)^k * gamma(k - 1//6) * gamma(k + 7//6) / (2^(2k+2) * gamma(k+1)) for k in big"0":big"24"])...)

const AIRY_ASYM_COEF = (
    (1.5707963267948966, 0.13124057851910487, 0.4584353787485384, 5.217255928936184, 123.97197893818594, 5038.313653002081, 312467.7049060495, 2.746439545069411e7, 3.2482560591146026e9, 4.97462635569055e11, 9.57732308323407e13, 2.2640712393216476e16, 6.447503420809101e18),
    (0.1636246173744684, 0.20141783231057064, 1.3848568733028765, 23.555289417250567, 745.2667344964557, 37835.063701047824, 2.8147130917899106e6, 2.8856687720069575e8, 3.8998976239149216e10, 6.718472897263214e12, 1.4370735281142392e15, 3.7367429394637446e17, 1.1608192617215053e20),
    (-1.5707963267948966, 0.15510250188621483, 0.4982993247266722, 5.515384839161109, 129.24738229725767, 5209.103946324185, 321269.61208650155, 2.812618811215662e7, 3.3166403972012258e9, 5.0676100258903735e11, 9.738286496397669e13, 2.298637212441062e16, 6.537678293827411e18),
    (0.22907446432425577, 0.22511404787652015, 1.4803642438754887, 24.70432792540913, 773.390007496322, 38999.21950723391, 2.8878225227454924e6, 2.950515261265541e8, 3.97712331943799e10, 6.837383921993536e12, 1.460066704564067e15, 3.7912939312807334e17, 1.176400728321794e20)
)

const pack_AIRY_ASYM_COEF = SIMDMath.pack_poly(AIRY_ASYM_COEF)
