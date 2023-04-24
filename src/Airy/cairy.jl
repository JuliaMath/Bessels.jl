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

function _airyai(z::Complex{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < T(2π/3)
            return exp(-z)
        else
            return 1 / z
        end
    end

    x, y = reim(z)

    check_conj = false
    if y < zero(T)
        z = conj(z)
        check_conj = true
    end

    if airy_large_argument_cutoff(x, y)
        r = airyaix_large_args(z)[1] * exp(-T(2/3) * z * sqrt(z))
    elseif airyai_levin_cutoff(x, y)
        r = airyaix_levin(z, Val(17)) * exp(-T(2/3) * z * sqrt(z))
    elseif airyai_power_series_cutoff(x, y)
        r = airyai_power_series(z)[1]
    else
        c = cispi(one(T)/3)
        r = c * _airyai(-z*c)  + conj(c) * _airyai(-z*conj(c))
    end
    
    return check_conj ? conj(r) : r
end

function _airyaix(z::Complex{T}) where T <: Union{Float32, Float64}
    x, y = real(z), imag(z)

    check_conj = false
    if y < zero(T)
        z = conj(z)
        check_conj = true
    end

    if airy_large_argument_cutoff(x, y)
        r = airyaix_large_args(z)[1]
    elseif airyai_levin_cutoff(x, y)
        r = airyaix_levin(z, Val(17))
    elseif airyai_power_series_cutoff(x, y)
        r = airyai_power_series(z)[1] * exp(T(2/3) * z * sqrt(z))
    else
        c = cispi(one(T)/3)
        r =  exp(T(2/3) * z * sqrt(z)) * (c * _airyai(-z*c)  + conj(c) * _airyai(-z*conj(c)))
    end
    return check_conj ? conj(r) : r
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

function _airyaiprime(z::Complex{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < T(2π/3)
            return -exp(-z)
        else
            return 1 / z
        end
    end

    x, y = reim(z)

    check_conj = false
    if y < zero(T)
        z = conj(z)
        check_conj = true
    end

    if airy_large_argument_cutoff(x, y)
        r = airyaix_large_args(z)[2] * exp(-T(2/3) * z * sqrt(z))
    elseif airyai_levin_cutoff(x, y)
        r = airyaiprimex_levin(z, Val(17)) * exp(-T(2/3) * z * sqrt(z))
    elseif airyai_power_series_cutoff(x, y)
        r = airyai_power_series(z)[2]
    else
        r = -cispi(T(2/3)) * _airyaiprime(-z*cispi(one(T)/3))  - cispi(-T(2/3)) * _airyaiprime(-z*cispi(-one(T)/3))
    end
    
    return check_conj ? conj(r) : r
end

function _airyaiprimex(z::Complex{T}) where T <: Union{Float32, Float64}
    x, y = reim(z)

    check_conj = false
    if y < zero(T)
        z = conj(z)
        check_conj = true
    end

    if airy_large_argument_cutoff(x, y)
        r = airyaix_large_args(z)[2]
    elseif airyai_levin_cutoff(x, y)
        r = airyaiprimex_levin(z, Val(17))
    elseif airyai_power_series_cutoff(x, y)
        r = airyai_power_series(z)[2] * exp(T(2/3) * z * sqrt(z))
    else
        r = exp(T(2/3) * z * sqrt(z)) * (-cispi(T(2)/3) * _airyaiprime(-z*cispi(one(T)/3))  - cispi(-T(2)/3) * _airyaiprime(-z*cispi(-one(T)/3)))
    end
    
    return check_conj ? conj(r) : r
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

function _airybi(z::Complex{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < T(2π/3)
            return exp(z)
        else
            return 1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airybi_large_args(z)[1]
    airybi_power_series_cutoff(x, y) && return airybi_power_series(z)[1]

    if x > zero(T)
        zz = T(2/3) * z * sqrt(z)
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
            xx = T(2/3) * xabs * sqrt(xabs)
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

function _airybiprime(z::Complex{T}) where T <: Union{Float32, Float64}
    if ~isfinite(z)
        if abs(angle(z)) < T(2π/3)
            return exp(z)
        else
            return -1 / z
        end
    end
    x, y = real(z), imag(z)
    airy_large_argument_cutoff(z) && return airybi_large_args(z)[2]
    airybi_power_series_cutoff(x, y) && return airybi_power_series(z)[2]

    if x > zero(T)
        zz = T(2/3) * z * sqrt(z)
        shift = 20
        order = T(2/3)
        inu, inum1 = besseli_power_series_inu_inum1(order + shift, zz)
        inu, inum1 = besselk_down_recurrence(zz, inum1, inu, order + shift - 1, order)

        inu2, inum2 = besseli_power_series_inu_inum1(-order + shift, zz)
        inu2, inum2 = besselk_down_recurrence(zz, inum2, inu2, -order + shift - 1, -order)
        return z / sqrt(3) * (inu + inu2)
    else
        if iszero(y)
            xabs = abs(x)
            xx = T(2/3) * xabs * sqrt(xabs)
            Jv, Yv = besseljy_positive_args(T(2)/3, xx)
            Jmv = -(Jv + sqrt(T(3))*Yv) / 2
            return convert(eltype(z), xabs * (Jv + Jmv) / sqrt(T(3)))
        else
            return -(cispi(T(2/3)) * _airybiprime(-z*cispi(one(T)/3)) + cispi(-T(2/3)) * _airybiprime(-z*cispi(-one(T)/3)))
        end
    end
end

#####
##### Power series for airyai(z) and airybi(z)
#####

# power series returns value and derivative
function airyai_power_series(z::Complex{T}) where T
    z2 = z * z
    z3 = z2 * z
    p = SIMDMath.horner_simd(z3, pack_AIRYAI_POW_COEF)
    # TODO: use complex shufflevector to SIMD this muladd
    ai = muladd(-p[2], z, p[1])
    aip = muladd(p[4], z2, -p[3])
    return ai, aip
end

function airybi_power_series(z::Complex{T}) where T
    z2 = z * z
    z3 = z2 * z
    p = SIMDMath.horner_simd(z3, pack_AIRYBI_POW_COEF)
    bi = muladd(p[2], z, p[1])
    bip = muladd(p[4], z2, p[3])
    return bi, bip
end

# cutoffs for power series valid for both airyai and airyaiprime
airyai_power_series_cutoff(x::T, y::T) where T <: Float64 = x > -4.5 || (abs(angle(complex(x, y)) < 9pi/10))
#airyai_power_series_cutoff(x::T, y::T) where T <: Float64 = x < 2 && abs(y) > -1.4*(x + 5.5)
airyai_power_series_cutoff(x::T, y::T) where T <: Float32 = x < 4.5f0 && abs(y) > -1.4f0*(x + 9.5f0)

# cutoffs for power series valid for both airybi and airybiprime
# has a more complicated validity as it works well close to positive real line and for small negative arguments also works for angle(z) ~ 2pi/3
# the statements are somewhat complicated but we want to hit this branch when we can as the other algorithms are 10x slower
# the Float32 cutoff can be simplified because it overlaps with the large argument expansion so there isn't a need for more complicated statements
airybi_power_series_cutoff(x::T, y::T) where T <: Float64 = (abs(y) > -1.4*(x + 5.5) && abs(y) < -2.2*(x - 4)) || (x > zero(T) && abs(y) < 3)
airybi_power_series_cutoff(x::T, y::T) where T <: Float32 = abs(complex(x, y)) < 9

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
airy_large_argument_cutoff(x::Float64, y::Float64) =  (x^2 + y^2) > 68.89000000000001 # 8.3^2
airy_large_argument_cutoff(x::Float32, y::Float32) =  (x^2 + y^2) > 16.0f0 # 8.3^2

airy_large_argument_cutoff(z::ComplexOrReal{Float32}) = abs(z) > 4

# valid in 0 <= angle(z) <= pi
# use conjugation for bottom half plane
airyai_large_args(z::Complex{T}) where T = airyaix_large_args(z) .* exp(-2/3 * z * sqrt(z))

function airybi_large_args(z::Complex{T}) where T
    if imag(z) < zero(T)
        out = conj.(airybix_large_args(conj(z)))
    else
        out = airybix_large_args(z)
    end
    return out .* exp(T(2/3) * z * sqrt(z))
end

@inline function airyaix_large_args(z::Complex{T}) where T
    xsqr = sqrt(z)
    xsqrx =  Base.FastMath.inv_fast(z * xsqr)
    A, B, C, D = compute_airy_asy_coef(z, xsqrx)
    
    if (real(z) < 0.0) && abs(imag(z)) < sqrt(3)*abs(real(z))
        e = exp(T(4/3) * z * xsqr)
        ai = muladd(B*im, e, A)
        aip = muladd(-D*im, e, C)
    else
        ai = A
        aip = C
    end

    xsqr = sqrt(xsqr)
    return ai * Base.FastMath.inv_fast(xsqr) * inv(PIPOW3O2(T)), aip * xsqr * inv(PIPOW3O2(T))
end

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

    pvec1 = SIMDMath.ComplexVec((p[1], p[3]))
    pvec2 = SIMDMath.ComplexVec((p[2], p[4]))

    zvec = SIMDMath.ComplexVec{2, Float64}((xsqrx.re, xsqrx.re), (xsqrx.im, xsqrx.im))
    zvec = SIMDMath.fmul(zvec, pvec2)
    a = SIMDMath.fadd(pvec1, zvec)
    b = SIMDMath.fsub(pvec1, zvec)

    A, B, C, D = b[1], a[1], b[2], a[2]
    return A, B, C, D
end

@generated function airyaix_levin(x::Complex{T}, ::Val{N}) where {T <: Union{Float32, Float64}, N}
    :(
        begin
            xsqr = sqrt(x)
            out = zero(typeof(x))
            t = GAMMA_ONE_SIXTH(T) * GAMMA_FIVE_SIXTHS(T) / 4
            a = inv(4*xsqr*x)

            l = @ntuple $N i -> begin
                out += t
                t *= -3 * a * (i - 5//6) * (i - 1//6) / i
                invt = @fastmath inv(t)
                Vec{4, T}((reim(out * invt)..., reim(invt)...))
            end
            return @fastmath levin_transform(l) / (T(π)^(3//2) * sqrt(xsqr))
        end
    )
end
@generated function airyaiprimex_levin(x::Complex{T}, ::Val{N}) where {T <: Union{Float32, Float64}, N}
    :(
        begin
            xsqr = sqrt(x)
            out = zero(typeof(x))
            t = -GAMMA_ONE_SIXTH(T) * GAMMA_FIVE_SIXTHS(T) / 4
            a = inv(4*xsqr*x)

            l = @ntuple $N i -> begin
                out += t
                t *= -3 * a * (i - 7//6) * (i + 1//6) / i
                invt = @fastmath inv(t)
                Vec{4, T}((reim(out * invt)..., reim(invt)...))
            end
            return @fastmath levin_transform(l) * sqrt(xsqr) / T(π)^(3//2) 
        end
    )
end

airyai_levin_cutoff(x::T, y::T) where T <: Union{Float32, Float64} =  x > 2.0 || (x > 1.0 && y > 4.3)

# Asymptotic Expansion coefficients

# to generate asymptotic expansions then split into even and odd coefficients
# tuple(Float64.([3^k * gamma(k + 1//6) * gamma(k + 5//6) / (2^(2k+2) * gamma(k+1)) for k in big"0":big"24"])...)
# tuple(Float64.([(3)^k * gamma(k - 1//6) * gamma(k + 7//6) / (2^(2k+2) * gamma(k+1)) for k in big"0":big"24"])...)

const pack_AIRY_ASYM_COEF = SIMDMath.pack_poly((
    (1.5707963267948966, 0.13124057851910487, 0.4584353787485384, 5.217255928936184, 123.97197893818594, 5038.313653002081, 312467.7049060495, 2.746439545069411e7, 3.2482560591146026e9, 4.97462635569055e11, 9.57732308323407e13, 2.2640712393216476e16, 6.447503420809101e18),
    (0.1636246173744684, 0.20141783231057064, 1.3848568733028765, 23.555289417250567, 745.2667344964557, 37835.063701047824, 2.8147130917899106e6, 2.8856687720069575e8, 3.8998976239149216e10, 6.718472897263214e12, 1.4370735281142392e15, 3.7367429394637446e17, 1.1608192617215053e20),
    (-1.5707963267948966, 0.15510250188621483, 0.4982993247266722, 5.515384839161109, 129.24738229725767, 5209.103946324185, 321269.61208650155, 2.812618811215662e7, 3.3166403972012258e9, 5.0676100258903735e11, 9.738286496397669e13, 2.298637212441062e16, 6.537678293827411e18),
    (0.22907446432425577, 0.22511404787652015, 1.4803642438754887, 24.70432792540913, 773.390007496322, 38999.21950723391, 2.8878225227454924e6, 2.950515261265541e8, 3.97712331943799e10, 6.837383921993536e12, 1.460066704564067e15, 3.7912939312807334e17, 1.176400728321794e20)
))

# Power series coefficients

# airyai
# tuple(Float64.([3^(-2k - 2//3) / (gamma(k + 2//3) * gamma(k+1)) for k in big"0":big"31"])...)
# tuple(Float64.([3^(-2k - 4//3) / (gamma(k + 4//3) * gamma(k+1)) for k in big"0":big"31"])...)
# airyaiprime
# tuple(Float64.([3^(-2k - 1//3) / (gamma(k + 1//3) * gamma(k+1)) for k in big"0":big"31"])...)
# tuple(Float64.([3^(-2k - 5//3) / (gamma(k + 5//3) * gamma(k+1)) for k in big"0":big"31"])...)
# airybi
# tuple(Float64.([3^(-2k - 1//6) / (gamma(k + 2//3) * gamma(k+1)) for k in big"0":big"31"])...)
# tuple(Float64.([3^(-2k - 5//6) / (gamma(k + 4//3) * gamma(k+1)) for k in big"0":big"31"])...)
# airybiprime
# tuple(Float64.([3^(-2k + 1//6) / (gamma(k + 1//3) * gamma(k+1)) for k in big"0":big"31"])...)
# tuple(Float64.([3^(-2k - 7//6) / (gamma(k + 5//3) * gamma(k+1)) for k in big"0":big"31"])...)

const pack_AIRYAI_POW_COEF = SIMDMath.pack_poly((
    (0.3550280538878172, 0.05917134231463621, 0.00197237807715454, 2.7394139960479725e-5, 2.0753136333696763e-7, 9.882445873188935e-10, 3.2295574748983444e-12, 7.689422559281772e-15, 1.3930113332032197e-17, 1.984346628494615e-20, 2.280858193671971e-23, 2.159903592492397e-26, 1.7142092003907912e-29, 1.156686370034272e-32, 6.717110162800651e-36, 3.392479880202349e-39, 1.5037588121464314e-42, 5.897093380966397e-46, 2.060479867563381e-49, 6.455137429709841e-53, 1.8234851496355484e-56, 4.6684207619957715e-60, 1.0882099678311822e-63, 2.3192880814816327e-67, 4.536948516200377e-71, 8.174682011171851e-75, 1.3610859159460292e-78, 2.1004412283117732e-82, 3.0126810503611208e-86, 4.026571839563112e-90, 5.026931135534472e-94, 5.875328582906115e-98),
    (0.2588194037928068, 0.021568283649400565, 0.0005135305630809659, 5.705895145344065e-6, 3.657625093169273e-8, 1.5240104554871968e-10, 4.456170922477184e-13, 9.645391607093471e-16, 1.6075652678489119e-18, 2.1264090844562326e-21, 2.286461381135734e-24, 2.0378443682136668e-27, 1.5299131893495997e-30, 9.8071358291641e-34, 5.430307768086435e-37, 2.623337086032094e-40, 1.1153644073265705e-43, 4.2057481422570536e-47, 1.4160768155747653e-50, 4.2833539491069734e-54, 1.1703152866412495e-57, 2.902567675201512e-61, 6.56392509091251e-65, 1.3589907020522795e-68, 2.585598748196879e-72, 4.536138154731366e-76, 7.361470552955803e-80, 1.108321372019844e-83, 1.5522708291594454e-87, 2.0275219816607177e-91, 2.4756068152145513e-95, 2.8318540553815505e-99),
    (0.2588194037928068, 0.08627313459760226, 0.003594713941566761, 5.705895145344065e-5, 4.7549126211200545e-7, 2.438416728779515e-9, 8.46672475270665e-12, 2.1219861535605637e-14, 4.0189131696222797e-17, 5.953945436477451e-20, 7.088030281520775e-23, 6.928670851926468e-26, 5.660678800593519e-29, 3.9228543316656404e-32, 2.335032340277167e-35, 1.206735059574763e-38, 5.4652855959001957e-42, 2.1869890339736677e-45, 7.78842248566121e-49, 2.4843452904820447e-52, 7.138923248511622e-56, 1.8576433121289676e-59, 4.397829810911382e-63, 9.512934914365956e-67, 1.8874870861837215e-70, 3.4474649975958385e-74, 5.815561736835084e-78, 9.08823525056272e-82, 1.3194302047855285e-85, 1.7842193438614314e-89, 2.2528022018452416e-93, 2.6619428120586576e-97),
    (0.1775140269439086, 0.011834268462927242, 0.0002465472596443175, 2.4903763600436113e-6, 1.48236688097834e-8, 5.81320345481702e-11, 1.6147787374491722e-13, 3.3432271996877272e-16, 5.35773589693546e-19, 6.842574581015913e-22, 7.12768185522491e-25, 6.171153121406848e-28, 4.511076843133661e-31, 2.8211862683762735e-34, 1.526615946091057e-37, 7.21804229830287e-41, 3.0075176242928623e-44, 1.1126591284842257e-47, 3.6794283349346094e-51, 1.094091089781329e-54, 2.941105080057336e-58, 7.182185787685802e-62, 1.6003087762223267e-65, 3.2666029316642714e-69, 6.131011508378888e-73, 1.0616470144379027e-76, 1.7013573949325364e-80, 2.5306520823033415e-84, 3.5031175004199077e-88, 4.5242380219810255e-92, 5.4640555821026875e-96, 6.184556403059069e-100)
))

const pack_AIRYBI_POW_COEF = SIMDMath.pack_poly((
    (0.6149266274460007, 0.10248777124100013, 0.0034162590413666706, 4.7448042241203764e-5, 3.5945486546366485e-7, 1.7116898355412612e-9, 5.5937576324877815e-12, 1.3318470553542337e-14, 2.412766404627235e-17, 3.4369891803806764e-20, 3.9505622762996283e-23, 3.7410627616473754e-26, 2.9690974298788695e-29, 2.0034395613217741e-32, 1.163437608200798e-35, 5.875947516165647e-39, 2.604586664967042e-42, 1.0214065352811929e-45, 3.568855818592568e-49, 1.1180625998097016e-52, 3.1583689260161066e-56, 8.08594195088609e-60, 1.884834953586501e-63, 4.017124794515134e-67, 7.858225341383282e-71, 1.415896457906898e-74, 2.3574699598849448e-78, 3.6380709257483713e-82, 5.2181166462254325e-86, 6.9742270064493885e-90, 8.706900132895616e-94, 1.0176367616755045e-97),
    (0.4482883573538264, 0.03735736311281886, 0.0008894610264956872, 9.882900294396524e-6, 6.335192496408029e-8, 2.6396635401700117e-10, 7.718314444941556e-13, 1.6706308322384318e-15, 2.7843847203973864e-18, 3.683048571954215e-21, 3.960267281671199e-24, 3.52964998366417e-27, 2.6498873751232507e-30, 1.698645753284135e-33, 9.405568955061657e-37, 4.5437531183872734e-40, 1.9318678224435686e-43, 7.284569466227635e-47, 2.4527169919958367e-50, 7.418986666654073e-54, 2.0270455373371784e-57, 5.0273946858560976e-61, 1.136905175453663e-64, 2.353840942968246e-68, 4.478388399863482e-72, 7.856821754146459e-76, 1.275044101614161e-79, 1.919668927452817e-83, 2.688611943211228e-87, 3.511771085699096e-91, 4.28787678351538e-95, 4.904915103540814e-99),
    (0.4482883573538264, 0.14942945245127545, 0.0062262271854698105, 9.882900294396525e-5, 8.235750245330437e-7, 4.223461664272019e-9, 1.4664797445388956e-11, 3.67538783092455e-14, 6.960961800993466e-17, 1.0312536001471802e-19, 1.2276828573180715e-22, 1.2000809944458178e-25, 9.804583287956028e-29, 6.79458301313654e-32, 4.0443946506765124e-35, 2.0901264344581458e-38, 9.466152329973487e-42, 3.78797612243837e-45, 1.3489943455977101e-48, 4.3030122666593625e-52, 1.236497777775679e-55, 3.2175325989479025e-59, 7.617264675539542e-63, 1.6476886600777724e-66, 3.269223531900342e-70, 5.97118453315131e-74, 1.007284840275187e-77, 1.5741285205113097e-81, 2.2853201517295438e-85, 3.0903585554152047e-89, 3.901967872998996e-93, 4.6106201973283655e-97),
    (0.30746331372300034, 0.020497554248200024, 0.0004270323801708338, 4.313458385563978e-6, 2.567534753311892e-8, 1.0068763738478007e-10, 2.796878816243891e-13, 5.790639371105364e-16, 9.279870787027826e-19, 1.1851686828898885e-21, 1.2345507113436338e-24, 1.068875074756393e-27, 7.813414289154919e-31, 4.8864379544433515e-34, 2.6441763822745408e-37, 1.2502015991841802e-40, 5.209173329934083e-44, 1.9271821420399865e-47, 6.372956818915299e-51, 1.895021355609664e-54, 5.0941434290582364e-58, 1.2439910693670906e-61, 2.771816108215443e-65, 5.657922245795964e-69, 1.0619223434301734e-72, 1.838826568710257e-76, 2.946837449856181e-80, 4.383217982829363e-84, 6.067577495610968e-88, 7.836210119606054e-92, 9.464021883582192e-96, 1.071196591237373e-99)
))

# promote Float16 values
for internalf in (:airyai, :airyaiprime, :airybi, :airybiprime), T in (:ComplexF16,)
    @eval $internalf(x::$T) = $T($internalf(ComplexF32(x)))
end

# promote power series and asymptotic expansion to Float64 precision
for internalf in (:airyai_power_series, :airybi_power_series, :airyai_large_args, :airybi_large_args), T in (:ComplexF32,)
    @eval $internalf(x::$T) = $T.($internalf(ComplexF64(x)))
end
