#                           Hankel functions
#
#                           besseljy(nu, x)
#            besselh(nu, x), hankelh1(nu, x), hankelh2(nu, x)
#
#    A numerical routine to compute both Bessel functions of the first J_{ν}(x) and second kind Y_{ν}(x)
#    for real orders and arguments of positive or negative value. Please see notes in src/besselj.jl and src/bessely.jl
#    for details on implementation as most of the routine is similar. A key difference is when the methods to compute bessely
#    don't also give besselj. We then rely on a continued fraction approach to compute J_{ν}(x)/J_{ν-1}(x) to get J_{ν}(x)
#    from  Y_{ν}(x) and Y_{ν-1}(x)[1]. The continued fraction approach is more quickly converging when nu and x are small in magnitude.
#    When x and nu are large and nu = x + ϵ, we fall back to computing J_{ν}(x) and Y_{ν}(x) separately as this was found to be more efficient.
#    
# [1] Ratis, Yu L., and P. Fernández de Córdoba. "A code to calculate (high order) Bessel functions based on the continued fractions method." 
#     Computer physics communications 76.3 (1993): 381-388.
#

#####
##### Generic routine for `besseljy`
#####

"""
    Bessels.besseljy(ν, x::T) where T <: Float64

Returns the Bessel function of the first ``J_{ν}(x)`` and second ``Y_{ν}(x)`` kind for order ν.

This method may be faster than calling `besselj(ν, x)` and `bessely(ν, x)` separately depending on argument range.
Results may be slightly different than calling individual functions in some domains due to using different algorithms.

# Examples

```
julia> jn, yn = Bessels.besseljy(1.8, 1.2)
(0.2086667754797278, -1.0931173556626879)
```

See also: [`besselh`](@ref Bessels.besselh(nu, [k=1,] x)), [`besselj(nu,x)`](@ref Bessels.besselj)), [`bessely(nu,x)`](@ref Bessels.bessely))
"""
function besseljy(nu::Real, x::T) where T
    isinteger(nu) && return besseljy(Int(nu), x)
    abs_nu = abs(nu)
    abs_x = abs(x)

    Jnu, Ynu =  besseljy_positive_args(abs_nu, abs_x)
   
    if nu >= zero(T)
        if x >= zero(T)
            return Jnu, Ynu
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Ynu * cispi(-nu) + 2im * besselj_positive_args(abs_nu, abs_x) * cospi(abs_nu)
        end
    else
        spi, cpi = sincospi(abs_nu)
        out = Jnu * cpi - Ynu * spi
        if x >= zero(T)
            return out, Ynu * cpi + Jnu * spi
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return cpi * (Ynu * cispi(nu) + 2im * Jnu * cpi) + Jnu * spi * cispi(abs_nu)
        end
    end
end

function besseljy(nu::Integer, x::T) where T
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    Jnu, Ynu =  besseljy_positive_args(abs_nu, abs_x)

    if nu >= zero(T)
        if x >= zero(T)
            return Jnu, Ynu
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Jnu * sg, Ynu * sg + 2im * sg * besselj_positive_args(abs_nu, abs_x)
        end
    else
        if x >= zero(T)
            return Jnu * sg, Ynu * sg
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #spi, cpi = sincospi(abs_nu)
            #return (cpi*Jnu - spi*Ynu) * sg, Ynu + 2im * besselj_positive_args(abs_nu, abs_x)
        end
    end
end

#####
#####  `besseljy` for positive arguments and orders
#####

function besseljy_positive_args(nu::Real, x::T) where T
    nu == 0 && return (besselj0(x), bessely0(x))
    nu == 1 && return (besselj1(x), bessely1(x))
    iszero(x) && return (besselj_positive_args(nu, x), bessely_positive_args(nu, x))

    # x < ~nu branch see src/U_polynomials.jl
    besseljy_debye_cutoff(nu, x) && return besseljy_debye(nu, x)

    # large argument branch see src/asymptotics.jl
    besseljy_large_argument_cutoff(nu, x) && return besseljy_large_argument(nu, x)

    # x > ~nu branch see src/U_polynomials.jl on computing Hankel function
    if hankel_debye_cutoff(nu, x)
        H = hankel_debye(nu, x)
        return real(H), imag(H)
    end
    # use forward recurrence if nu is an integer up until the continued fraction becomes inefficient
    if isinteger(nu) && nu < 150
        Y0 = bessely0(x)
        Y1 = bessely1(x)
        Ynm1, Yn = besselj_up_recurrence(x, Y1, Y0, 1, nu-1)

        ratio_Jv_Jvm1 = besselj_ratio_jnu_jnum1(nu, x)
        Jn = 2 / (π*x * (Ynm1 - Yn / ratio_Jv_Jvm1))
        return Jn, Yn
    end

    # use power series for small x and for when nu > x
    if bessely_series_cutoff(nu, x) && besselj_series_cutoff(nu, x)
        Yn, Jn = bessely_power_series(nu, x)
        return Jn, Yn
    end

    # for x ∈ (6, 19) we use Chebyshev approximation and forward recurrence
    if besseljy_chebyshev_cutoff(nu, x)
        Yn, Ynp1 = bessely_chebyshev(nu, x)
        ratio_Jvp1_Jv = besselj_ratio_jnu_jnum1(nu+1, x)
        Jnp1 = 2 / (π*x * (Yn - Ynp1 / ratio_Jvp1_Jv))
        return Jnp1 / ratio_Jvp1_Jv, Yn
    end

    # at this point x > 19.0 (for Float64) and fairly close to nu
    # shift nu down and use the debye expansion for Hankel function (valid x > nu) then use forward recurrence
    nu_shift = ceil(nu) - floor(Int, -3//2 + x + Base.Math._approx_cbrt(-411*x)) + 2
    v2 = maximum((nu - nu_shift, modf(nu)[1] + 1))

    Hnu = hankel_debye(v2, x)
    Hnum1 = hankel_debye(v2 - 1, x)

    # forward recurrence is stable for Hankel when x >= nu
    if x >= nu
        H = besselj_up_recurrence(x, Hnu, Hnum1, v2, nu)[1]
        return real(H), imag(H)
    else
        # At this point besselj can not be calculated with forward recurrence
        # We could calculate it from bessely using the continued fraction approach like the following
        #        Yn, Ynp1 = besselj_up_recurrence(x, imag(Hnu), imag(Hnum1), v2, nu)
        #        ratio_Jvp1_Jv = besselj_ratio_jnu_jnum1(nu+1, x)
        #        Jnp1 = 2 / (π*x * (Yn - Ynp1 / ratio_Jvp1_Jv))
        #        return Jnp1 / ratio_Jvp1_Jv, Yn
        # However, the continued fraction approach is slowly converging for large arguments
        # We will fall back to computing besselj separately instead
        return besselj_positive_args(nu, x), besselj_up_recurrence(x, imag(Hnu), imag(Hnum1), v2, nu)[1]
    end
end

#####
#####  Continued fraction for J_{ν}(x)/J_{ν-1}(x)
#####

# implements continued fraction to compute ratio of J_{ν}(x)/J_{ν-1}(x)
# using equation 22 and 24 of [1]
# in general faster converging for small magnitudes of x and nu and nu >> x
function besselj_ratio_jnu_jnum1(n, x::T) where T
    MaxIter = 5000
    xinv = inv(x)
    xinv2 = 2 * xinv
    d = x / (n + n)
    a = d
    h = a
    b = muladd(2, n, 2) * xinv
    for _ in 1:MaxIter
        d = inv(b - d)
        a *= muladd(b, d, -1)
        h = h + a
        b = b + xinv2
        abs(a / h) <= eps(T) && break
    end
    return h
end

#####
#####  Hankel functions
#####

"""
    besselh(nu, [k=1,] x)

Returns the Bessel function of the third kind of order `nu` (the Hankel function).

```math
H^{(1)}_{\\nu}(x) = J_{\\nu}(x) + i Y_{\\nu}(x)
H^{(2)}_{\\nu}(x) = J_{\\nu}(x) - i Y_{\\nu}(x)
```

`k` must be 1 or 2, selecting [`hankelh1`](@ref) or [`hankelh2`](@ref), respectively.

# Examples

```
julia> besselh(1.2, 1, 9.2)
0.2513215427211038 + 0.08073652619125624im
```

See also: [`Bessels.besseljy`](@ref Bessels.besseljy(nu, x)), [`besselj(nu,x)`](@ref Bessels.besselj)), [`bessely(nu,x)`](@ref Bessels.bessely))
"""
function besselh(nu::Real, k::Integer, x)
    Jn, Yn = besseljy(nu, x)
    if k == 1
        return complex(Jn, Yn)
    elseif k == 2
        return complex(Jn, -Yn)
    else
        throw(ArgumentError("k must be 1 or 2"))
    end
end

function besselh(nu::AbstractRange, k::Integer, x::T) where T
    (nu[1] >= 0 && step(nu) == 1) || throw(ArgumentError("nu must be >= 0 with step(nu)=1"))
    if nu[end] < x
        out = Vector{Complex{T}}(undef, length(nu))
        out[1], out[2] = besselh(nu[1], k, x), besselh(nu[2], k, x)
        return besselj_up_recurrence!(out, x, nu)
    else
        Jn = besselj(nu, x)
        Yn = bessely(nu, x)
        if k == 1
            return complex.(Jn, Yn)
        elseif k == 2
            return complex.(Jn, -Yn)
        else
            throw(ArgumentError("k must be 1 or 2"))
        end
    end
end

"""
    hankelh1(nu, x)
Bessel function of the third kind of order `nu`, ``H^{(1)}_\\nu(x)``.
"""
hankelh1(nu, x) = besselh(nu, 1, x)

"""
    hankelh2(nu, x)
Bessel function of the third kind of order `nu`, ``H^{(2)}_\\nu(x)``.
"""
hankelh2(nu, x) = besselh(nu, 2, x)
