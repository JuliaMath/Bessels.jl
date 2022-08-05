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

    Jnu = besselj_positive_args(abs_nu, abs_x)
    Ynu = bessely_positive_args(abs_nu, abs_x)
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
            spi, cpi = sincospi(abs_nu)
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return (cpi*Jnu - spi*Ynu) * sg, Ynu + 2im * besselj_positive_args(abs_nu, abs_x)
        end
    end
end

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
        Ynm1, Yn = besselj_up_recurrence(x, bessely1(x), bessely0(x), 1, nu-1)

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
    nu_shift = floor(nu) - ceil(Int, -1.5 + x + Base.Math._approx_cbrt(-411*x))
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

hankelh1(nu, x) = besselh(nu, 1, x)
hankelh2(nu, x) = besselh(nu, 2, x)
