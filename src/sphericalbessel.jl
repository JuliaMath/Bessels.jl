function sphericalbesselj(nu::Real, x::T) where T
    isinteger(nu) && return sphericalbesselj(Int(nu), x)
    abs_nu = abs(nu)
    abs_x = abs(x)

    Jnu = sphericalbesselj_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        if x >= zero(T)
            return Jnu
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Jnu * cispi(abs_nu)
        end
    else
        Ynu = sphericalbessely_positive_args(abs_nu, abs_x)
        spi, cpi = sincospi(abs_nu)
        out = Jnu * cpi - Ynu * spi
        if x >= zero(T)
            return out
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return out * cispi(nu)
        end
    end
end

function sphericalbesselj(nu::Integer, x::T) where T
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    Jnu = sphericalbesselj_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        return x >= zero(T) ? Jnu : Jnu * sg
    else
        if x >= zero(T)
            return Jnu * sg
        else
            Ynu = sphericalbessely_positive_args(abs_nu, abs_x)
            spi, cpi = sincospi(abs_nu)
            return (cpi*Jnu - spi*Ynu) * sg
        end
    end
end

function sphericalbesselj_positive_args(nu::Real, x::T) where T
    if x^2 / (4*nu + 110) < eps(T)
        # small arguments power series expansion
        x2 = x^2 / 4
        coef = evalpoly(x2, (1, -inv(3/2 + nu), -inv(5 + nu), -inv(21/2 + nu), -inv(18 + nu)))
        a = sqrt(T(pi)/2) / (gamma(T(3)/2 + nu) * 2^(nu + T(1)/2))
        return x^nu * a * coef
    elseif isinteger(nu)
        if (x >= nu && nu < 250) || (x < nu && nu < 60)
            return sphericalbesselj_recurrence(nu, x)
        else
            return SQPIO2(T) * besselj(nu + 1/2, x) / sqrt(x)
        end
    else
        return SQPIO2(T) * besselj(nu + 1/2, x) / sqrt(x)
    end
end

# very accurate approach however need to consider some performance issues
# if recurrence is stable (x>=nu) can generate very fast up to orders around 250
# for larger orders it is more efficient to use other expansions
# if (x<nu) we can use forward recurrence from sphericalbesselj_recurrence and
# then use a continued fraction approach. However, for largish orders (>60) the
# continued fraction is slower converging and more efficient to use other methods
function sphericalbesselj_recurrence(nu::Integer, x::T) where T
    if x >= nu
        # forward recurrence if stable
        xinv = inv(x)
        s, c = sincos(x)
        sJ0 = s * xinv
        sJ1 = (sJ0 - c) * xinv

        nu_start = one(T)
        while nu_start < nu + 0.5
            sJ0, sJ1 = sJ1, muladd((2*nu_start + 1)*xinv, sJ1, -sJ0)
            nu_start += 1
        end
        return sJ0
    elseif x < nu
        # compute sphericalbessely with forward recurrence and use continued fraction
        sYnm1, sYn = sphericalbessely_forward_recurrence(nu, x)
        H = besselj_ratio_jnu_jnum1(nu + 3/2, x)
        return inv(x^2 * (H*sYnm1 - sYn))
    end
end



function sphericalbessely(nu::Real, x::T) where T
    isinteger(nu) && return sphericalbessely(Int(nu), x)
    abs_nu = abs(nu)
    abs_x = abs(x)

    Ynu = sphericalbessely_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        if x >= zero(T)
            return Ynu
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Ynu * cispi(-nu) + 2im * besselj_positive_args(abs_nu, abs_x) * cospi(abs_nu)
        end
    else
        Jnu = sphericalbesselj_positive_args(abs_nu, abs_x)
        spi, cpi = sincospi(abs_nu)
        if x >= zero(T)
            return Ynu * cpi + Jnu * spi
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return cpi * (Ynu * cispi(nu) + 2im * Jnu * cpi) + Jnu * spi * cispi(abs_nu)
        end
    end
end
function sphericalbessely(nu::Integer, x::T) where T
    abs_nu = abs(nu)
    abs_x = abs(x)
    sg = iseven(abs_nu) ? 1 : -1

    Ynu = sphericalbessely_positive_args(abs_nu, abs_x)
    if nu >= zero(T)
        if x >= zero(T)
            return Ynu
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Ynu * sg + 2im * sg * besselj_positive_args(abs_nu, abs_x)
        end
    else
        if x >= zero(T)
            return Ynu * sg
        else
            return throw(DomainError(x, "Complex result returned for real arguments. Complex arguments are currently not supported"))
            #return Ynu + 2im * besselj_positive_args(abs_nu, abs_x)
        end
    end
end

function sphericalbessely_positive_args(nu::Real, x::T) where T
    if besseljy_debye_cutoff(nu, x)
        # for very large orders use expansion nu >> x to avoid overflow in recurrence
        return SQPIO2(T) * besseljy_debye(nu + 1/2, x)[2] / sqrt(x)
    elseif isinteger(nu) && nu < 250
        return sphericalbessely_forward_recurrence(Int(nu), x)[1]
    else
        return SQPIO2(T) * bessely(nu + 1/2, x) / sqrt(x)
    end
end

function sphericalbessely_forward_recurrence(nu::Integer, x::T) where T
    xinv = inv(x)
    s, c = sincos(x)
    sY0 = -c * xinv
    sY1 = xinv * (sY0 - s)

    nu_start = one(T)
    while nu_start < nu + 0.5
        sY0, sY1 = sY1, muladd((2*nu_start + 1)*xinv, sY1, -sY0)
        nu_start += 1
    end
    return sY0, sY1
end
