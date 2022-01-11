#=
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 2000 by Stephen L. Moshier
https://github.com/jeremybarnes/cephes/blob/master/bessel/j0.c
https://github.com/jeremybarnes/cephes/blob/master/bessel/j1.c
=#
function bessely0(x::Float64)
    T = Float64
    if x <= zero(x)
        if iszero(x)
            return -Inf64
        else
            return throw(DomainError(x, "NaN result for non-NaN input."))
        end
    elseif isinf(x)
        return zero(x)
    end
    if x <= 5.0
        z = x * x
        w = evalpoly(z, YP_y0(T)) / evalpoly(z, YQ_y0(T))
        w += TWOOPI(Float64) * log(x) * besselj0(x)
        return w
    else
        w = 5.0 / x
        z = 25.0 / (x * x)
        p = evalpoly(z, PP_y0(T)) / evalpoly(z, PQ_y0(T))
        q = evalpoly(z, QP_y0(T)) / evalpoly(z, QQ_y0(T))
        xn = x - PIO4(T)
        p = p * sin(xn) + w * q * cos(xn);
        return p * SQ2OPI(T) / sqrt(x)
    end
end
function bessely1(x::Float64)
    T = Float64
    if x <= zero(x)
        if iszero(x)
            return -Inf64
        else
            return throw(DomainError(x, "NaN result for non-NaN input."))
        end
    elseif isinf(x)
        return zero(x)
    end

    if x <= 5.0
        z = x * x
        w = x * (evalpoly(z, YP_y1(T)) / evalpoly(z, YQ_y1(T)))
        w += TWOOPI(T) * (besselj1(x) * log(x) - inv(x))
        return w
    else
        w = 5.0 / x
        z = w * w
        p = evalpoly(z, PP_j1(T)) / evalpoly(z, PQ_j1(T))
        q = evalpoly(z, QP_j1(T)) / evalpoly(z, QQ_j1(T))
        xn = x - THPIO4(T)
        p = p * sin(xn) + w * q * cos(xn)
        return p * SQ2OPI(T) / sqrt(x)
    end
end
#=
function bessely(n::Int, x)
    if n < 0
        n = -n
        if n & 1 == 0
            sign = 1
        else sign = -1
        end
    else
        sign = 1
    end

    if n == 0
        return sign * bessely0(x)
    elseif n == 1
        return sign * bessely1(x)
    end

    if x <= 0.0
        return NaN
    end

    anm2 = bessely0(x)
    anm1 = bessely1(x)
    an = zero(x)

    k = 1
    r = 2 * k

    for _ in 1:n
        an = r * anm1 / x - anm2
        anm2 = anm1
        anm1 = an
        r += 2.0
    end

    return sign * an
end
=#