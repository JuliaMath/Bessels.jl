#=
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 2000 by Stephen L. Moshier
https://github.com/jeremybarnes/cephes/blob/master/bessel/j0.c
https://github.com/jeremybarnes/cephes/blob/master/bessel/j1.c
=#
function besselj0(x::Float64)
    T = Float64
    x = abs(x)
    iszero(x) && return one(x)
    isinf(x) && return zero(x)

    if x <= 5
        z = x * x
        if x < 1.0e-5
            return 1.0 - z / 4.0
        end

        DR1 = 5.78318596294678452118E0
        DR2 = 3.04712623436620863991E1

        p = (z - DR1) * (z - DR2)
        p = p * evalpoly(z, RP_j0(T)) / evalpoly(z, RQ_j0(T))
        return p
    else
        w = 5.0 / x
        q = 25.0 / (x * x)

        p = evalpoly(q, PP_j0(T)) / evalpoly(q, PQ_j0(T))
        q = evalpoly(q, QP_j0(T)) / evalpoly(q, QQ_j0(T))
        xn = x - PIO4(T)
        p = p * cos(xn) - w * q * sin(xn)
        return p * SQ2OPI(T) / sqrt(x)
    end
end
function besselj0(x::Float32)
    T = Float32
    x = abs(x)
    iszero(x) && return zero(x)
    isinf(x) && return zero(x)

    if x <= 2.0f0
        z = x * x
        if x < 1.0f-3
            return 1.0f0 - 0.25f0 * z
        end
        DR1 = 5.78318596294678452118f0
        p = (z - DR1) * evalpoly(z, JP_j0(T))
        return p
    else
        q = inv(x)
        w = sqrt(q)
        p = w * evalpoly(q, MO_j0(T))
        w = q * q
        xn = q * evalpoly(w, PH_j0(T)) - PIO4(Float32)
        p = p * cos(xn + x)
        return p
    end
end

function besselj1(x::Float64)
    T = Float64
    x = abs(x)
    iszero(x) && return zero(x)
    isinf(x) && return zero(x)

    if x <= 5.0
        z = x * x
        w = evalpoly(z, RP_j1(T)) / evalpoly(z, RQ_j1(T))
        w = w * x * (z - 1.46819706421238932572e1) * (z - 4.92184563216946036703e1)
        return w
    else
        w = 5.0 / x
        z = w * w
        p = evalpoly(z, PP_j1(T)) / evalpoly(z, PQ_j1(T))
        q = evalpoly(z, QP_j1(T)) / evalpoly(z, QQ_j1(T))
        xn = x - THPIO4(T)
        p = p * cos(xn) - w * q * sin(xn)
        return p * SQ2OPI(T) / sqrt(x)
    end
end

function besselj1(x::Float32)
    x = abs(x)
    iszero(x) && return zero(x)
    isinf(x) && return zero(x)

    if x <= 2.0f0
        z = x * x
        Z1 = 1.46819706421238932572f1
        p = (z - Z1) * x * evalpoly(z, JP32)
        return p
    else
        q = inv(x)
        w = sqrt(q)
        p = w * evalpoly(q, MO132)
        w = q * q
        xn = q * evalpoly(w, PH132) - THPIO4(Float32)
        p = p * cos(xn + x)
        return p
    end
end

function besselj(n::Int, x::Float64)
    if n < 0
        n = -n
        if (n & 1) == 0
            sign = 1
        else
            sign = -1
        end
    else
        sign = 1
    end

    if x < zero(x)
        if (n & 1)
            sign = -sign
            x = -x
        end
    end

    if n == 0
        return sign * besselj0(x)
    elseif n == 1
        return sign * besselj1(x)
    elseif n == 2
        return sign * (2.0 * besselj1(x) / x  -  besselj0(x))
    end

    #if x < MACHEP
     #   return 0.0
    #end

    k = 40 # or 53
    pk = 2 * (n + k)
    ans = pk
    xk = x * x

    for _ in 1:k
        pk -= 2.0
        ans = pk - (xk / ans)
    end

    ans = x / ans

    pk = 1.0
    pkm1 = inv(ans)
    k = n - 1
    r = 2 * k

    for _ in 1:k
        pkm2 = (pkm1 * r  -  pk * x) / x
	    pk = pkm1
	    pkm1 = pkm2
	    r -= 2.0
    end
    if abs(pk) > abs(pkm1)
        ans = besselj1(x) / pk
    else
        ans = besselj0(x) / pkm1
    end

    return sign * ans
end
