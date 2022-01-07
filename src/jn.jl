
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
