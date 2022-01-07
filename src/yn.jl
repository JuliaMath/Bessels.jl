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
