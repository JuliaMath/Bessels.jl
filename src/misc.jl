using Base.Math: sin_kernel, cos_kernel, rem_pio2_kernel, DoubleFloat64, DoubleFloat32
function sin_double(n,y)
    n = n&3
    if n == 0
        return sin_kernel(y)
    elseif n == 1
        return cos_kernel(y)
    elseif n == 2
        return -sin_kernel(y)
    else
        return -cos_kernel(y)
    end
end

"""
    computes sin(sum(xs)+sum(xlos)) where xs, xlos are sorted by absolute value
    and xlos are all <=pi/4 (and therefore don't need to be reduced)
    Doing this is much more accurate than the naive sin(sum(xs)+sum(xlos))
"""
function sin_sum(xs::Tuple{Vararg{Float64}}, xlos::Tuple{Vararg{Float64}})
    n = 0
    hi, lo = 0.0, 0.0
    for x in xs
        n_i, y = rem_pio2_kernel(x)
        n += n_i
        s = y.hi + hi
        lo += (y.hi - (s - hi)) + y.lo
        hi = s
    end
    for x in xlos
        s = x + hi
        lo += (x - (s - hi))
        hi = s
    end
    while hi > pi/4
        hi -= pi/2
        lo -= 6.123233995736766e-17
        n += 1
    end
    while hi < -pi/4
        hi += pi/2
        lo += 6.123233995736766e-17
        n -= 1
    end
    sin_double(n,DoubleFloat64(hi, lo))
end

function sin_sum(xs::Tuple{Vararg{Float32}}, xlos::Tuple{Vararg{Float32}})
    n = 0
    y = 0.0
    for x in xs
        n_i, y_i = rem_pio2_kernel(x)
        n += n_i
        y += y_i.hi
    end
    for x in xlos
        y += x
    end
    while y > pi/4
        y -= pi/2
        n += 1
    end
    while y < -pi/4
        y += pi/2
        n -= 1
    end

    sin_double(n,DoubleFloat32(y))
end
