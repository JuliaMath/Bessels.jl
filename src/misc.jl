# function to more accurately compute cos(x + xn)
# see https://github.com/heltonmc/Bessels.jl/pull/13
# written by @oscardssmith
function cos_sum(x, xn)
    s = x + xn
    n, r = Base.Math.rem_pio2_kernel(s)
    lo = r.lo - ((s - x) - xn)
    hi = r.hi + lo
    y = Base.Math.DoubleFloat64(hi, r.hi-hi+lo)
    n = n&3
    if n == 0
        return Base.Math.cos_kernel(y)
    elseif n == 1
        return -Base.Math.sin_kernel(y)
    elseif n == 2
        return -Base.Math.cos_kernel(y)
    else
        return Base.Math.sin_kernel(y)
    end
end
# function to more accurately compute sin(x + xn)
function sin_sum(x, xn)
    s = x + xn
    n, r = Base.Math.rem_pio2_kernel(s)
    lo = r.lo - ((s - x) - xn)
    hi = r.hi + lo
    y = Base.Math.DoubleFloat64(hi, r.hi-hi+lo)
    n = n&3
    if n == 0
        return Base.Math.sin_kernel(y)
    elseif n == 1
        return Base.Math.cos_kernel(y)
    elseif n == 2
        return -Base.Math.sin_kernel(y)
    else
        return -Base.Math.cos_kernel(y)
    end
end
