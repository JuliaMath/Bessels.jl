# forward recurrence relation for besselk and besseli
# outputs both (bessel(x, nu_end), bessel(x, nu_end+1)
# x = 0.1; k0 = besselk(0,x); k1 = besselk(1,x);
# besselk_up_recurrence(x, k1, k0, 1, 5) will give besselk(5, x)
function besselk_up_recurrence(x, jnu, jnum1, nu_start, nu_end)
    x2 = 2 / x
    while nu_start < nu_end + 0.5 # avoid inexact floating points when nu isa float
        jnum1, jnu = jnu, muladd(nu_start*x2, jnu, jnum1)
        nu_start += 1
    end
    return jnum1, jnu
end
function besselk_up_recurrence!(out, x::T, nu_range) where T
    x2 = 2 / x
    k = 3
    for nu in nu_range[2:end-1]
        out[k] = muladd(nu*x2, out[k-1], out[k-2])
        k += 1
    end
    return out
end

# forward recurrence relation for besselj and bessely
# outputs both (bessel(x, nu_end), bessel(x, nu_end+1)
# x = 0.1; y0 = bessely0(x); y1 = bessely1(x);
# besselj_up_recurrence(x, y1, y0, 1, 5) will give bessely(5, x)
function besselj_up_recurrence(x::T, jnu, jnum1, nu_start, nu_end) where T
    x2 = 2 / x
    while nu_start < nu_end + 0.5 # avoid inexact floating points when nu isa float
        jnum1, jnu = jnu, muladd(nu_start*x2, jnu, -jnum1)
        nu_start += 1
    end
    # need to check if NaN resulted during loop from subtraction of infinities
    # this could happen if x is very small and nu is large which eventually results in overflow (-> -Inf)
    # we use this routine to generate bessely(nu::Int, x) in the forward direction starting from bessely0, bessely1
    # NaN inputs need to be checked before getting to this routine 
    return isnan(jnum1) ? (-T(Inf), -T(Inf)) : (jnum1, jnu)
end
function besselj_up_recurrence!(out, x::T, nu_range) where T
    x2 = 2 / x
    k = 3
    for nu in nu_range[2:end-1]
        out[k] = muladd(nu*x2, out[k-1], -out[k-2])
        k += 1
    end
    return out
end

# backward recurrence relation for besselj and bessely
# outputs both (bessel(x, nu_end), bessel(x, nu_end-1)
# x = 0.1; j10 = besselj(10, x); j11 = besselj(11, x);
# besselj_down_recurrence(x, j10, j11, 10, 1) will give besselj(1, x)
function besselj_down_recurrence(x, jnu, jnup1, nu_start, nu_end)
    x2 = 2 / x
    while nu_start > nu_end - 0.5
        jnup1, jnu = jnu, muladd(nu_start*x2, jnu, -jnup1)
        nu_start -= 1
    end
    return jnup1, jnu
end

function besselj_down_recurrence!(out, x::T, nu_range) where T
    x2 = 2 / x
    k = length(nu_range) - 2
    for nu in nu_range[end-1:-1:2]
        out[k] = muladd(nu*x2, out[k+1], -out[k+2])
        k -= 1
    end
    return out
end

#=
# currently not used
# backward recurrence relation for besselk and besseli
# outputs both (bessel(x, nu_end), bessel(x, nu_end-1)
# x = 0.1; k0 = besseli(10,x); k1 = besseli(11,x);
# besselk_down_recurrence(x, k0, k1, 10, 1) will give besseli(1, x)
@inline function besselk_down_recurrence(x, jnu, jnup1, nu_start, nu_end)
    x2 = 2 / x
    while nu_start > nu_end - 0.5 # avoid inexact floating points when nu isa float
        jnup1, jnu = jnu, muladd(nu_start*x2, jnu, jnup1)
        nu_start -= 1
    end
    return jnup1, jnu
end
=#
function besselk_down_recurrence!(out, x::T, nu_range) where T
    x2 = 2 / x
    k = length(nu_range) - 2
    for nu in nu_range[end-1:-1:2]
        out[k] = muladd(nu*x2, out[k+1], out[k+2])
        k -= 1
    end
    return out
end
