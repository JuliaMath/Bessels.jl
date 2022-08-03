# forward recurrence relation for besselk and besseli
# outputs both (bessel(x, nu_end), bessel(x, nu_end+1)
# x = 0.1; k0 = besselk(0,x); k1 = besselk(1,x);
# besselk_up_recurrence(x, k1, k0, 1, 5) will give besselk(5, x)
@inline function besselk_up_recurrence(x, jnu, jnum1, nu_start, nu_end)
    x2 = 2 / x
    while nu_start < nu_end + 0.5 # avoid inexact floating points when nu isa float
        jnum1, jnu = jnu, muladd(nu_start*x2, jnu, jnum1)
        nu_start += 1
    end
    return jnum1, jnu
end

# forward recurrence relation for besselj and bessely
# outputs both (bessel(x, nu_end), bessel(x, nu_end+1)
# x = 0.1; y0 = bessely0(x); y1 = bessely1(x);
# besselj_up_recurrence(x, y1, y0, 1, 5) will give bessely(5, x)
@inline function besselj_up_recurrence(x, jnu, jnum1, nu_start, nu_end)
    x2 = 2 / x
    while nu_start < nu_end + 0.5 # avoid inexact floating points when nu isa float
        jnum1, jnu = jnu, muladd(nu_start*x2, jnu, -jnum1)
        nu_start += 1
    end
    return jnum1, jnu
end
# backward recurrence relation for besselj and bessely
# outputs both (bessel(x, nu_end), bessel(x, nu_end-1)
# x = 0.1; j10 = besselj(10, x); j11 = besselj(11, x);
# besselj_down_recurrence(x, j10, j11, 10, 1) will give besselj(1, x)
@inline function besselj_down_recurrence(x, jnu, jnup1, nu_start, nu_end)
    x2 = 2 / x
    while nu_start > nu_end - 0.5
        jnup1, jnu = jnu, muladd(nu_start*x2, jnu, -jnup1)
        nu_start -= 1
    end
    return jnup1, jnu
end


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

