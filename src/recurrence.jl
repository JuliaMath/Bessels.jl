# no longer used for besseli but could be used in future for Jn, Yn
#=
@inline function down_recurrence(x, in, inp1, nu, branch)
    # this prevents us from looping through large values of nu when the loop will always return zero
    (iszero(in) || iszero(inp1)) && return zero(x)
    (isinf(inp1) || isinf(inp1)) && return in

    inm1 = in
    x2 = 2 / x
    for n in branch:-1:nu+1
        a = x2 * n
        inm1 = muladd(a, in, inp1)
        inp1 = in
        in = inm1
    end
    return inm1
end
=#
@inline function up_recurrence(x, k0, k1, nu)
    nu == 0 && return k0
    nu == 1 && return k1

    # this prevents us from looping through large values of nu when the loop will always return zero
    (iszero(k0) || iszero(k1)) && return zero(x) 

    k2 = k0
    x2 = 2 / x
    for n in 1:nu-1
        a = x2 * n
        k2 = muladd(a, k1, k0)
        k0 = k1
        k1 = k2
    end
    return k2, k0
end

@inline function besselj_up_recurrence(x, jnu, jnum1, nu_start, nu_end)
    jnup1 = jnum1
    x2 = 2 / x
    for n in nu_start:nu_end
        a = x2 * n
        jnup1 = muladd(a, jnu, -jnum1)
        jnum1 = jnu
        jnu = jnup1
    end
    return jnup1, jnum1
end
@inline function besselj_down_recurrence(x, jnu, jnup1, nu_start, nu_end)
    jnum1 = jnup1
    x2 = 2 / x
    for n in nu_start:-1:nu_end
        a = x2 * n
        jnum1 = muladd(a, jnu, -jnup1)
        jnup1 = jnu
        jnu = jnum1
    end
    return jnum1, jnup1
end