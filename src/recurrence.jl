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