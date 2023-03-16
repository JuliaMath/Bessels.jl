# test the airy functions
# these are prone to some cancellation as they are indirectly calculated from relations to bessel functions
# this is amplified for negative arguments

# test Float64 for positive arguments
for line in eachline("data/airy/airy_positive_args.csv")
    T = Float64
    local x, aix, aiprimex, bix, biprimex
    x, aix, aiprimex, bix, biprimex = parse.(Float64, split(line))
    @test isapprox(airyaix(x), aix, rtol=4.4e-16)
    @test isapprox(airybix(x), bix, rtol=5.5e-16)
    @test isapprox(airyaiprimex(x), aiprimex, rtol=4.9e-16)
    @test isapprox(airybiprimex(x), biprimex, rtol=5.9e-16)
    if x < 105.0
        x_big = BigFloat(x)
        scale = exp(-2 * sqrt(x_big) * x_big / 3)
        if x < 2.0
            tol = 4e-16
        else
            tol = 2.4e-16 * sqrt(x) * x
        end
        @test isapprox(airyai(x), T(aix * scale), rtol=tol)
        @test isapprox(airybi(x), T(bix / scale), rtol=tol)
        @test isapprox(airyaiprime(x), T(aiprimex * scale), rtol=tol)
        @test isapprox(airybiprime(x), T(biprimex / scale), rtol=tol)
    end
end

# test Float64 for negative arguments
for line in eachline("data/airy/airy_negative_args.csv")
    local x, aix, aiprimex, bix, biprimex
    x, ai, aiprime, bi, biprime = parse.(Float64, split(line))
    if x >= -9.5
        tol = 2.4e-16
        @test isapprox(airyai(x), ai, rtol=tol)
        @test isapprox(airybi(x), bi, rtol=tol)
        @test isapprox(airyaiprime(x), aiprime, rtol=tol)
        @test isapprox(airybiprime(x), biprime, rtol=tol)
    elseif x >= -1e8
        tol = 0.8e-16 * abs(x)^(5/4)
        tol2 = 0.8e-16 * abs(x)^(7/4)
        @test isapprox(airyai(x), ai, atol=tol)
        @test isapprox(airybi(x), bi, atol=tol)
        @test isapprox(airybix(x), bi, atol=tol)
        @test isapprox(airyaiprime(x), aiprime, atol=tol2)
        @test isapprox(airybiprime(x), biprime, atol=tol2)
        @test isapprox(airybiprimex(x), biprime, atol=tol2)
    end
end

for x in [0.0, 0.01, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 50.0], a in 0:pi/12:2pi
    z = x*exp(im*a)
    @test isapprox(airyai(z), SpecialFunctions.airyai(z), rtol=5e-10)
    @test isapprox(airyaiprime(z), SpecialFunctions.airyaiprime(z), rtol=5e-10)
    @test isapprox(airybi(z), SpecialFunctions.airybi(z), rtol=1e-11)
    @test isapprox(airybiprime(z), SpecialFunctions.airybiprime(z), rtol=1e-11)
end


# test Inf
# results match mathematica using Limit[AiryAi[x], x -> Infinity]
# for scaled arguments Limit[AiryAi[x] * Exp[2 * Sqrt[x] * x / 3], x -> Infinity]
@test airyai(Inf) === 0.0
@test airyaiprime(Inf) === 0.0
@test airybi(Inf) === Inf
@test airybiprime(Inf) === Inf

@test airyaix(Inf) === 0.0
@test airyaiprimex(Inf) === -Inf
@test airybix(Inf) === 0.0
@test airybiprimex(Inf) === Inf

# test negative infinite arguments
@test_throws DomainError airyai(-Inf)
# @test airyaiprime(-Inf) === Nan # value is indeterminate
@test_throws DomainError airybi(-Inf)
@test_throws DomainError airybix(-Inf)
# @test airybiprime(-Inf) === Nan # value is indeterminate

@test airyai(Inf + 0.0im) === exp(-(Inf + 0.0im))
@test airyaiprime(Inf + 0.0im) === -exp(-(Inf + 0.0im))
@test airyai(-Inf + 0.0im) === 1 / (-Inf + 0.0im)
@test airyaiprime(-Inf + 0.0im) === 1 / (-Inf + 0.0im)
@test airybi(Inf + Inf*im) === exp((Inf + Inf*im))
@test airybi(-Inf + 10.0*im) === 1 / (-Inf + 10.0*im)
@test airybiprime(Inf + 0.0*im) === exp((Inf + 0.0*im))
@test airybiprime(-Inf + 0.0*im) === -1 / (-Inf + 0.0*im)

# test NaNs
for f in (:airyai, :airyaix, :airyaiprime, :airyaiprimex, :airybi, :airybix, :airybiprime, :airybiprimex)
    @test @eval isnan($f(NaN))
end

# test Float16 and Float32 types
for f in (:airyai, :airyaix, :airyaiprime, :airyaiprimex, :airybi, :airybix, :airybiprime, :airybiprimex), T in (:Float16, :Float32)
    @test @eval $f($T(1.2)) isa $T
end

# test ComplexF16 and ComplexF32 types
for f in (:airyai, :airyaiprime, :airybi, :airybiprime), T in (:ComplexF16, :ComplexF32)
    @test @eval $f($T(1.2 + 1.1im)) isa $T
end
