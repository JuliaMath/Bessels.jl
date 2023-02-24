# test the airy functions
# these are prone to some cancellation as they are indirectly calculated from relations to bessel functions
# this is amplified for negative arguments

# test Float64 for positive arguments
for line in eachline("data/airy/airy_positive_args.csv")
    x, aix, aiprimex, bix, biprimex = parse.(Float64, split(line))
    @test isapprox(airyaix(x), aix, rtol=4.4e-16)
    @test isapprox(airybix(x), bix, rtol=5.5e-16)
    @test isapprox(airyaiprimex(x), aiprimex, rtol=4.4e-16)
    @test isapprox(airybiprimex(x), biprimex, rtol=5.9e-16)
    if x < 105.0
        x_big = BigFloat(x)
        scale = exp(-2 * sqrt(x) * x / 3)
        if x < 2.0
            tol = 5e-16
        else
            tol = 2.6e-16 * sqrt(x) * x
        end
        @test isapprox(airyai(x), aix * scale, rtol=tol)
        @test isapprox(airybi(x), bix / scale, rtol=tol)
        @test isapprox(airyaiprime(x), aiprimex * scale, rtol=tol)
        @test isapprox(airybiprime(x), biprimex / scale, rtol=tol)
    end
end

# test Float64 for negative arguments
for line in eachline("data/airy/airy_negative_args.csv")
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
        @test isapprox(airyaiprime(x), aiprime, atol=tol2)
        @test isapprox(airybiprime(x), biprime, atol=tol2)
    end
end

#=
for x in [0.0; 1e-17:0.1:100.0]
    @test isapprox(airyai(x), SpecialFunctions.airyai(x), rtol=2e-13)
    @test isapprox(airyai(-x), SpecialFunctions.airyai(-x), rtol=3e-12)

    @test isapprox(airyaiprime(x), SpecialFunctions.airyaiprime(x), rtol=2e-13)
    @test isapprox(airyaiprime(-x), SpecialFunctions.airyaiprime(-x), rtol=5e-12)

    @test isapprox(airybi(x), SpecialFunctions.airybi(x), rtol=2e-13)
    @test isapprox(airybi(-x), SpecialFunctions.airybi(-x), rtol=5e-12)

    @test isapprox(airybiprime(x), SpecialFunctions.airybiprime(x), rtol=2e-13)
    @test isapprox(airybiprime(-x), SpecialFunctions.airybiprime(-x), rtol=5e-12)
end

# Float32
for x in [0.0; 0.5:0.5:30.0]
    @test isapprox(airyai(x), SpecialFunctions.airyai(x), rtol=2e-13)
    @test isapprox(airyai(-x), SpecialFunctions.airyai(-x), rtol=3e-12)

    @test isapprox(airyaiprime(x), SpecialFunctions.airyaiprime(x), rtol=2e-13)
    @test isapprox(airyaiprime(-x), SpecialFunctions.airyaiprime(-x), rtol=5e-12)

    @test isapprox(airybi(x), SpecialFunctions.airybi(x), rtol=2e-13)
    @test isapprox(airybi(-x), SpecialFunctions.airybi(-x), rtol=5e-12)

    @test isapprox(airybiprime(x), SpecialFunctions.airybiprime(x), rtol=2e-13)
    @test isapprox(airybiprime(-x), SpecialFunctions.airybiprime(-x), rtol=5e-12)
end

for x in [0.0, 0.01, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 50.0], a in 0:pi/12:2pi
    z = x*exp(im*a)
    @test isapprox(airyai(z), SpecialFunctions.airyai(z), rtol=5e-10)
    @test isapprox(airyaiprime(z), SpecialFunctions.airyaiprime(z), rtol=5e-10)
    @test isapprox(airybi(z), SpecialFunctions.airybi(z), rtol=1e-11)
    @test isapprox(airybiprime(z), SpecialFunctions.airybiprime(z), rtol=1e-11)
end

# test Inf

@test iszero(airyai(Inf))
@test iszero(airyaiprime(Inf))
@test isinf(airybi(Inf))
@test isinf(airybiprime(Inf))

@test airyai(Inf + 0.0im) === exp(-(Inf + 0.0im))
@test airyaiprime(Inf + 0.0im) === -exp(-(Inf + 0.0im))
@test airyai(-Inf + 0.0im) === 1 / (-Inf + 0.0im)
@test airyaiprime(-Inf + 0.0im) === 1 / (-Inf + 0.0im)
@test airybi(Inf + Inf*im) === exp((Inf + Inf*im))
@test airybi(-Inf + 10.0*im) === 1 / (-Inf + 10.0*im)
@test airybiprime(Inf + 0.0*im) === exp((Inf + 0.0*im))
@test airybiprime(-Inf + 0.0*im) === -1 / (-Inf + 0.0*im)


# test Float16 types
@test airyai(Float16(1.2)) isa Float16
@test airyai(ComplexF16(1.2)) isa ComplexF16
@test airyaiprime(Float16(1.9)) isa Float16
@test airyaiprime(ComplexF16(1.2)) isa ComplexF16
@test airybi(Float16(1.2)) isa Float16
@test airybi(ComplexF16(1.2)) isa ComplexF16
@test airybiprime(Float16(1.9)) isa Float16
@test airybiprime(ComplexF16(1.2)) isa ComplexF16
=#