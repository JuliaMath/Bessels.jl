# test the airy functions
# these are prone to some cancellation as they are indirectly calculated from relations to bessel functions
# this is amplified for negative arguments
# the implementations provided by SpecialFunctions.jl suffer from same inaccuracies
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
