# test the airy functions
# these are prone to some cancellation as they are indirectly calculated from relations to bessel functions
# this is amplified for negative arguments

function arb_airyaix(x)
    x = ArbNumerics.ArbFloat(x)
    return Float64(ArbNumerics.airyai(x) * exp(2 * x * sqrt(x) / 3))
end
function arb_airyaiprimex(x)
    x = ArbNumerics.ArbFloat(x)
    return Float64(ArbNumerics.airyaiprime(x) * exp(2 * x * sqrt(x) / 3))
end
function arb_airyai(x)
    x = ArbNumerics.ArbFloat(x)
    return Float64(ArbNumerics.airyai(x))
end
function arb_airyaiprime(x)
    x = ArbNumerics.ArbFloat(x)
    return Float64(ArbNumerics.airyaiprime(x))
end
function arb_airybi(x)
    x = ArbNumerics.ArbFloat(x)
    return Float64(ArbNumerics.airybi(x))
end
function arb_airybiprime(x)
    x = ArbNumerics.ArbFloat(x)
    return Float64(ArbNumerics.airybiprime(x))
end
function arb_airybix(x)
    x = ArbNumerics.ArbFloat(x)
    return Float64(ArbNumerics.airybi(x) * exp(-2 * x * sqrt(complex(x)) / 3))
end
function arb_airybiprimex(x)
    x = ArbNumerics.ArbFloat(x)
    return Float64(ArbNumerics.airybiprime(x) * exp(-2 * x * sqrt(complex(x)) / 3))
end

x = rand(10000)*105.0
for _x in x
    @show _x
    @test isapprox(airyai(_x), arb_airyai(_x), rtol=5e-13)
    @test isapprox(airyaiprime(_x), arb_airyaiprime(_x), rtol=6e-13)
    @test isapprox(airyai(-_x), arb_airyai(-_x), atol=5e-14)
    @test isapprox(airyaiprime(-_x), arb_airyaiprime(-_x), rtol=5e-11, atol=1e-13)


    @test isapprox(Bessels.airyaix(_x), arb_airyaix(_x), rtol=8e-16)
    @test isapprox(Bessels.airyaiprimex(_x), arb_airyaiprimex(_x), rtol=8e-16)


    @test isapprox(airybi(_x), arb_airybi(_x), rtol=5e-13)
    @test isapprox(airybiprime(_x), arb_airybiprime(_x), rtol=5e-13)

    @test isapprox(airybi(-_x), arb_airybi(-_x), atol=4e-14)
    @test isapprox(airybiprime(-_x), arb_airybiprime(-_x), rtol=8e-11, atol=4e-14)

    @test isapprox(Bessels.airybix(_x), arb_airybix(_x), rtol=4e-15)
    @test isapprox(Bessels.airybiprimex(_x), arb_airybiprimex(_x), rtol=4e-15, atol=5e-14)

    @test isapprox(Bessels.airybix(-_x), arb_airybi(-_x), atol=8e-14) # scaled should be same for neg args

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