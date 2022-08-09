# test the airy functions
# these are prone to some cancellation as they are indirectly calculated from relations to bessel functions
# this is amplified for negative arguments
# the implementations provided by SpecialFunctions.jl suffer from same inaccuracies
for x in [0.0; rand(10000)*100]
    @test isapprox(airyai(x), SpecialFunctions.airyai(x), rtol=1e-12)
    @test isapprox(airyai(-x), SpecialFunctions.airyai(-x), rtol=1e-9)

    @test isapprox(airyaiprime(x), SpecialFunctions.airyaiprime(x), rtol=1e-12)
    @test isapprox(airyaiprime(-x), SpecialFunctions.airyaiprime(-x), rtol=1e-9)

    @test isapprox(airybi(x), SpecialFunctions.airybi(x), rtol=1e-12)
    @test isapprox(airybi(-x), SpecialFunctions.airybi(-x), rtol=1e-9)

    @test isapprox(airybiprime(x), SpecialFunctions.airybiprime(x), rtol=1e-12)
    @test isapprox(airybiprime(-x), SpecialFunctions.airybiprime(-x), rtol=1e-9)
end
