@test besselj0(0.0) ≈ 1.0

x = 0.01:0.01:100.0

j0_SpecialFunctions = SpecialFunctions.besselj0.(big.(x))
@assert j0_SpecialFunctions[1] isa BigFloat

j0_64 = besselj0.(Float64.(x))
j0_32 = besselj0.(Float32.(x))
j0_big = besselj0.(big.(x))

@test j0_64[1] isa Float64
@test j0_32[1] isa Float32
@test j0_big[1] isa BigFloat


@test j0_64 ≈ j0_SpecialFunctions
@test j0_32 ≈ j0_SpecialFunctions

@test isapprox(j0_big, j0_SpecialFunctions, atol=1.5e-34)



j1_SpecialFunctions = SpecialFunctions.besselj1.(big.(x))
@assert j1_SpecialFunctions[1] isa BigFloat

j1_64 = besselj1.(Float64.(x))


@test j1_64[1] isa Float64

@test j1_64 ≈ j1_SpecialFunctions

