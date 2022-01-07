x = 0.01:0.01:100.0

y0_SpecialFunctions = SpecialFunctions.bessely0.(big.(x))
@assert y0_SpecialFunctions[1] isa BigFloat

y0_64 = bessely0.(Float64.(x))
y0_32 = bessely0.(Float32.(x))
y0_big = bessely0.(big.(x))

@test y0_64[1] isa Float64
@test y0_32[1] isa Float32
@test y0_big[1] isa BigFloat

@test y0_SpecialFunctions ≈ y0_64
@test y0_SpecialFunctions ≈ y0_32

@test isapprox(y0_big, y0_SpecialFunctions, atol=1.5e-34)


y1_SpecialFunctions = SpecialFunctions.bessely1.(big.(x))
@assert y1_SpecialFunctions[1] isa BigFloat

y1_64 = bessely1.(Float64.(x))


@test y1_64[1] isa Float64

@test y1_64 ≈ y1_SpecialFunctions

