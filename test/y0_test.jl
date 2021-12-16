x = 0.001:0.001:1000.0
y0_SF = SpecialFunctions.bessely0.(x)
y0_64 = bessely0.(Float64.(x))
y0_32 = bessely0.(Float32.(x))

@test y0_SF ≈ y0_64
@test y0_SF ≈ y0_32
