x = 0.001:0.001:1000.0
@test SpecialFunctions.bessely0.(x) ≈ bessely0.(x)
@test isinf(bessely0(0.0))