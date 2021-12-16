@test besselj0(0.0) ≈ 1.0
x = 0.0:0.001:1000.0
@test SpecialFunctions.besselj0.(x) ≈ besselj0.(x)
