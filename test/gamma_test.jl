x = rand(10000)*170
@test SpecialFunctions.gamma.(BigFloat.(x)) ≈ Bessels.gamma.(x)
@test SpecialFunctions.gamma.(BigFloat.(-x)) ≈ Bessels.gamma.(-x)
@test isnan(Bessels.gamma(NaN))
@test isinf(Bessels.gamma(Inf))

x = [0, 1, 2, 3, 8, 15, 20, 30]
@test SpecialFunctions.gamma.(x) ≈ Bessels.gamma.(x)
