x = rand(10000)*170
@test SpecialFunctions.gamma.(BigFloat.(x)) ≈ Bessels.gamma.(x)
@test SpecialFunctions.gamma.(BigFloat.(-x)) ≈ Bessels.gamma.(-x)
@test isnan(Bessels.gamma(NaN))
@test isinf(Bessels.gamma(Inf))
