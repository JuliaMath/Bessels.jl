@test besselj0(0.0) ≈ 1.0

x = 0.001:0.001:1000.0
j0_SF = SpecialFunctions.besselj0.(x)
j0_64 = besselj0.(Float64.(x))
j0_32 = besselj0.(Float32.(x))

@test j0_SF ≈ j0_64
@test j0_SF ≈ j0_32
