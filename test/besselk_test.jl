# general array for testing input to SpecialFunctions.jl
x = 1e-5:0.01:50.0

### Tests for besselk0
k0_SpecialFunctions = SpecialFunctions.besselk.(0, x) 

k0_64 = besselk0.(Float64.(x))
k0_32 = besselk0.(Float32.(x))

# make sure output types match input types
@test k0_64[1] isa Float64
@test k0_32[1] isa Float32

# test against SpecialFunctions.jl
@test k0_64 ≈ k0_SpecialFunctions
@test k0_32 ≈ k0_SpecialFunctions

### Tests for besselk0x
k0x_SpecialFunctions = SpecialFunctions.besselkx.(0, x) 

k0x_64 = besselk0x.(Float64.(x))
k0x_32 = besselk0x.(Float32.(x))

# make sure output types match input types
@test k0x_64[1] isa Float64
@test k0x_32[1] isa Float32

# test against SpecialFunctions.jl
@test k0x_64 ≈ k0x_SpecialFunctions
@test k0x_32 ≈ k0x_SpecialFunctions

### Tests for besselk1
k1_SpecialFunctions = SpecialFunctions.besselk.(1, x) 

k1_64 = besselk1.(Float64.(x))
k1_32 = besselk1.(Float32.(x))

# make sure output types match input types
@test k1_64[1] isa Float64
@test k1_32[1] isa Float32

# test against SpecialFunctions.jl
@test k1_64 ≈ k1_SpecialFunctions
@test k1_32 ≈ k1_SpecialFunctions

### Tests for besselk1x
k1x_SpecialFunctions = SpecialFunctions.besselkx.(1, x) 

k1x_64 = besselk1x.(Float64.(x))
k1x_32 = besselk1x.(Float32.(x))

# make sure output types match input types
@test k1x_64[1] isa Float64
@test k1x_32[1] isa Float32

# test against SpecialFunctions.jl
@test k1x_64 ≈ k1x_SpecialFunctions
@test k1x_32 ≈ k1x_SpecialFunctions

### Tests for besselk
@test besselk(0, 2.0) == besselk0(2.0)
@test besselk(1, 2.0) == besselk1(2.0)

@test besselk(5, 8.0) ≈ SpecialFunctions.besselk(5, 8.0)
@test besselk(5, 88.0) ≈ SpecialFunctions.besselk(5, 88.0)

@test besselk(100, 3.9) ≈ SpecialFunctions.besselk(100, 3.9)
@test besselk(100, 234.0) ≈ SpecialFunctions.besselk(100, 234.0)

@test iszero(besselk(20, 1000.0))
@test isinf(besselk(250, 5.0))

### Tests for besselkx
@test besselkx(0, 12.0) == besselk0x(12.0)
@test besselkx(1, 89.0) == besselk1x(89.0)

@test besselkx(15, 82.123) ≈ SpecialFunctions.besselk(15, 82.123)*exp(82.123)
@test besselkx(105, 182.123) ≈ SpecialFunctions.besselk(105, 182.123)*exp(182.123)
