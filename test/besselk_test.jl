# general array for testing input to SpecialFunctions.jl
x = 0.01:0.01:10.0

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
