# general array for testing input to SpecialFunctions.jl
x32 = [-1.0; 0.0; 1e-6; 0.01:0.01:60.0]
x64 = [-1.0; 0.0; 1e-6; 0.01:0.01:600.0]

### Tests for besseli0
i032_SpecialFunctions = SpecialFunctions.besseli.(0, x32)
i064_SpecialFunctions = SpecialFunctions.besseli.(0, x64) 


i0_64 = besseli0.(Float64.(x64))
i0_32 = besseli0.(Float32.(x32))

# make sure output types match input types
@test i0_64[1] isa Float64
@test i0_32[1] isa Float32

# test against SpecialFunctions.jl
@test i0_64 ≈ i064_SpecialFunctions
@test i0_32 ≈ i032_SpecialFunctions

### Tests for besseli0x
i0x32_SpecialFunctions = SpecialFunctions.besselix.(0, x32)
i0x64_SpecialFunctions = SpecialFunctions.besselix.(0, x64) 

i0x_64 = besseli0x.(Float64.(x64))
i0x_32 = besseli0x.(Float32.(x32))

# make sure output types match input types
@test i0x_64[1] isa Float64
@test i0x_32[1] isa Float32

# test against SpecialFunctions.jl
@test i0x_64 ≈ i0x64_SpecialFunctions
@test i0x_32 ≈ i0x32_SpecialFunctions


### Tests for besseli1
i132_SpecialFunctions = SpecialFunctions.besseli.(1, x32)
i164_SpecialFunctions = SpecialFunctions.besseli.(1, x64) 


i1_64 = besseli1.(Float64.(x64))
i1_32 = besseli1.(Float32.(x32))

# make sure output types match input types
@test i1_64[1] isa Float64
@test i1_32[1] isa Float32

# test against SpecialFunctions.jl
@test i1_64 ≈ i164_SpecialFunctions
@test i1_32 ≈ i132_SpecialFunctions

### Tests for besseli1x
i1x32_SpecialFunctions = SpecialFunctions.besselix.(1, x32) 
i1x64_SpecialFunctions = SpecialFunctions.besselix.(1, x64) 

i1x_64 = besseli1x.(Float64.(x64))
i1x_32 = besseli1x.(Float32.(x32))

# make sure output types match input types
@test i1x_64[1] isa Float64
@test i1x_32[1] isa Float32

# test against SpecialFunctions.jl
@test i1x_64 ≈ i1x64_SpecialFunctions
@test i1x_32 ≈ i1x32_SpecialFunctions


# test for besseli
# test small arguments and order
m = 0:1:200; x = 0.1f0:0.5f0:90.0f0
t = [besseli(m, x) for m in m, x in x]
@test t[10] isa Float32
@test t ≈ Float32.([SpecialFunctions.besseli(m, x) for m in m, x in x])

#Float 64
m = 0:1:200; x = 0.1:0.5:150.0
t = [besseli(m, x) for m in m, x in x]

@test t[10] isa Float64
@test t ≈ [SpecialFunctions.besseli(m, x) for m in m, x in x]

@test besselix(10, 2.0) ≈ SpecialFunctions.besselix(10, 2.0)
@test besselix(100, 14.0) ≈ SpecialFunctions.besselix(100, 14.0)
@test besselix(120, 504.0) ≈ SpecialFunctions.besselix(120, 504.0)
