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
m = 0:1:200; x = 0.5f0:0.5f0:90.0f0
@test besseli(10, 1.0f0) isa Float32
@test besseli(2, 80.0f0) isa Float32
@test besseli(112, 80.0f0) isa Float32

for m in m, x in x
    @test besseli(m, x) ≈ Float32(SpecialFunctions.besseli(m, x))
end

#Float 64
m = 0:1:200; x = 0.1:0.5:150.0
@test besseli(10, 1.0) isa Float64
@test besseli(Int16(10), Float16(1.0)) isa Float16

@test besseli(2, 80.0) isa Float64
@test besseli(112, 80.0) isa Float64
t = [besseli(m, x) for m in m, x in x]

@test t[10] isa Float64
@test t ≈ [SpecialFunctions.besseli(m, x) for m in m, x in x]

t = [besselix(m, x) for m in m, x in x]
@test t[10] isa Float64
@test t ≈ [SpecialFunctions.besselix(m, x) for m in m, x in x]
@test besselix(Int16(10), Float16(1.0)) isa Float16

## Tests for besseli

## test all numbers and orders for 0<nu<100
x = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0, 1.001, 1.01, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 1.9, 2.5, 3.0, 3.5, 4.0]
nu = [0.01,0.1, 0.5, 0.8, 1, 1.23, 2,2.56, 4,5.23, 6,9.2, 10,12.89, 15, 19.1, 20, 25, 30, 33.123, 40, 45, 50, 51.5, 55, 60, 65, 70, 72.34, 75, 80, 82.1, 85, 88.76, 90, 92.334, 95, 99.87,100, 110, 125, 145.123, 150, 160.789]
for v in nu, xx in x
    xx *= v
    sf = SpecialFunctions.besseli(v, xx)
    @test isapprox(besseli(v, xx), Float64(sf), rtol=2e-13)
    @test isapprox(besseli(Float32(v), Float32(xx)), Float32(sf))
end

# test nu_range
@test besseli(0:250, 2.0) ≈ SpecialFunctions.besseli.(0:250, 2.0) rtol=1e-13
@test besseli(0.5:1:10.5, 2.0) ≈ SpecialFunctions.besseli.(0.5:1:10.5, 2.0) rtol=1e-13
@test Bessels.besseli!(zeros(Float64, 10), 1:10, 1.0) ≈ besseli(1:10, 1.0)

### need to fix method ambiguities for other functions ###### 

# test Inf
@test isinf(besseli(2, Inf))

 ### tests for negative arguments

(v, x) = 12.0, 3.2
@test besseli(v,x) ≈ 7.1455266650203694069897133431e-7

(v,x) = 13.0, -1.0
@test besseli(v,x) ≈ -1.995631678207200756444e-14

(v,x) = 12.6, -3.0
#@test besseli(v,x) ≈ -2.725684975265100582482e-8 + 8.388795775899337839603e-8 * im

(v, x) = -8.0, 4.2
@test besseli(v,x) ≈ 0.0151395115677545706602449919627

(v, x) = 12.3, 8.2
@test besseli(v,x) ≈ 0.113040018422133018059759059298

(v, x) = -12.3, 8.2
@test besseli(v,x) ≈ 0.267079696793126091886043602895

(v, x) = -14.0, -9.9
@test besseli(v,x) ≈ 0.2892290867115615816280234648

(v, x) = -14.6, -10.6
#@test besseli(v,x) ≈ -0.157582642056898598750175404443 - 0.484989503203097528858271270828*im
