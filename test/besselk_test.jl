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

# test half-integer orders:
for (v,x) in Iterators.product(-35/2:1.0:81/2, range(0.0, 30.0, length=51)[2:end])
  @test besselk(v, x) ≈ SpecialFunctions.besselk(v, x)
  @test besselk(Float32(v), Float32(x)) ≈ SpecialFunctions.besselk(Float32(v), Float32(x))
end

# test small arguments and order
m = 0:40; x = [1e-6; 1e-4; 1e-3; 1e-2; 0.1; 1.0:2.0:500.0]
for m in m, x in x
    @test besselk(m, x) ≈ SpecialFunctions.besselk(m, x)
end

# test medium arguments and order
m = 30:200; x = 5.0:5.0:100.0
t = Float64.([besselk(m, x) for m in m, x in x])
@test t ≈ [SpecialFunctions.besselk(m, x) for m in m, x in x]

# test large orders
m = 200:5:1000; x = 400.0:10.0:1200.0
t = Float64.([besselk(m, x) for m in m, x in x])
@test t ≈ [SpecialFunctions.besselk(m, x) for m in m, x in x]

# Float 32 tests for aysmptotic expansion
m = 20:5:200; x = 5.0f0:2.0f0:400.0f0
t = [besselk(m, x) for m in m, x in x]
@test t[10] isa Float32
@test t ≈ Float32.([SpecialFunctions.besselk(m, x) for m in m, x in x])

# test for low values and medium orders
m = 20:5:50; x = [1f-3, 1f-2, 1f-1, 1f0, 1.5f0, 2.0f0, 4.0f0]
t = [besselk(m, x) for m in m, x in x]
@test t[5] isa Float32
@test t ≈ Float32.([SpecialFunctions.besselk(m, x) for m in m, x in x])

@test iszero(besselk(20, 1000.0))
#@test isinf(besselk(250, 5.0))

### Tests for besselkx
@test besselkx(0, 12.0) ≈ besselk0x(12.0)
@test besselkx(1, 89.0) ≈ besselk1x(89.0)

@test besselkx(15, 82.123) ≈ SpecialFunctions.besselk(15, 82.123)*exp(82.123)
@test besselkx(105, 182.123) ≈ SpecialFunctions.besselk(105, 182.123)*exp(182.123)
@test besselkx(4, 3.0) ≈ SpecialFunctions.besselk(4, 3.0)*exp(3.0)
@test besselkx(3.2, 1.1) ≈ SpecialFunctions.besselk(3.2, 1.1)*exp(1.1)
@test besselkx(8.2, 9.1) ≈ SpecialFunctions.besselk(8.2, 9.1)*exp(9.1)

## Tests for besselk

## test all numbers and orders for 0<nu<100
x = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0, 1.001, 1.01, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 1.9, 2.5, 3.0, 3.5, 4.0]
nu = [0.01,0.1, 0.5, 0.8, 1, 1.23, 2,2.56, 4,5.23, 6,9.2, 10,12.89, 15, 19.1, 20, 25, 30, 33.123, 40, 45, 50, 51.5, 55, 60, 65, 70, 72.34, 75, 80, 82.1, 85, 88.76, 90, 92.334, 95, 99.87,100, 110, 125, 145.123, 150, 160.789]
for v in nu, xx in x
    xx *= v
    sf = SpecialFunctions.besselk(v, xx)
    @test isapprox(besselk(v, xx), Float64(sf), rtol=2e-13)
    @test isapprox(besselk(Float32(v), Float32(xx)), Float32(sf))
end

# test nu_range
@test besselk(0:50, 2.0) ≈ SpecialFunctions.besselk.(0:50, 2.0) rtol=1e-13
@test besselk(0.5:1:10.5, 12.0) ≈ SpecialFunctions.besselk.(0.5:1:10.5, 12.0) rtol=1e-13
@test besselk(1:700, 800.0) ≈ SpecialFunctions.besselk.(1:700, 800.0) rtol=1e-13

# test Float16
@test besselk(Int16(10), Float16(1.0)) isa Float16
@test besselkx(Int16(10), Float16(1.0)) isa Float16

# test Inf
@test iszero(besselk(2, Inf))

### tests for negative arguments

(v, x) = 12.0, 3.2
@test besselk(v,x) ≈ 56331.504348755621996013084096

(v,x) = 13.0, -1.0
#@test besselk(v,x) ≈ -1921576392792.994084565 - 6.26946181952681217841e-14*im

(v,x) = 12.6, -3.0
#@test besselk(v,x) ≈ -135222.7926354826692727 - 416172.9627453652120223*im

(v, x) = -8.0, 4.2
@test besselk(v,x) ≈ 3.65165949039881135495282699061

(v, x) = 12.3, 8.2
@test besselk(v,x) ≈ 0.299085139926840649406079315812

(v, x) = -12.3, 8.2
@test besselk(v,x) ≈ 0.299085139926840649406079315812

(v, x) = -14.0, -9.9
#@test besselk(v,x) ≈ 0.100786833375605803570325345603 - 0.90863997401752715470886289641*im

(v, x) = -14.6, -10.6
#@test besselk(v,x) ≈ -0.0180385087581148387140033906859 - 1.54653251445680014758965158559*im
