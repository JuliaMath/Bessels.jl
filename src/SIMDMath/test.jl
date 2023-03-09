using SIMDMath
using Test


# Test horner_simd
let
    NT = 12
    P = (
                ntuple(n -> rand()*(-1)^n / n, NT),
                ntuple(n -> rand()*(-1)^n / n, NT),
                ntuple(n -> rand()*(-1)^n / n, NT),
                ntuple(n -> rand()*(-1)^n / n, NT)
            )
    
    x = 0.9
    a = horner_simd(x, pack_horner(P))
    @test evalpoly(x, P[1]) == a[1].value
    @test evalpoly(x, P[2]) == a[2].value
    @test evalpoly(x, P[3]) == a[3].value
    @test evalpoly(x, P[4]) == a[4].value

    NT = 24
    P32 = (
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
                ntuple(n -> Float32(rand()*(-1)^n / n), NT),
            )
    
    x = 1.2f0
    a = horner_simd(x, pack_horner(P32))
    @test evalpoly(x, P32[1]) == a[1].value
    @test evalpoly(x, P32[2]) == a[2].value
    @test evalpoly(x, P32[3]) == a[3].value
    @test evalpoly(x, P32[4]) == a[4].value
    @test evalpoly(x, P32[5]) == a[5].value
    @test evalpoly(x, P32[6]) == a[6].value
    @test evalpoly(x, P32[7]) == a[7].value
    @test evalpoly(x, P32[8]) == a[8].value

    NT = 4
    P16 = (
                ntuple(n -> Float16(rand()*(-1)^n / n), NT),
                ntuple(n -> Float16(rand()*(-1)^n / n), NT),
                ntuple(n -> Float16(rand()*(-1)^n / n), NT),
            )
    
    x = Float16(0.8)
    a = horner_simd(x, pack_horner(P16))
    @test evalpoly(x, P16[1]) ≈ a[1].value
    @test evalpoly(x, P16[2]) ≈ a[2].value
    @test evalpoly(x, P16[3]) ≈ a[3].value

end

# test second, fourth, and eighth order horner schemes
# note that the higher order schemes are more prone to overflow when using lower precision
# because you have to directly compute x^2, x^4, x^8 before the routine
let
    for N in [2, 3, 4, 5, 6, 7, 10, 13, 17, 20, 52, 89], x in [0.1, 0.5, 1.5, 4.2, 45.0]
        poly = ntuple(n -> rand()*(-1)^n / n, N)
        @test evalpoly(x, poly) ≈ horner2(x, pack_horner2(poly)) ≈ horner4(x, pack_horner4(poly)) ≈ horner8(x, pack_horner8(poly))
    end
    for N in [2, 3, 4, 5, 6, 7, 10], x32 in [0.1f0, 0.8f0, 2.4f0, 8.0f0, 25.0f0]
        poly32 = ntuple(n -> Float32(rand()*(-1)^n / n), N)
        @test evalpoly(x32, poly32) ≈ horner2(x32, pack_horner2(poly32)) ≈ horner4(x32, pack_horner4(poly32)) ≈ horner8(x32, pack_horner8(poly32))
    end
    for N in [2, 3, 4, 5, 6], x in [0.1, 0.5, 2.2]
        poly16 = ntuple(n -> Float16(rand()*(-1)^n / n), N)
        x16 = Float16.(x)
        @test evalpoly(x16, poly16) ≈ horner2(x16, pack_horner2(poly16)) ≈ horner4(x16, pack_horner4(poly16)) ≈ horner8(x16, pack_horner8(poly16))
    end

end
