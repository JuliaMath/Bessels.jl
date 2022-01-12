function chbevl(x::Union{Float32, Float64}, coeff)
    b0 = coeff[1]
    b1 = zero(x)
    b2 = zero(x)

    for i in 2:length(coeff)
        b2 = b1
        b1 = b0
        b0 = muladd(x, b1, -b2) + coeff[i]
    end

    return (b0 - b2)/2
end
