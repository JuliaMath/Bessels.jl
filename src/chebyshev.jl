function chbevl(x::Float64, coeff)
    b0 = coeff[1]
    b1 = zero(x)
    b2 = zero(x)

    for i in 2:length(coeff)
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + coeff[i]
    end

    return 0.5 * (b0 - b2)
end
function chbevl(x::Float32, coeff)
    b0 = coeff[1]
    b1 = zero(x)
    b2 = zero(x)

    for i in 2:length(coeff)
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + coeff[i]
    end

    return 0.5f0 * (b0 - b2)
end
