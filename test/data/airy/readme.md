```julia
using ArbNumerics, DelimitedFiles

let
    # set to higher default precision
    setprecision(ArbFloat, 2500)

    # generate xdata
    x1 = map(_ -> rand().* 10.0 .^ (rand(-300:-5)), 1:100)
    x2 = map(_ -> rand().* 10.0 .^ (rand(-5:6)), 1:1000)
    x3 = map(_ -> rand().* 10.0 .^ (rand(6:300)), 1:100)
    x = sort([x1; x2; x3])
    x_arb = ArbFloat.(x)

    # generate airy data

    # positive arguments for scaled arguments
    aix = @. Float64(airyai(x_arb) * exp(2 * x_arb * sqrt(x_arb) / 3))
    aiprimex = @. Float64(airyaiprime(x_arb) * exp(2 * x_arb * sqrt(x_arb) / 3))
    bix = @. Float64(airybi(x_arb) * exp(-2 * x_arb * sqrt(x_arb) / 3))
    biprimex = @. Float64(airybiprime(x_arb) * exp(-2 * x_arb * sqrt(x_arb) / 3))

    writedlm("airy_positive_args.csv", [x aix aiprimex bix biprimex])

    # negative arguments for unscaled arguments
    ai = @. Float64(airyai(-x_arb))
    aiprime = @. Float64(airyaiprime(-x_arb))
    bi = @. Float64(airybi(-x_arb))
    biprime = @. Float64(airybiprime(-x_arb))

    writedlm("airy_negative_args.csv", [-x ai aiprime bi biprime])
end
```