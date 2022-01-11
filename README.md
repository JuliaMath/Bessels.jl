# Bessels

[![Build Status](https://github.com/"heltonmc"/Bessels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/"heltonmc"/Bessels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/"heltonmc"/Bessels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/"heltonmc"/Bessels.jl)


# Benchmarks

Comparing the relative speed (`SpecialFunctions.jl / Bessels.jl`) for a vector of values between 0 and 100.

## Float32

| function | Relative speed |
| ------------- | ------------- |
| besselj0  | 1.7x  |
| besselj1  | 1.7x |
| bessely0  | 1.6x  |
| bessely1  | 1.6x  |
| besseli0  | 24x  |
| besseli1  | 22x  |
| besselk0  | 13x  |
| besselk1  | 15x  |

## Float64

| function | Relative speed |
| ------------- | ------------- |
| besselj0  | 1.7x  |
| besselj1  | 1.9x |
| bessely0  | 1.5x  |
| bessely1  | 1.5x  |
| besseli0  | 9x  |
| besseli1  | 7x  |
| besselk0  | 5x  |
| besselk1  | 5x  |
