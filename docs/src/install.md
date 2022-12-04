### Installation requirements

Bessels requires Julia v1.6 or later but it is recommended to use the latest stable release. Bessels uses CI tools provided by GitHub Actions to run tests on Julia versions 1.6, current stable release, and nightly builds on Linux operating systems using `ubuntu-latest`. Though, Bessels is written entirely in Julia and therefore should work on any of the [supported platforms](https://julialang.org/downloads/#supported_platforms) such as MacOS and Windows.

### Installation

Install Julia by [downloading](https://julialang.org/downloads/) the latest version from the offical site and follow the [platform specific installations](https://julialang.org/downloads/platform/). 

You can add Bessels using Julia's package manager by typing `] add Bessels` in the Julia prompt.

```julia
julia> ] # ']' should be pressed

(@v1.8) pkg> add Bessels
```

The package manager can also be used verify the installation by running the bundled tests, check the installed version, or update to the latest release.

```julia
julia> ]

# run tests
(@v1.8) pkg> test Bessels

# check version
(@v1.8) pkg> status Bessels
Status `~/.julia/environments/v1.8/Project.toml`
  [0e736298] Bessels v0.2.7

# update to latest release
(@v1.8) pkg> update Bessels
```

### Running

At this point you are ready to start using Bessels.jl!

```julia
julia> using Bessels

julia> besselj0(1.2)
0.6711327442643626

julia> besselj(1.8, 10.1)
0.2374319718222891
```