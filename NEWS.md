# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For pre-1.0.0 releases we will increment the minor version when we export any new functions or alter exported API.
For bug fixes, performance enhancements, or fixes to unexported functions we will increment the patch version.
**Note**: The exported API can be considered very stable and likely will not change without serious consideration.

# Unreleased

### Added
 - add an unexport method (`Bessels.besseljy(nu, x)`) for faster computation of `besselj` and `bessely` (#33)
 - add exported methods for Hankel functions `besselh(nu, k, x)`, `hankelh1(nu, x)`, `hankelh2(nu, x)` (#33)
 - add exported methods for spherical bessel function `sphericalbesselj(nu, x)`, `sphericalbesselj(nu, x)`, (#38)

### Fixed
 - fix cutoff in `bessely` to not return error for integer orders and small arguments (#33)

# Version 0.1.0

Initial release of Bessels.jl

### Added

 - support bessel functions (`besselj`, `bessely`, `besseli`, `besselk`) for real arguments
 - provide a gamma function (`Bessels.gamma`) for real arguments