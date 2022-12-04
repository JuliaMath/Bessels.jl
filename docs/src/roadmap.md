## Bessels.jl Roadmap

The major development goals can be summarized by the following groups.

### Support for complex arguments

The current development and immediate goals of this project is to provide support for complex variables in all the existing functions. Currently, the Airy functions `airyai`, `airyaiprime`, `airybi`, and `airybiprime` are the only supported functions using complex variables. This will be implemented in a step approach focusing on single argument functions with fixed order (e.g. $\nu=0, 1$) then expanded to the general case of arbitrary order. Therefore, current focus is on developing complex routines for `besselj0(z)`, `besselj1(z)`, `bessely0(z)`, `bessely1(z)`, `besselk0(z)`, `besselk1(z)`, `besseli0(z)`, and `besseli1(z)`.

### Adding additional special functions

A nice list of Bessel and related functions is provided by [mpmath](https://mpmath.org/doc/current/functions/bessel.html). Any function listed there would be a good candidate for inclusion in this package. Functions on the current scope are the Kelvin, Struve, and Scorer type functions. If other functions (or the ones listed) are desired, please open an issue!

### Develop higher precision routines

Current development has focused on single and double precison routine (`Float32` and `Float64`). We intend to also provide support for higher precision types such as double-double (`Double64`) and/or quadruple  (`Float128`) precision in the future. 

### Derivatives through automatic differentiation

Typically, the derivatives of certain special functions with respect to argument can be obtained through definitions. However, derivative with respect to the order usually does not have a simple expression. Automatic differentiation could be a powerful tool to quickly compute higher order derivatives using native Julia code. This is not an immediate goal but would be a welcome contribution.

### Accuracy and Peformance improvements

Speed improvements are beneficial in most cases as these low level functions are typically called many times. Performance gains can come from using better algorithms or through better support for parallelization (e.g., vectorization, SIMD). In general, any speed improvements will only be incoporated that maintain the current relative accuracy. Accuracy improvements may come at the cost of a slightly longer runtime. These type of improvements will come on a more ad-hoc basis.