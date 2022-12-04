## Contributors Guide

Contributions of any kind are very welcome! Bessels.jl is a community-driven project and is under the JuliaMath organization.
If you are interested in getting involved with the development of Bessels.jl please feel free to open an issue. In general, contributions will likely fall under four main categories:

**1. Implementing a new function**

Any Bessel type or related function is welcome for inclusion in this library. Please read the scope section for more details but in general any function listed [here](https://mpmath.org/doc/current/functions/bessel.html) would be a good addition. Opening an issue would be a great start to further discuss a particular function or implementation details. If you already have an existing function ready to go please open a pull request. Porting functions from MIT compatible implementation is also acceptable.

In general, Bessels.jl aims to provide highly accurate implementations. We do not have a set error tolerance we are willing to accept as it will depend on the function. For example, the `gamma` function is a very important and widely used function so it is necessary to provide maximal errors less than 2 Units in the last place (ULP). For other functions, such as the Bessel function, which oscillate around zero it is much more difficult to provide such tight error tolerances. Therefore, relative tolerances better than `1e-14` would be a good target with a slightly less tolerance around the zeros. Single variable functions should be able to provide better tolerances than multi-variable functions.

Even if the function isn't quite to those error tolerances, please open a pull request to discuss further. It might be good for a single precision implementation or there might be opportunities to improve the errors of the existing implementation. Though, there are fairly strict criteria that the function should be non-allocating and type stable.

**2. Improving exisiting function**

Improving the accuracy or speed of any implementation would be a great contribution. There are plenty of opportunities so please open an issue if you are interested and we can point you to a good function to work with. Accuracy improvements are always welcome whereas any speed improvements would also need to maintain the current level of accuracy. Ideally, implementations could trend to more accurate and faster but there will always be some tradeoff.

**3. Filing bug reports, writing tests, feature requests, improve documentation**

Testing the accuracy and performance of each function is also welcome. In the future, we hope to have an automated way to print out the error and performance of each function so any help with this would be appreciated. Additionally, contributing better tests and writing documentation is very helpful. Filing feature requests are also very helpful.

**4. Other miscellaneous contributions**

Please also share any papers or discussions on implementation details!

#### Pull Requests

Contributions should be made via pull requests on GitHub. This can be done by forking Bessels.jl and making commits on your own fork. After you have made all necessary changes please submit a pull request against the main branch. We will then review and provide any feedback on the changes. Once you a submit a PR, the automated tests will run through CI. Therefore, please also include tests if you are implementing a new function and review the results of the CI run. We typically keep all tests passing on the main branch, so any CI failures are most likely related to the PR. By submitting a pull request, you are agreeing for contributed code to be under the MIT license.

**Code Style:** There is not an explicit code style that we use. Please try to keep it as consistent as possible with existing code.