# bvpTune
Machine learning based library for tuning the numerical settings of boundary value problem (BVP) solver.

This library can be used to find the optimal numerical settings of the BVP solver.
This solver is a self-implemented C++ version of the bvp4c solver from MATLAB[[1]](#1) that uses a Newton-Armijo method instead of a
simplified Newton method for solving the system of non-linear equations.

The library uses machine learning regression model that maps the numerical settings to
solver performance criteria for faster predictions. This model is then used to optimize the settings 
using the optuna library[[2]](#2).

The functions provide by the library are listed as folows:

 ### 1)  getOptimalGridPoints(test_case_type):
      returns the optimal number of grid points for the specified test_case_type of the boundary value problem.
      
 ### 2)  getOptimalODEevals(test_case_type)
      returns the optimal number of ode evaluations for the specified test_case_type of the boundary value problem.
      
 ### 3)  getOptimalResiduum(test_case_type)
      returns the optimal residuum for the specified test_case_type of the boundary value problem.
 
 ### 4)  getOptimalODEevalsAndGridPoints(test_case_type)
      returns parato front numerical settings as a dataframe that optimizes the ode evaluations and the grid points 
      for the specified test_case_type of the boundary value problem.
 
 ### 5)  getOptimalGridPointsAndResiduum(test_case_type)
      returns parato front numerical settings as a dataframe that optimizes the grid points and the residuum
      for the specified test_case_type of the boundary value problem.
   
 ### 6)  getOptimalResiduumAndODEevals(test_case_type)
      returns parato front numerical settings as a dataframe that optimizes the residuun and the ode evaluations 
      for the specified test_case_type of the boundary value problem.
   
 ### 7)  getOptimizedSettings(test_case_type)
      returns parato front numerical settings as a dataframe that optimizes the ode evaluations, grid points, and the residuum
      for the specified test_case_type of the boundary value problem
   
 ### 8)  getSolvabiltyStatus(test_case_type, max_grid_points, newton_critical_tolerance, newton_armijo_probes, newton_max_iterations, newton_tolerance,                          add_factor, remove_factor, use_collocation_scaling)
       returns whether the specified numerical setting is solvable or not for the specified test_case_type of 
       the boundary value problem.
   
 ### 9)  getSolverPerformance(test_case_type, max_grid_points, newton_critical_tolerance, newton_armijo_probes, newton_max_iterations, newton_tolerance,                        add_factor,remove_factor, use_collocation_scaling)
       returns the solver statistics of specified numerical setting for the specified test_case_type of the boundary value problem.
       
### 10)  visualize(dataframe)
        visualize the 2d/3d pareto front plots for the provided dataframe returned by the optimization functions.
        
        
**Reference Boundary Value Problems**

| test_case_type | Refrence Problem Number | Problem Type
| ----------- | ----------- |----------- |
|1|1|Linear|
|2|3|Linear|
|3|4|Linear|
|4|7|Linear|
|5|19|Non-linear|
|6|20|Non-linear|
|7|22|Non-linear|
|8|23|Non-linear|
|9|24|Non-linear|
|10|33|Non-linear|

The reference problems are selected from [[3]](#3).

## References
<a id="1">[1]</a>
Kierzenka, Jacek and Shampine, Lawrence F, ``A BVP solver based on residual control and the Maltab PSE``, ACM Transactions on Mathematical Software (TOMS)., ACM New York, NY, USA, vol. 27, no. 3, pp. 299--316, 2001.

<a id="2">[2]</a>
Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori, ``Optuna: A Next-generation Hyperparameter Optimization Framework``, Proceedings of the 25rd {ACM} {SIGKDD} International Conference on Knowledge Discovery and Data Mining., pp. 2623--2631, 2019.

<a id="3">[3]</a> 
Soetaert, Karline and Cash, Jeff and Mazzia, Francesca. (2010). 
``Package bvpSolve, solving testproblems``.


