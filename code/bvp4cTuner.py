import numpy as np
import pickle
import pandas as pd
import optuna

class bvpFineTuner:
    """class that handles the fine-tuning of congiguration parameters of bvp4c solver from MATLAB"""
    
    def __init__(self):
        self.optimizer = Optimizer()
        
    def getSolvabiltyStatus(self, test_case_type, max_grid_points, newton_critical_tolerance,
                             newton_armijo_probes, newton_max_iterations, newton_tolerance,
                             add_factor, remove_factor, use_collocation_scaling):
        array = np.array([test_case_type, max_grid_points, np.log(newton_critical_tolerance),
                             newton_armijo_probes, newton_max_iterations, np.log(newton_tolerance),
                             add_factor, remove_factor, use_collocation_scaling])
        array = array.reshape(1, -1)
        
        if bool(self.optimizer.getSolvabiltyStatus(array)):
            return 'The problem is solvable'
        else:
            return 'The problem is not solvable' 
    
    def getSolverPerformance(self, test_case_type, max_grid_points, newton_critical_tolerance,
                             newton_armijo_probes, newton_max_iterations, newton_tolerance, 
                             add_factor, remove_factor, use_collocation_scaling):
        array = np.array([test_case_type, max_grid_points, np.log(newton_critical_tolerance),
                          newton_armijo_probes, newton_max_iterations, np.log(newton_tolerance),
                          add_factor, remove_factor, use_collocation_scaling])
        array = array.reshape(1, -1)
        
        if bool(self.optimizer.getSolvabiltyStatus(array)):
            stats = self.optimizer.getSolverPerformance(array)
            return print('\n','nODEevals: ', int(stats[0][0]), '\n','nGridPoints: ', int(stats[0][1]), '\n','maxResiduum: ', stats[0][2], '\n')
        else:
            return 'The problem is not solvable'
            
    def getOptimalODEevals(self, test_case_type):
        print("Calculating the best setting")
        self.optimizer.setTestCaseType(test_case_type)
        best_trial = self.optimizer.singleCriterion('nODEevals')
        best_trial.params['newton_critical_tolerance'] = np.exp(best_trial.params['newton_critical_tolerance'])
        best_trial.params['newton_tolerance'] = np.exp(best_trial.params['newton_tolerance'])
        print('Numerical Settings: ', best_trial.params)
        print('ODE Evaluations: ', int(best_trial.values[0]))

    def getOptimalGridPoints(self, test_case_type):
        print("Calculating the best setting")
        self.optimizer.setTestCaseType(test_case_type)
        best_trial = self.optimizer.singleCriterion('nGridPoints')
        best_trial.params['newton_critical_tolerance'] = np.exp(best_trial.params['newton_critical_tolerance'])
        best_trial.params['newton_tolerance'] = np.exp(best_trial.params['newton_tolerance'])
        print('Numerical Settings: ', best_trial.params)
        print('Grid Points: ', int(best_trial.values[0]))

    def getOptimalResiduum(self, test_case_type):
        print("Calculating the best setting")
        self.optimizer.setTestCaseType(test_case_type)
        best_trial = self.optimizer.singleCriterion('maxResiduum')
        best_trial.params['newton_critical_tolerance'] = np.exp(best_trial.params['newton_critical_tolerance'])
        best_trial.params['newton_tolerance'] = np.exp(best_trial.params['newton_tolerance'])
        print('Numerical Settings: ', best_trial.params)
        print('Maximum Residdum: ', best_trial.values)
        
    def getOptimalODEevalsAndGridPoints(self, test_case_type):
        print("Calculating the pareto front settings")
        self.optimizer.setTestCaseType(test_case_type)
        best_trials = self.optimizer.twoCriteria('EvalGp')
        for i in range(0, len(best_trials)):
            print('\n')
            best_trials[i].params['newton_critical_tolerance'] = np.exp(best_trials[i].params['newton_critical_tolerance'])
            best_trials[i].params['newton_tolerance'] = np.exp(best_trials[i].params['newton_tolerance'])
            print('Numerical Settings: ', best_trials[i].params)
            print('ODE Evaluations: ', int(best_trials[i].values[1]), 'Grid Points: ', int(best_trials[i].values[0]))
            print('\n')

    def getOptimalGridPointsAndResiduum(self, test_case_type):
        print("Calculating the pareto front settings")
        self.optimizer.setTestCaseType(test_case_type)
        best_trials = self.optimizer.twoCriteria('GpRes')
        for i in range(0, len(best_trials)):
            print('\n')
            best_trials[i].params['newton_critical_tolerance'] = np.exp(best_trials[i].params['newton_critical_tolerance'])
            best_trials[i].params['newton_tolerance'] = np.exp(best_trials[i].params['newton_tolerance'])
            print('Numerical Settings: ', best_trials[i].params)
            print('Grid Points: ', int(best_trials[i].values[0]), 'Maximum Residuum: ', best_trials[i].values[1])
            print('\n')

    def getOptimalResiduumAndODEevals(self, test_case_type):
        print("Calculating the pareto front settings")
        self.optimizer.setTestCaseType(test_case_type)
        best_trials = self.optimizer.twoCriteria('ResEval')
        for i in range(0, len(best_trials)):
            print('\n')
            best_trials[i].params['newton_critical_tolerance'] = np.exp(best_trials[i].params['newton_critical_tolerance'])
            best_trials[i].params['newton_tolerance'] = np.exp(best_trials[i].params['newton_tolerance'])
            print('Numerical Settings: ', best_trials[i].params)
            print('Maximum Residuum: ', best_trials[i].values[0], 'ODE Evaluations: ', int(best_trials[i].values[1]))
            print('\n')

    def getOptimizedSettings(self, test_case_type):
        print("Calculating the pareto front settings")
        self.optimizer.setTestCaseType(test_case_type)
        best_trials = self.optimizer.threeCriteria()
        for i in range(0, len(best_trials)):
            print('\n')
            best_trials[i].params['newton_critical_tolerance'] = np.exp(best_trials[i].params['newton_critical_tolerance'])
            best_trials[i].params['newton_tolerance'] = np.exp(best_trials[i].params['newton_tolerance'])
            print('Numerical Settings: ', best_trials[i].params)
            print('ODE Evaluations: ', int(best_trials[i].values[0]), 'Grid Points: ', int(best_trials[i].values[1]),
                    'Maximum Residuum: ', best_trials[i].values[2])
            print('\n')