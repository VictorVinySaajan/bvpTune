import numpy as np
import pickle
import pandas as pd
import optuna
from visualizer import Visualizer
from optimizer import Optimizer


class bvpFineTuner:
    """class that handles the fine-tuning of congiguration parameters of bvp4c solver from MATLAB"""

    def __init__(self):
        self.optimizer = Optimizer()
        self.visualizer = Visualizer()

    def setTrailNumber(self, trials):
        self.optimizer.setTrailNumber(trials)

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
            return print('\n', 'nODEevals: ', int(stats[0][0]), '\n', 'nGridPoints: ', int(stats[0][1]), '\n',
                         'maxResiduum: ',
                         stats[0][2], '\n')
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
        row = pd.DataFrame(
            columns=['test_case_type', 'max_grid_points', 'newton_critical_tolerance', 'newton_armijo_probes',
                     'newton_max_iterations', 'newton_tolerance', 'add_factor', 'remove_factor',
                     'use_collocation_scaling', 'ODE_Evaluations', 'Grid_Points'])
        for i in range(0, len(best_trials)):
            best_trials[i].params['newton_critical_tolerance'] = np.exp(
                best_trials[i].params['newton_critical_tolerance'])
            best_trials[i].params['newton_tolerance'] = np.exp(best_trials[i].params['newton_tolerance'])
            row.loc[i] = [test_case_type, best_trials[i].params['max_grid_points'],
                          best_trials[i].params['newton_critical_tolerance'],
                          best_trials[i].params['newton_armijo_probes'],
                          best_trials[i].params['newton_max_iterations'], best_trials[i].params['newton_tolerance'],
                          best_trials[i].params['add_factor'], best_trials[i].params['remove_factor'],
                          best_trials[i].params['use_collocation_scaling'], best_trials[i].values[0],
                          best_trials[i].values[1]]
        print("Completed")
        return row

    def getOptimalGridPointsAndResiduum(self, test_case_type):
        print("Calculating the pareto front settings")
        self.optimizer.setTestCaseType(test_case_type)
        best_trials = self.optimizer.twoCriteria('GpRes')
        row = pd.DataFrame(
            columns=['test_case_type', 'max_grid_points', 'newton_critical_tolerance', 'newton_armijo_probes',
                     'newton_max_iterations', 'newton_tolerance', 'add_factor', 'remove_factor',
                     'use_collocation_scaling', 'Grid_Points', 'Maximum_Residuum'])
        for i in range(0, len(best_trials)):
            best_trials[i].params['newton_critical_tolerance'] = np.exp(
                best_trials[i].params['newton_critical_tolerance'])
            best_trials[i].params['newton_tolerance'] = np.exp(best_trials[i].params['newton_tolerance'])
            row.loc[i] = [test_case_type, best_trials[i].params['max_grid_points'],
                          best_trials[i].params['newton_critical_tolerance'],
                          best_trials[i].params['newton_armijo_probes'],
                          best_trials[i].params['newton_max_iterations'], best_trials[i].params['newton_tolerance'],
                          best_trials[i].params['add_factor'], best_trials[i].params['remove_factor'],
                          best_trials[i].params['use_collocation_scaling'], best_trials[i].values[0],
                          best_trials[i].values[1]]
        print("Completed")
        return row

    def getOptimalResiduumAndODEevals(self, test_case_type):
        print("Calculating the pareto front settings")
        self.optimizer.setTestCaseType(test_case_type)
        best_trials = self.optimizer.twoCriteria('ResEval')
        row = pd.DataFrame(
            columns=['test_case_type', 'max_grid_points', 'newton_critical_tolerance', 'newton_armijo_probes',
                     'newton_max_iterations', 'newton_tolerance', 'add_factor', 'remove_factor',
                     'use_collocation_scaling', 'Maximum_Residuum', 'ODE_Evaluations'])
        for i in range(0, len(best_trials)):
            best_trials[i].params['newton_critical_tolerance'] = np.exp(
                best_trials[i].params['newton_critical_tolerance'])
            best_trials[i].params['newton_tolerance'] = np.exp(best_trials[i].params['newton_tolerance'])
            row.loc[i] = [test_case_type, best_trials[i].params['max_grid_points'],
                          best_trials[i].params['newton_critical_tolerance'],
                          best_trials[i].params['newton_armijo_probes'],
                          best_trials[i].params['newton_max_iterations'], best_trials[i].params['newton_tolerance'],
                          best_trials[i].params['add_factor'], best_trials[i].params['remove_factor'],
                          best_trials[i].params['use_collocation_scaling'], best_trials[i].values[0],
                          best_trials[i].values[1]]
        print("Completed")
        return row

    def getOptimizedSettings(self, test_case_type):
        print("Calculating the pareto front settings")
        self.optimizer.setTestCaseType(test_case_type)
        best_trials = self.optimizer.threeCriteria()
        row = pd.DataFrame(
            columns=['test_case_type', 'max_grid_points', 'newton_critical_tolerance', 'newton_armijo_probes',
                     'newton_max_iterations', 'newton_tolerance', 'add_factor', 'remove_factor',
                     'use_collocation_scaling', 'ODE_Evaluations', 'Grid_Points', 'Maximum_Residuum'])
        for i in range(0, len(best_trials)):
            best_trials[i].params['newton_critical_tolerance'] = np.exp(
                best_trials[i].params['newton_critical_tolerance'])
            best_trials[i].params['newton_tolerance'] = np.exp(best_trials[i].params['newton_tolerance'])

            row.loc[i] = [test_case_type, best_trials[i].params['max_grid_points'],
                          best_trials[i].params['newton_critical_tolerance'],
                          best_trials[i].params['newton_armijo_probes'],
                          best_trials[i].params['newton_max_iterations'], best_trials[i].params['newton_tolerance'],
                          best_trials[i].params['add_factor'], best_trials[i].params['remove_factor'],
                          best_trials[i].params['use_collocation_scaling'], best_trials[i].values[0],
                          best_trials[i].values[1],
                          best_trials[i].values[2]]
        print("Completed")
        return row

    def visualize(self, dataframe):
        if len(dataframe.columns) == 11:
            self.visualizer.visualize2d(dataframe)
        elif len(dataframe.columns) == 12:
            self.visualizer.visualize3d(dataframe)
        else:
            print('Dataframe is not in right format. Please provided the dataframe returned by bvpFineTuner functions')

def main():
    tuner = bvpFineTuner()
    tuner.setTrailNumber(500)

    optimizedSettings = tuner.getOptimalResiduumAndODEevals(test_case_type=8)
    tuner.visualize(optimizedSettings)

if __name__=='__main__':
    main()