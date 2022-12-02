import numpy as np
import pickle
import pandas as pd
import optuna

class Optimizer:
    """class for handling multi-objective optimization"""
    
    def __init__(self):
        optuna.logging.disable_default_handler()
        self.rgr = pickle.load(open('best_rgr.sav', 'rb'))
        self.rgr_scaler = pickle.load(open('scaler_rgr.pkl', 'rb'))
        self.clf = pickle.load(open('best_clf', 'rb'))
        self.clf_scaler = pickle.load(open('scaler_clf.pkl', 'rb'))
        self.test_case_type = None
        
    def setTestCaseType(self, tc_type):
        self.test_case_type = tc_type
        
    def getSolvabiltyStatus(self, array):
        return self.clf.predict(self.clf_scaler.transform(array))
    
    def getSolverPerformance(self, array):
        return self.rgr.predict(self.rgr_scaler.transform(array))

    def predictions(self, test_case_type, max_grid_points, newton_critical_tolerance, newton_armijo_probes, newton_max_iterations,
                    newton_tolerance, add_factor, remove_factor, use_collocation_scaling):
        array = np.array([test_case_type, max_grid_points, newton_critical_tolerance,
                             newton_armijo_probes, newton_max_iterations, newton_tolerance,
                             add_factor, remove_factor, use_collocation_scaling])
        array = array.reshape(1, -1)

        if self.getSolvabiltyStatus(array) == 0:
            return np.array([[float('inf'), float('inf'), float('inf')]])
        else:
            return self.getSolverPerformance(array)

    def objective(self, trial):
        params = {'test_case_type': self.test_case_type,

                  'max_grid_points': trial.suggest_int('max_grid_points', 100, 10000),
                  'newton_critical_tolerance': trial.suggest_float('newton_critical_tolerance',
                                                                     np.log(1.0027016845009548e-12),
                                                                     np.log(9.93227151144808e+307)),
                  'newton_armijo_probes': trial.suggest_int('newton_armijo_probes', 1, 10),
                  'newton_max_iterations': trial.suggest_int('newton_max_iterations', 1, 100),
                  'newton_tolerance': trial.suggest_float('newton_tolerance',
                                                            np.log(1.0000700155834424e-12),
                                                            np.log(0.0099994154229888)),
                  'add_factor': trial.suggest_float('add_factor', 1, 1000),
                  'remove_factor': trial.suggest_float('remove_factor', 1.5422041344544035e-06,
                                                         1.9999960347604355),
                  'use_collocation_scaling': trial.suggest_categorical('use_collocation_scaling', [0, 1])}
        
        prediction = self.predictions(**params)

        return prediction[0][0], prediction[0][1], prediction[0][2]

    def optimizenODEevals(self, trial):
        x, y, z = self.objective(trial)
        return x

    def optimizenGridPoints(self, trial):
        x, y, z = self.objective(trial)
        return y

    def optimizemaxResiduum(self, trial):
        x, y, z = self.objective(trial)
        return z

    def optimize_ODEevals_gridPoints(self, trial):
        x, y, z = self.objective(trial)
        return x, y
    
    def optimize_gridPoints_maxResiduum(self, trial):
        x, y, z = self.objective(trial)
        return y, z

    def optimize_maxResiduum_ODEevals(self, trial):
        x, y, z = self.objective(trial)
        return z, x

    def singleCriterion(self, objective):
        study = optuna.create_study(directions=['minimize'])
        if objective == 'nODEevals':
            study.optimize(self.optimizenODEevals, n_trials=50)
        elif objective == 'nGridPoints':
            study.optimize(self.optimizenGridPoints, n_trials=50)
        else:
            study.optimize(self.optimizemaxResiduum,     n_trials=50)

        return study.best_trial

    def twoCriteria(self, objective):
        study = optuna.create_study(directions=['minimize', 'minimize'])
        if objective == 'EvalGp':
            study.optimize(self.optimize_ODEevals_gridPoints, n_trials=50)
        elif objective == 'GpRes':
            study.optimize(self.optimize_gridPoints_maxResiduum, n_trials=50)
        else:
            study.optimize(self.optimize_maxResiduum_ODEevals, n_trials=50)

        return study.best_trials


    def threeCriteria(self):
        study = optuna.create_study(directions=['minimize', 'minimize', 'minimize'])
        study.optimize(self.objective, n_trials=50)
        return study.best_trials