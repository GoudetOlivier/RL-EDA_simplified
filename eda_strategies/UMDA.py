
import numpy as np
import torch.nn.functional as F
import torch

from eda_strategies.Abstract_EDA import Abstract_EDA


import numpy as np
import torch


class UMDA(Abstract_EDA):

    def __init__(self, N, lambda_, mu, device):

        Abstract_EDA.__init__(self, N,  lambda_, device)

        self.first_threshold = torch.nn.Threshold(1/self.N, 1/self.N)
        self.second_threshold = torch.nn.Threshold( - 1 + 1/self.N, - 1 + 1/self.N)
        
        self.mu = mu

    def reset_learned_parameters(self, nb_runs):

        self.proba = torch.ones((nb_runs, 1, self.N, 1)).to(self.device)* 0.5


    def sample_solutions(self):


        solution = torch.bernoulli(self.proba.repeat([1, self.lambda_, 1, 1]))




        return solution


    def toString(self):

        return "Strategy_UMDA"


    def updateDistribution(self, solutionList, scoreList):


        sorted, indices = torch.sort(scoreList, dim=1)

        sorted_solutionList = (solutionList.squeeze(3)).gather(1, indices.unsqueeze(2).repeat([1,1,self.N]))


        #Update the proba
        self.proba = torch.mean(sorted_solutionList[:, (self.lambda_ - self.mu ):, :], dim=1).unsqueeze(1).unsqueeze(3)

        ## Apply lower and upper bound in order to keep each proba in the range [1/N, 1 - 1/N]
        self.proba = self.first_threshold(self.proba)
        self.proba = - self.second_threshold(-self.proba)




    def get_lower_bound_hyperparameters(self):
        return np.array([1,1])

    def get_upper_bound_hyperparameters(self):
        return np.array([self.N,self.N])

    def get_init_hyperparameters(self):

        params = np.array([self.N,self.N])

        return params

    def update_hyperparameters(self, params):

        self.lambda_ = int(params[0])
        self.mu = int(params[0])

        if(self.mu > self.lambda_):
            self.mu = self.lambda_


