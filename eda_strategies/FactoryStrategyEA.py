
import torch.nn as nn
import torch
from eda_strategies.UMDA import UMDA
from eda_strategies.PBIL import PBIL
from eda_strategies.RL_EDA import RL_EDA



class FactoryStrategyEA:


    def createStrategyEA(self, typeStrategy, N, lambda_, mu, rho, beta, device,  numberHiddenLayersG, nh, nb_train, epsilon, learning_rate, dim_variables, activation):

        if (typeStrategy == "UMDA"):
            return UMDA(N, lambda_, mu, device)

        elif(typeStrategy == "PBIL"):

            return PBIL(N, lambda_,mu, rho, device)


        elif (typeStrategy == "RL-EDA"):
            
            print("OK")

            return RL_EDA(N,  lambda_, beta, device, numberHiddenLayersG, nh, nb_train,  epsilon, learning_rate, dim_variables, activation)
