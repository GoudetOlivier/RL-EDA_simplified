import numpy as np
import argparse
import datetime
import torch
import os
import random
import math

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA

from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning')

    # General arguments
    parser.add_argument('type_problem', type=str, help='type_problem : QUBO, NK or NK3')
    parser.add_argument('dim', type=int, help='Instance size')
    parser.add_argument('type_instance', type=int, help='Type instance. Corresponding to K for NK landscape, or to the type of PUBOi distribution for QUBO instances')

    # General options
    parser.add_argument('--type_strategy', type=str, default="RL-EDA", help='type_strategy : RL-EDA, UMDA, PBIL')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--device', type=str, default="cuda:0", help='device')
    parser.add_argument('--nb_instances_test', type=int, default=10, help="Number of different instances for the test")
    parser.add_argument('--nb_restarts', type=int, default=10, help="Nb independent restarts to solve each instance")
    parser.add_argument('--budget', type=int, default=10000, help='number of calls to the objective function for each run')

    # Generator parameters
    parser.add_argument('--lambda_', type=int, default=10, help='lambda : size pop EDA')
    parser.add_argument('--numberHiddenLayersG', type=int, default=-1, help='number of hidden layers in the generator')
    parser.add_argument('--nh', type=int, default=100, help='number of neurons in each hidden layer of the generator')
    parser.add_argument('--activation', type=str, default="relu", help='activation function in generator')
     
    parser.add_argument('--epsilon', type=float, default=0.001, help='probability threshold')




    # RL options
    parser.add_argument('--beta', type=float, default=1, help='beta : KL coefficient')
    parser.add_argument('--nb_train', type=int, default=50, help='nb epoch training at each iteration')
    parser.add_argument('--learning_rate', type=int, default=0.001, help='learning rate optimizer')

    # UMDA and PBIL parameters
    parser.add_argument('--mu', type=int, default=3, help='mu parameter for UMDA and PBIL algorithms')
    parser.add_argument('--rho', type=float, default=0.1, help='rho parameter for PBIL algorithm')


    args = parser.parse_args()

    device = args.device
    type_strategy = args.type_strategy
    dim = args.dim
    type_instance = args.type_instance
    type_problem = args.type_problem
    nb_restarts = args.nb_restarts
    nb_instances_test = args.nb_instances_test
    seed = args.seed
    lambda_ = args.lambda_
    verbose = args.verbose
    budget = args.budget
    beta = args.beta
    nb_train = args.nb_train
    epsilon = args.epsilon
    learning_rate = args.learning_rate
    typeStrategy = args.type_strategy
    activation = args.activation
    
    numberHiddenLayersG = args.numberHiddenLayersG
    nh = args.nh

 


    mu = args.mu
    rho = args.rho

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    N = dim

    if not os.path.exists("results/" + typeStrategy):
        os.mkdir("results/" + typeStrategy)

    if not os.path.exists("results/" + typeStrategy + "/" + type_problem):
        os.mkdir("results/" + typeStrategy + "/" + str(type_problem))

    if not os.path.exists("results/" + typeStrategy + "/" + type_problem + "/" + str(dim)):
        os.mkdir("results/" + typeStrategy + "/" + type_problem + "/" + str(dim))

    if not os.path.exists("results/" + typeStrategy + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance)):
        os.mkdir("results/" + typeStrategy + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance))

    pathResult = "results/" + typeStrategy + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance) + "/"


    # Chargement des instances QUBO
    if (type_problem == "QUBO"):

        instance_path = "instances/QUBO/"
        tensor_Q_test = getTensorInstances_QUBO(instance_path, nb_instances_test, nb_restarts, N, type_instance, device,
                                                "test")
        

    # Chargement des instances NK
    elif (type_problem == "NK"):

        D = 2
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)

        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            "instances/nk/" + str(dim) + "/" + str(type_instance) + "/", nb_instances_test, nb_restarts, lambda_, dim,
            D, type_instance, device)

    # Chargement des instances NK3 (NK avec 3 catégories)
    elif(type_problem == "NK3"):

        D = 3
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)


        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK("instances/nk3/" + str(dim) + "/" + str(type_instance) + "/", nb_instances_test, nb_restarts, lambda_, dim, D, type_instance, device)

    if (type_problem == "NK3"):
        dim_variables = [3 for i in range(N)] 
    elif (type_problem == "NK"):
        dim_variables = [2 for i in range(N)]      
    elif (type_problem == "QUBO"):
        dim_variables = [2 for i in range(N)]


    if(nh == -1):
        nh = N

    if(numberHiddenLayersG == -1):
        numberHiddenLayersG = int(max(0,math.log2(N)  + max(dim_variables) - 8))


    name_file_result = "Test_" + type_strategy + "_" + type_problem + "_N_" + str(N) + "_t_" + str(
        type_instance) + "_lambda_" + str(lambda_) + "_beta_" + str(beta) + "_nb_train_" + str(
        nb_train)  + "_L_" + str(numberHiddenLayersG) + "_nh_" + str(nh) + "_activation_" + str(activation)  + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(seed) + ".txt"



    # Création de l'EDA
    factory = FactoryStrategyEA()
    
    

        
    strategy = factory.createStrategyEA(typeStrategy, dim, lambda_, mu, rho, beta, device, numberHiddenLayersG, nh,
                                         nb_train, epsilon, learning_rate, dim_variables, activation)


    # Lancement des (nb_instances * nb_restarts) runs
    if (type_problem == "QUBO"):
        list_scores = get_Score_trajectoriesQUBO_cuda(strategy, N, nb_instances_test, nb_restarts, budget, lambda_,
                                                      tensor_Q_test, device, verbose, pathResult + name_file_result)

    elif (type_problem == "NK" or type_problem == "NK3"):
        list_scores = get_Score_trajectoriesNK_cuda(strategy, N, type_instance, D, nb_instances_test, nb_restarts,
                                                    budget, lambda_,
                                                    vectorIndex_th, tensor_matrix_locus,
                                                    tensor_matrix_contrib, device, verbose,
                                                    pathResult + name_file_result)

    print(list_scores)
    average_test_score = np.mean(list_scores)

    print("average_test_score : " + str(average_test_score))




