
import torch
from eda_strategies.Abstract_EDA import Abstract_EDA

import numpy as np
from torch.distributions import kl_divergence

from utils.rl_eda_utils import RL_EDA_generator, OrderGenerator
from torch.distributions import Categorical

import torch.nn.utils.prune as prune
import copy

class RL_EDA(Abstract_EDA):

    def __init__(self, N,  lambda_,  beta,  device,  numberHiddenLayersG, nh,    nb_train, epsilon, learning_rate, dim_variables):

        Abstract_EDA.__init__(self, N, lambda_, device)

        self.numberHiddenLayersG = numberHiddenLayersG
        self.nh = nh
        self.nb_train = nb_train
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.beta = beta

        # Creating functions to set thresholds for probabilities
        self.first_threshold = torch.nn.Threshold(self.epsilon, self.epsilon)
        self.second_threshold = torch.nn.Threshold(- 1 + self.epsilon, - 1 + self.epsilon)
        
        # Calculation of advantage scores for the lambdas individuals in each generation (scaled linearly between -1 and 1)
        self.advantages = torch.tensor(np.linspace(-1, 1, num=self.lambda_)).to(self.device)

        self.dim_variables = dim_variables

        
        if dim_variables is not None:
            self.max_dim = max(dim_variables)

    # Initializing solution generators: order generator and value generator
    def reset_learned_parameters(self, nb_runs):

        

        self.nb_runs = nb_runs

        # Initializing order generator 
        self.orderGenerator = OrderGenerator(self.device,  nb_runs, self.lambda_, self.N).to(self.device)
        
        # Initializing value generator 
        self.generator = RL_EDA_generator((self.nb_runs, self.lambda_, self.N), self.nh,self.lambda_,numberHiddenLayersG=self.numberHiddenLayersG, device=self.device, cat_sizes= self.dim_variables).to(self.device)
        
        self.generator.reset_parameters()


        #  Calculating specific masks when the number of categories in the problem varies. Note: This is not used for NK3 instances because the number of categories is always 3.
        self.different_number_of_categories = False
        self.mask_categorical = False
        
        if (self.dim_variables is not None):
            
            if(len(set(self.dim_variables))!=1):
                
                self.different_number_of_categories = True

                self.mask_categorical = torch.zeros(self.N, self.max_dim)
                self.mask_categorical2 = torch.ones(self.N, self.max_dim)

                for idx, dim in enumerate(self.dim_variables):
                    self.mask_categorical[idx, dim:] = -float("inf")
                    self.mask_categorical2[idx, dim:] = 0

                self.mask_categorical = self.mask_categorical.unsqueeze(0).unsqueeze(0).repeat([nb_instances, self.lambda_, 1, 1]).to(self.device)
                self.mask_categorical2 = self.mask_categorical2.unsqueeze(0).unsqueeze(0).repeat([nb_instances, self.lambda_, 1, 1]).to(self.device)
                

    ## Solution Generation
    def sample_solutions(self):


        with torch.no_grad():

            ## Initializing the population. 0 means the value has not been assigned. We will set it to -1 or 1 once it has been assigned.
            ## We generate individuals in parallel for the nb_runs and lambda of each independent run.
            ## However, the generation of the N variables is sequential.

            new_pop = torch.zeros((self.nb_runs, self.lambda_, self.N)).to(self.device)

            ## Generating a random order, to which causal masks from the directed acyclic graph used for the generation correspond
            order_variables, dag = self.orderGenerator.get_order()

            order_variables = order_variables.long().data
            self.mask_gen = dag.long().data

            ## Sequential Generation Using the Autoregressive Model
            for i in range(0, self.N):


                 ## Retrieving the mask for the variable to be generated
                DAG_input = self.mask_gen.gather(3, order_variables[:, :, i].unsqueeze(2).data.unsqueeze(3).repeat(1, 1, self.N, 1))

                # Calculating specific masks if the number of categories varies for each variable
                if (self.different_number_of_categories):
                    mask_output = self.mask_categorical.gather(2, order_variables[:, :, i].unsqueeze(2).data.unsqueeze(3).repeat(1, 1, 1, self.max_dim)).squeeze(2)
                    mask_output2 = self.mask_categorical2.gather(2, order_variables[:, :, i].unsqueeze(2).data.unsqueeze(3).repeat(1, 1, 1, self.max_dim)).squeeze(2)
                else:
                    mask_output = None
                        
                ## Calculating the probability of generating each value for the variable in question
                probas = self.generator(new_pop, DAG_input.squeeze(),  mask_output, order_variables[:, :, i])

                ## Application of the probability threshold [epsilon, 1 - epsilon]
                probas = self.first_threshold(probas)
                probas = - self.second_threshold(-probas)

                if (self.different_number_of_categories):
                    probas = probas*mask_output2
                        
                ## Sampling the actual value of the variable in question based on the probabilities
                if(self.dim_variables is not None):

                    categorical_dist = Categorical(probas)
                    variable_ouput = categorical_dist.sample().float()

                else:
                    variable_ouput = torch.bernoulli(probas)

                ## Inserting the new variable value into each solution
                new_pop.scatter_(2, order_variables[:, :, i].unsqueeze(2), variable_ouput.unsqueeze(2))


        return new_pop.unsqueeze(3).data

    def toString(self):

        return "Strategy_RL_EDA_"



    def updateDistribution(self, solutionList, scoreList):

        
        ## Sorting solutions and sampling masks by fitness Value
        sorted, indices = torch.sort(scoreList, dim=1)

        input_pop = (solutionList.squeeze(3)).gather(1, indices.unsqueeze(2).repeat([1, 1, self.N])).detach()
        target = input_pop.data
        
        sorted_mask_gen = self.mask_gen.gather(1, indices.unsqueeze(2).unsqueeze(3).repeat([1, 1, self.N, self.N])) 


        ## Initializing the Adam optimizer
        optimizer = torch.optim.Adam(list(self.generator.parameters()), lr=self.learning_rate)

   
        ## Recalculation of the probabilities of sampling each value of variable (calculation of \pi_old(x) for PPO/GRPO)
        with torch.no_grad():

            init_distributions = self.generator(input_pop.data, sorted_mask_gen, self.mask_categorical).data

            init_distributions = self.first_threshold(init_distributions)
            init_distributions = - self.second_threshold(-init_distributions)

            if (self.dim_variables is not None):
                proba_action_init = init_distributions.gather(3, target.unsqueeze(3).long()).squeeze(3)
            else:
                proba_action_init = torch.where(target == 1, init_distributions, 1 - init_distributions)



        pbar = range(self.nb_train)


        ## Launch of the EDA Generator Update Loop
        for epoch in pbar:

            optimizer.zero_grad()

            ## Random sampling of new training masks
            _, train_mask = self.orderGenerator.get_order()

            ## Calculating probabilities using these new input masks and the solutions that had been generated
            probas_g = self.generator(input_pop.data, train_mask, self.mask_categorical)

            ## Application of the probability threshold
            probas_g = self.first_threshold(probas_g)
            probas_g = - self.second_threshold(-probas_g)

            ## Retrieving the probability corresponding to the value generated for each variable
            if (self.dim_variables is not None):
                generated_probas_action = probas_g.gather(3, target.unsqueeze(3).long()).squeeze(3)
            else:
                generated_probas_action = torch.where(target == 1, probas_g,
                                                    1 - probas_g)


            ##  Calculating the Importance Sampling Ratio for PPO/GRPO
            ratio = -generated_probas_action / proba_action_init.data

            ##  Weighting by Advantage Scores
            weighted_loss = torch.transpose(ratio, 1, 2) * self.advantages

                            
                            
            ###Calculation of the KL divergence between the old model and the new one
    
            if(self.dim_variables is not None):
                d_kl = kl_divergence(torch.distributions.Categorical(probs=init_distributions.data),
                                            torch.distributions.Categorical(probs=probas_g)).mean()
            else:
                d_kl = kl_divergence(torch.distributions.bernoulli.Bernoulli(probs=init_distributions.data),
                                            torch.distributions.bernoulli.Bernoulli(probs=probas_g)).mean()
                                
                
            #d_kl = kl_divergence(torch.distributions.bernoulli.Bernoulli(probs=init_distributions.data),torch.distributions.bernoulli.Bernoulli(probs=probas_g)).mean()
            
            
            ## Calculating the global loss 
            global_loss = torch.mean(weighted_loss)  + self.beta * d_kl

            ## Gradient calculation
            global_loss.backward()

            ## Updating generator parameters
            optimizer.step()
            
            


