import math
import torch
from torch.nn import Parameter
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm

class ChannelBatchNorm1d(_BatchNorm):
    def __init__(self, num_channels, num_features, *args, **kwargs):
        super(ChannelBatchNorm1d, self).__init__(num_channels*num_features, *args, **kwargs)
        self.num_channels = num_channels
        self.num_features = num_features

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        _input = input.contiguous().view(-1, self.num_channels * self.num_features)
        output = super(ChannelBatchNorm1d, self).forward(_input)
        return torch.transpose(output.view(-1, self.num_channels, self.num_features), 0, 1)


class OrderGenerator(torch.nn.Module):

    def __init__(self, device, nb_runs, size_pop, N):
        super(OrderGenerator, self).__init__()

        self.N = N
        self.nb_runs = nb_runs
        self.size_pop = size_pop
        self.device = device

        self.A = torch.tensor(np.triu(np.ones((self.nb_runs, size_pop, self.N, self.N)), k=1)).to(
            self.device).float()


    def get_order(self):

        with torch.no_grad():

            #order = torch.argsort(torch.rand(self.nb_runs, self.size_pop, self.N)).to(self.device)
            order = torch.argsort(torch.rand(self.nb_runs, self.size_pop, self.N, device=self.device), dim=-1)
            P = torch.nn.functional.one_hot(order, num_classes=self.N).float()
            mask = torch.transpose(P, 2, 3) @ self.A.float() @ P

        return order, mask





class LinearCustom(torch.nn.Module):

    def __init__(self, nb_instances, channels, in_features, out_features, size_pop, batch_size=-1, bias=True):
        super(LinearCustom, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels

        self.size_pop = size_pop

        self.weight = Parameter(torch.Tensor(nb_instances, self.in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(nb_instances, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data, order_variable=None):



        if (order_variable is not None):
            output = data @ self.weight
            output = (output + self.bias.unsqueeze(1)).squeeze(-1)

        else:

            output = data @ self.weight.unsqueeze(1)
            output = (output + self.bias.unsqueeze(1).unsqueeze(1)).squeeze(-1)


        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class RL_EDA_generator(torch.nn.Module):
    """Ensemble of all the generators."""


    def __init__(self, data_shape, nh, size_pop,  numberHiddenLayersG=1,
                 device="cuda:0", cat_sizes=None, activation="Tanh"):
        """Init the model."""
        super(RL_EDA_generator, self).__init__()


        self.size_pop = size_pop

        nb_vars = data_shape[2]

        self.nb_vars = nb_vars
        self.batch_size = data_shape[0]
        
        if(activation == "Tanh"):
            self.activation = torch.nn.Tanh()
        elif(activation == "relu"):
            self.activation = torch.nn.ReLU()
        elif(activation == "gelu"):
            self.activation = torch.nn.GELU()
        elif(activation == "leaky_relu"):
            self.activation = torch.nn.LeakyReLU(negative_slope=0.01)

        self.sizes = cat_sizes

        if cat_sizes is not None:

            self.max_cat_size = max(cat_sizes)
            size_data_input = self.max_cat_size * self.nb_vars
            output_dim = self.max_cat_size

        else:
            output_dim = 1
            size_data_input = nb_vars

        self.device = device




        output_dim = output_dim * nb_vars
        index_variables = torch.arange(nb_vars).to(device)

        if cat_sizes is not None:
            self.tensor_index_variables = index_variables.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat([data_shape[0],size_pop,1,1,self.max_cat_size])
        else:

            self.tensor_index_variables = index_variables.unsqueeze(0).unsqueeze(0).repeat(
                [data_shape[0], size_pop, 1]).unsqueeze(3)


        self.list_input_layer = []
        self.list_input_layer.append(LinearCustom(data_shape[0], nb_vars, size_data_input, nh, size_pop))


        self.list_hidden_layer = []

        print("numberHiddenLayersG")
        print(numberHiddenLayersG)

        #self.list_hidden_layer = torch.nn.ModuleList()        

        for i in range(numberHiddenLayersG):
            self.list_hidden_layer.append(LinearCustom(data_shape[0], nb_vars, nh, nh, size_pop))
            #self.list_batch_norm.append(ChannelBatchNorm1d(data_shape[0],  nh))


        self.output_layer = LinearCustom(data_shape[0], nb_vars, nh, output_dim, size_pop)



    def forward(self, data, mask_input, mask_output, order_variables=None):

        if (order_variables is not None):

            if self.sizes is not None:

                data_input_tmp = torch.nn.functional.one_hot(data.long(), self.max_cat_size).float() * 2 -1
                data_input = data_input_tmp * mask_input.unsqueeze(3)
                data_input = data_input.view(data_input.size()[0], data_input.size()[1], -1)
                
            else:
            
                data_input = (data  * 2 - 1)* mask_input


            out = self.list_input_layer[0](data_input, order_variables)
            out = self.activation(out)

            for idx, hidden_layer in enumerate(self.list_hidden_layer):
                out = hidden_layer(out, order_variables)
                out = self.activation(out)

            output = self.output_layer(out, order_variables)


            if self.sizes is not None:
                output = output.view(self.batch_size,self.size_pop,self.nb_vars,  self.max_cat_size)
                output = torch.gather(output, 2, order_variables.unsqueeze(2).unsqueeze(3).repeat([1,1,1,self.max_cat_size])).squeeze(2)
            else:
                output = torch.gather(output,2,order_variables.unsqueeze(2)).squeeze(-1)

        else:

            data = data.unsqueeze(2).repeat(1, 1, self.nb_vars, 1)
            

            if self.sizes is not None:

                data_input_tmp = torch.nn.functional.one_hot(data.long(), self.max_cat_size).float() * 2 -1
                data_input = data_input_tmp * mask_input.unsqueeze(4)
                data_input = data_input.view(data_input.size()[0], data_input.size()[1], data_input.size()[2], -1)

            else:            
            
                data_input = (data  * 2 - 1)* mask_input

            

            out = self.list_input_layer[0](data_input)
            out = self.activation(out)

            for idx, hidden_layer in enumerate(self.list_hidden_layer):
                out = hidden_layer(out)
                out = self.activation(out)

            output = self.output_layer(out)

            if self.sizes is not None:

                output = output.view(self.batch_size, self.size_pop, self.nb_vars, self.nb_vars, self.max_cat_size)
                output = torch.gather(output, 3, self.tensor_index_variables).squeeze(3)

            else:
                
                output = torch.gather(output, 3, self.tensor_index_variables).squeeze(-1)

        if self.sizes is not None:

            if(mask_output  is not None):
                output = output + mask_output

            output = torch.softmax(output ,  -1)

        else:
            output = torch.sigmoid(output)

        return output

    def reset_parameters(self):

        #self.input_layer.reset_parameters()

        self.list_input_layer[0].reset_parameters()
        self.list_input_layer[0].to(self.device)

        self.output_layer.reset_parameters()

        for hidden_layer in self.list_hidden_layer:
            hidden_layer.reset_parameters()
            hidden_layer.to(self.device)
            






