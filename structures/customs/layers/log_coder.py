import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as Tensor

from ..act_func import ActivationFunction as act

'''
        Assumptions:
        The specific intensity of optical emission spectra is defined as

            I_ij = A_ij * K_ij * hv_ij * N_i = w_ij * N_i

        where,
        A_ij: Einstein A coefficient
        K_ij: Sensitivity factor
        hv_ij: Energy of the transition
        N_i: Density of the species

        We can define the w_ij as

            w_ij = A_ij * K_ij * hv_ij

        We want to find the N, T, that satisfies the following equation:

            N_i = N * exp(-E_i / T)/sum(exp(-E_i / T)) = N * frac(z_i)

        However, they are mixture of species, like following:

            {I_1, I_2, ... I_n} = {w_1 * N_1, w_2 * N_2, ... w_n * N_n}

            = N^A \union N^B \union N^C \union ... \union N^n
            
            where,
            N^A = {w_1 * N^A*frac(z_1^A), w_2 * N^A*frac(z_2^A), ... w_n * N^A*frac(z_i^A)} 
            N^B = {w_1 * N^B*frac(z_1^B), w_2 * N^B*frac(z_2^B), ... w_n * N^B*frac(z_j^B)} 
            ...
            N^n = {w_1 * N^n*frac(z_1^n), w_2 * N^n*frac(z_2^n), ... w_n * N^n*frac(z_k^n)} 

        We have to find the energies, E_ij, that satisfies the above equation, then,

            sum(I_ij^A/w_ij^A) = N^A
            sum(I_ij^B/w_ij^B) = N^B
            ...
            sum(I_ij^n/w_ij^n) = N^n

        Here, what we have to do is to train the network what species corresponds to given wavelength of OES signal.
        How can we do that?

        First, based on the above equation, we will construct the 1 layer network,
        and gradually reduces the number of nodes in the layer.

        The saturation point of the number of nodes would be the optimal number of species.
        After that, we will repeat the same process for the next layer.
        The final layer would be the optimal number of species.
    
            
'''

class UnitLogEncoder(nn.Module):
    def __init__(self, **params):
        super(UnitLogEncoder, self).__init__()
        self.params = params
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.activation_function = params.get('activation_function', 'ReLU')

        self.Encoder = nn.Linear(self.input_dim, self.output_dim+1)
        self.act = act(self.activation_function)

    def forward(self, N: Tensor):
        logN = torch.log(N, dim=-1)
        logN, invT = self.Encoder(logN).split(self.output_dim, dim=-1)
        T = torch.exp(invT)
        N = torch.exp(self.act(logN))
        return N, T

'''
        Assumptions:
        There are quantities, that are summations of densities of specific species.
        The elements of the summation follow the parition rule for specific quantities.

        We call the quantities, that are summations of densities of specific species, as "Seed densities".

        Therfore,
        \scriptN \equiv "Set of seed densities"
        Seed densities are defined as the densities of the total species that follows the partition function.
        They are not varied from the chemical reaction.

        Now, ij species, byproducts of the i-th chemical reaction, in the j-th state, density is defined as
        (here the j-th state means the byproduct of next chemical reaction of byproduct species,
        such as j-th state in Argon; 1s2,1s3, ... or j-th byproduct of CF3; CF2, CF, C.
        As you know, these relations can be linked like chain with effective partition function
        as an example, lowest state of C, and upper states in molecular band spectrum of C):

        \scriptN_ij = \scriptN_i * z_ij / sum(z_ik) = \scriptN_i*softmax(z_ij) ...(1)

        here, z_ij is the j-th species partition function (Boltzmann form).

        z_ij = exp(-(E_ij / \scriptT_i)) ...(2)

        here, 
        \scriptT \equiv "Seed 1/temperatures"
        Seed temperature is corresponding to the j-th species.
        They are not varied from the chemical reaction.

        \scriptE \equiv "Seed energies"
        Seed energies are defined as the energies of the total species that follows the partition function.
        They are not varied from the chemical reaction.

        What we here want to do is training the E_ij, g_ij, alpha_i, n_i.
        The set of Seed densities and Seed temperatures would be the property of given dataset.
        Take the log for (2),

        \log z_ij = - (\E_ij * \scriptT_i)^n_i ...(3)

        Assume, There is a well constructed encoder, encodes the \scriptZ from signal I (the subset of dataset),
        that follows \scriptZ = {\scriptN ,  \scriptT}

        the subset of \scriptZ = \scriptz(+) \equiv \scriptN
        the subset of \scriptZ = \scriptz(-) \equiv \scriptT ...(4)

        we can easily substitute (4) into (3),

        \log z_ij = - (\E_ij * \scriptz(-)[i])

        for vector log z_i, that is set of log z_ij,

        \log z_i = - (\w_i * \scriptz(-)[i]), where \w_i \in \doubleR^[the number of species for 1-th chemical reaction]
        ...(5)

        \log z = Linear(\scriptz(-)[i])
        z = exp(Linear(\scriptz(-)[i]))

        by substituting (5) into \log (1),

        N^1 = softmax(\scriptz(+) * exp(Linear(\scriptz(-)[0])))
        N^2 = softmax(N^1 * exp(Linear(\scriptz(-)[1])))
        ...
        N^n = softmax(N^n-1 * exp(Linear(\scriptz(-)[n])))

'''
class UnitLogDecoder(nn.Module):
    def __init__(self, **params):
        super(UnitLogDecoder, self).__init__()
        self.params = params

        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.activation_function = params.get('activation_function', 'ReLU')

        self.Energies = nn.Linear(self.input_dim, self.output_dim)
        self.act = act(self.activation_function)

    def forward(self, N: Tensor, Ti: Tensor):
        return F.softmax(-self.act(self.Energies(Ti)),dim=-1) * N
