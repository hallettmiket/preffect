import os
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from _distributions import (
    NegativeBinomial,
    ZeroInflatedNegativeBinomial
)        

from torch.nn.functional import linear

import numpy as np
from _utils import (
    print_loss_table,
    check_for_nans,
    check_folder_access,
    copy_file,
    one_hot_encode
)


class Encoder(nn.Module):
    """
    Encoder neural network module using Graph Attention Networks (GATs)

    :ivar layer1: First graph attention layer.
    :vartype layer1: GATv2Conv
    :ivar layer2: Second linear transformation layer.
    :vartype layer2: nn.Linear
    :ivar layer3: Third linear transformation layer.
    :vartype layer3: nn.Linear
    :ivar mu_layer: Linear layer to compute the mean of the latent space representation.
    :vartype mu_layer: nn.Linear
    :ivar logvar_layer: Linear layer to compute the log variance of the latent space representation.
    :vartype logvar_layer: nn.Linear
    :ivar leaky_relu: LeakyReLU activation function.
    :vartype leaky_relu: nn.LeakyReLU
    """

    def __init__(self, in_channels, k, r_prime, r, h, alpha, dropout, model_type, correction):
        """
        Initialize the Encoder instance

        :param in_channels: Number of input features.
        :type in_channels: int
        :param k: List of integers representing the number of categories for each correction variable.
                  None values indicate continuous variables.
        :type k: List[int]
        :param r_prime: Dimensionality of the intermediate space.
        :type r_prime: int
        :param r: Dimensionality of the output space (latent space).
        :type r: int
        :param h: Number of attention heads for multi-head attention.
        :type h: int
        :param alpha: Negative slope for the LeakyReLU activation function.
        :type alpha: float
        :param dropout: Dropout probability for regularization.
        :type dropout: float
        :param model_type: Type of the model ('single', 'full', or 'simple').
        :type model_type: str
        :param correction: Flag indicating whether to apply correction using categorical variables.
        :type correction: bool
        """
        super(Encoder, self).__init__()

        self.model_type = model_type
  
        k_sans_none = [elem for elem in k if elem is not None]
        k_none = len(k) - len(k_sans_none)

        if correction is True:
            self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=sz, embedding_dim=r) for sz in k_sans_none])
            adjusted_input_dim = in_channels + len(k_sans_none) * r + k_none 
        else:
            adjusted_input_dim = in_channels


        self.layer1 = GATv2Conv(in_channels=adjusted_input_dim,
                                out_channels=r_prime,
                                heads=h,
                                negative_slope=alpha)

        self.layer1_simple = nn.Linear(in_features = adjusted_input_dim, out_features=r_prime * len(k))


        self.layer2 = nn.Linear(in_features = r_prime * h, out_features=r_prime)
        self.layer3 = nn.Linear(in_features = r_prime, out_features=r_prime)

        self.mu = nn.Linear(in_features=r_prime + len(k), out_features=r)
        self.logvar  = nn.Linear(in_features=r_prime + len(k), out_features=r) 


        self.layer2_simple = nn.Linear(in_features = r_prime * len(k), out_features=r_prime)
        self.mu_simple = nn.Linear(in_features=r_prime, out_features=r)
        self.logvar_simple = nn.Linear(in_features=r_prime, out_features=r)

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.apply(self.init_weights)

    def prepare_latent_space_with_korrection_vars(self, K, k, lat_space):
        r"""
        :param K: List of correction variable tensors.
        :type K: List[torch.Tensor]
        :param k: List of integers representing the number of categories for each correction variable.
                None values indicate continuous variables.
        :type k: List[int]
        :param lat_space: Latent space representation tensor.
        :type lat_space: torch.Tensor

        :return: A tuple containing:
                - `h`: The modified latent space tensor with correction variables incorporated.
                - `total_cat`: The sum of the embedded categorical variables, or None if no categorical variables are present.
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]
        """

        continuous_indices = [i for i, x in enumerate(k) if x is None]
        categorical_indices = [i for i, x in enumerate(k) if x is not None]
    
        h = lat_space
        if len(continuous_indices)>0:
            #h = torch.cat((h, torch.stack(K[continuous_indices], dim=1)), dim=1) 
            h = torch.cat((h, torch.stack([K[i] for i in continuous_indices], dim=1)), dim=1)
        
        if len(categorical_indices)> 0:
            total_cat = None
            for idx in categorical_indices:
                
                num_embeddings = self.embeddings[0].num_embeddings

                holder = self.embeddings[idx](K[idx])
                if total_cat is None:
                    total_cat = holder
                else:
                    total_cat += holder
                h = torch.cat((h, holder), dim = 1)
   
        return h, total_cat

    def encode(self, X, ejs, K, k, correction):
        """
        Perform encoding using graph attention layers

        :param X: Input feature matrix (zeroed counts).
        :type X: torch.Tensor
        :param ejs: Edge indices defining the graph structure.
        :type ejs: torch.Tensor
        :param K: List of correction variable tensors.
        :type K: List[torch.Tensor]
        :param k: List of integers representing the number of categories for each correction variable.
                None values indicate continuous variables.
        :type k: List[int]
        :param correction: Flag indicating whether to apply correction using the correction variables.
        :type correction: bool

        :return: A tuple containing:
                - `mu`: Mean of the encoded input.
                - `logvar`: Log variance of the encoded input.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        if correction is True:
            h, total_cat = self.prepare_latent_space_with_korrection_vars(K, k, X)
        else:
            h = X

        if self.model_type in ('single', 'full'):
            h = self.layer1(h, ejs)
            h = F.elu(h)
            h = self.dropout1(h)

            h = self.layer2(h)
            h = F.elu(h)
            h = self.dropout2(h)

            # for testing: third layer
            h = self.dropout3(F.elu(self.layer3(h)))

            # K needs a reshape for the concatenation
            K_stack = torch.stack(K, dim=1)   

            mu = self.leaky_relu(self.mu(torch.cat((h, K_stack), dim=1)))
            logvar = self.leaky_relu(self.logvar(torch.cat((h, K_stack), dim=1)))
            
            if correction is True:
                mu += total_cat

            

        elif self.model_type=='simple':
            h = self.layer1_simple(h)
            h = F.elu(h)
            h = self.dropout1(h)

            h = self.layer2_simple(h)
            h = F.elu(h)    
            h = self.dropout2(h)
            
            mu = self.leaky_relu(self.mu_simple(h))

            if correction is True:
                mu += total_cat
                unique_rows, counts = torch.unique(total_cat, dim=0, return_counts=True)


            logvar = self.leaky_relu(self.logvar_simple(h))
        
        return mu, logvar

    def init_weights(self, m):
        """
        Initialize weights of Linear layers 
        
        :param m: A PyTorch module instance.
        :type m: nn.Module

        :return: None
        """

        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, GATv2Conv):
            for param in m.parameters():
                if param.dim() > 1:  # Typically, weights have more than one dimension
                    torch.nn.init.xavier_uniform_(param)
                else:  # Biases typically have one dimension
                    torch.nn.init.zeros_(param)

class Decoder(nn.Module):
    """
    Decoder neural network module for a GAT-based autoencoder

    :param r: Dimensionality of the input latent space.
    :type r: int
    :param r_prime: Dimensionality of the intermediate space.
    :type r_prime: int
    :param k: List of the number of categories for each categorical variable. None values are ignored.
    :type k: List[int]
    :param final: Number of output features.
    :type final: int
    :param alpha: Negative slope for the LeakyReLU activation function.
    :type alpha: float
    :param dropout: Dropout probability for regularization.
    :type dropout: float
    :param model_type: Type of the model ('single', 'full', or 'simple').
    :type model_type: str
    :param correction: Flag indicating whether to apply correction using categorical variables.
    :type correction: bool

    :ivar layer1: First linear transformation layer.
    :vartype layer1: nn.Linear
    :ivar layer2: Second linear transformation layer.
    :vartype layer2: nn.Linear
    :ivar layer3: Third linear transformation layer.
    :vartype simple_layer: nn.Linear
    :ivar leaky_relu: LeakyReLU activation function.
    :vartype leaky_relu: nn.LeakyReLU
    :ivar dropout1: Dropout layer after the first linear transformation.
    :vartype dropout1: nn.Dropout
    :ivar dropout2: Dropout layer after the second linear transformation.
    :vartype dropout2: nn.Dropout
    :ivar dropout: Dropout layer for the 'simple' model type.
    :vartype dropout: nn.Dropout
    """

    def __init__(self, r, r_prime, k, final, alpha, dropout, model_type, correction):
        """
        Initialize Decoder instance

        Args:
            r (int): Dimensionality of the input latent space
            r_prime (int): Dimensionality of intermediate space
            k (int): Number of clinical/technical variables
            final (int): Number of output features
            alpha (float): Negative slope for LeakyReLU
        """

        super(Decoder, self).__init__()
        self.model_type = model_type

        k_sans_none = [elem for elem in k if elem is not None]
        k_none = len(k) - len(k_sans_none)

        if correction is True:
            adjusted_input_dim = r + len(k_sans_none) * r + k_none 
        else:
            adjusted_input_dim = r

        # single and full models
        self.layer1 = nn.Linear(in_features=adjusted_input_dim, out_features=r_prime)
        self.layer2 = nn.Linear(in_features = r_prime, out_features=r_prime)
        self.layer3 = nn.Linear(in_features=r_prime, out_features=final)

        # simple model
        self.simple_layer = nn.Linear(in_features=adjusted_input_dim, out_features=r_prime)
        
        if correction is True:
            self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=sz, embedding_dim=r) for sz in k_sans_none])

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(self.init_weights)

    def prepare_latent_space_with_korrection_vars(self, K, k, lat_space):
        r"""
        :param K: List of correction variable tensors.
        :type K: List[torch.Tensor]
        :param k: List of integers representing the number of categories for each correction variable.
                None values indicate continuous variables.
        :type k: List[int]
        :param lat_space: Latent space representation tensor.
        :type lat_space: torch.Tensor

        :return: A tuple containing:
                - `h`: The modified latent space tensor with correction variables incorporated.
                - `total_cat`: The sum of the embedded categorical variables, or None if no categorical variables are present.
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]
        """

        continuous_indices = [i for i, x in enumerate(k) if x is None]
        categorical_indices = [i for i, x in enumerate(k) if x is not None]

        h = lat_space
        if len(continuous_indices)>0:
            h = torch.cat((h, torch.stack([K[i] for i in continuous_indices], dim=1)), dim=1)

        if len(categorical_indices)> 0:
            total_cat = None
            for idx in categorical_indices:
                holder = self.embeddings[idx](K[idx])
                if total_cat is None:
                    total_cat = holder
                else:
                    total_cat += holder

                h = torch.cat((h, holder), dim = 1)
        
        return h, total_cat
    
    def decode(self, Z, ejs, K, k, correction):
        """
        Decodes using linear transformations and activations

        :param Z: Encoded latent space representation tensor.
        :type Z: torch.Tensor
        :param ejs: Placeholder parameter (not used in the method).
        :type ejs: Any
        :param K: List of correction variable tensors.
        :type K: List[torch.Tensor]
        :param k: List of integers representing the number of categories for each correction variable.
                None values indicate continuous variables.
        :type k: List[int]
        :param correction: Flag indicating whether to apply correction using the correction variables.
        :type correction: bool

        :return: Decoded output tensor in the original feature space.
        :rtype: torch.Tensor
        """

        if correction is True:
            h, _ = self.prepare_latent_space_with_korrection_vars(K, k, Z)
        else:  
            h = Z

        if self.model_type in ('single', 'full'):
            h = self.dropout1( self.leaky_relu( self.layer1 ( h ) ) )
            h = self.dropout2( self.leaky_relu( self.layer2 ( h ) ) )
            h = self.sigmoid( self.layer3( h ) )

        elif self.model_type=='simple':
            h = self.dropout(self.leaky_relu(self.simple_layer(h)))

        return h

    def init_weights(self, m):
        """
        Initialize weights of Linear layers using Xavier initialization

        :param m: A PyTorch module instance.
        :type m: nn.Module

        :return: None
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)

            if m.bias is not None:
                m.bias.data.fill_(0.01)



class LibEncoder(nn.Module):
    """
    Library-size encoder module using a linear transformation layer followed
    by LeakyReLU activation

    :ivar layer_lib1: Linear layer to encode the combined feature of log library sizes and variables.
    :vartype layer_lib1: nn.Linear
    :ivar layer_lib_mu: Linear layer to encode the mean for the library size.
    :vartype layer_lib_mu: nn.Linear
    :ivar layer_lib_logvar: Linear layer to encode the log variance (logvar) for the library size.
    :vartype layer_lib_logvar: nn.Linear
    :ivar leaky_relu: Leaky ReLU activation function.
    :vartype leaky_relu: nn.LeakyReLU
    :ivar dropout1: Dropout layer to prevent overfitting.
    :vartype dropout1: nn.Dropout
    :ivar embeddings: Module list of embedding layers for categorical correction variables.
    :vartype embeddings: nn.ModuleList
    """
    def __init__(self, k, r_prime, r, alpha, dropout, correction):
        """
        Initialize LibEncoder instance

        :param k: List of integers representing the number of categories for each correction variable.
                  None values indicate continuous variables.
        :type k: List[int]
        :param r_prime: Dimensionality of the intermediate feature space.
        :type r_prime: int
        :param r: Dimensionality of the output feature space (latent space).
        :type r: int
        :param alpha: Negative slope for the LeakyReLU activation function.
        :type alpha: float
        :param dropout: Dropout rate used in the dropout layer to prevent overfitting.
        :type dropout: float
        :param correction: Flag indicating whether to apply correction using categorical variables.
        :type correction: bool
        """
        super(LibEncoder, self).__init__()

        k_sans_none = [elem for elem in k if elem is not None]
        k_none = len(k) - len(k_sans_none)

        # add up number of levels over all cat correction variables and use r per each; continous variables are tacked on.
        if correction is True:
            adjusted_input_dim = 1 + len(k_sans_none) * r + k_none 
        else:
            adjusted_input_dim = 1

        self.layer_lib1 = nn.Linear(in_features=adjusted_input_dim, out_features=r_prime)
        self.layer_lib_mu = nn.Linear(in_features=r_prime, out_features=r)
        self.layer_lib_logvar = nn.Linear(in_features=r_prime, out_features=r)

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout1 = nn.Dropout(p=dropout)

        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=sz, embedding_dim=r) for sz in k_sans_none])

        self.apply(self.init_weights)
        
    def prepare_latent_space_with_korrection_vars(self, K, k, lat_space):
        r"""
        :param K: List of correction variable tensors.
        :type K: List[torch.Tensor]
        :param k: List of integers representing the number of categories for each correction variable.
                None values indicate continuous variables.
        :type k: List[int]
        :param lat_space: Latent space representation tensor.
        :type lat_space: torch.Tensor

        :return: A tuple containing:
                - `h`: The modified latent space tensor with correction variables incorporated.
                - `total_cat`: The sum of the embedded categorical variables, or None if no categorical variables are present.
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]
        """

        continuous_indices = [i for i, x in enumerate(k) if x is None]
        categorical_indices = [i for i, x in enumerate(k) if x is not None]

        h = lat_space
        if len(continuous_indices)>0:
            #h = torch.cat((h, torch.stack(K[continuous_indices], dim=1)), dim=1) 
            h = torch.cat((h, torch.stack([K[i] for i in continuous_indices], dim=1)), dim=1)

        if len(categorical_indices)> 0:
            total_cat = None
            for idx in categorical_indices:
                holder = self.embeddings[idx](K[idx])
                if total_cat is None:
                    total_cat = holder
                else:
                    total_cat += holder
                h = torch.cat((h, holder), dim = 1)
        return h, total_cat

    def encode(self, log_lib, K, k, correction):
        """
        Perform the encoding operation for log library sizes and variables

        :param log_lib: Tensor of log library sizes.
        :type log_lib: torch.Tensor
        :param K: List of correction variable tensors.
        :type K: List[torch.Tensor]
        :param k: List of integers representing the number of categories for each correction variable.
                None values indicate continuous variables.
        :type k: List[int]
        :param correction: Flag indicating whether to apply correction using the correction variables.
        :type correction: bool

        :return: A tuple containing:
                - `mu`: Tensor representing the mean of the latent space representation.
                - `logvar`: Tensor representing the log variance of the latent space representation.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """

        if correction is True:
            h, total_cat = self.prepare_latent_space_with_korrection_vars(K, k, log_lib.view(-1,1))
        else:
            h = log_lib.view(-1,1)

        #h = self.leaky_relu(self.dropout1(self.layer_lib1(h)))
        h = self.dropout1(self.leaky_relu(self.layer_lib1(h)))
        
        mu = self.layer_lib_mu(h)
        logvar = self.layer_lib_logvar(h)

        if correction is True:        
            mu += total_cat

        return mu, logvar

    def init_weights(self, m):
        """
        Initialize weights of Linear layers using Xavier initialization

        :param m: A PyTorch module instance.
        :type m: nn.Module

        :return: None
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class LibDecoder(nn.Module):
    """
    Library-size decoder module that decodes latent variables back into
    the original space

    :ivar lib_decode_size_factor: Linear layer for decoding the latent variables.
    :vartype lib_decode_size_factor: nn.Linear
    :ivar lib_decode_size_factor_2: Linear layer for further decoding the library size.
    :vartype lib_decode_size_factor_2: nn.Linear
    :ivar leaky_relu: Leaky ReLU activation function.
    :vartype leaky_relu: nn.LeakyReLU
    :ivar embeddings: Module list of embedding layers for categorical correction variables.
    :vartype embeddings: nn.ModuleList

    """
    def __init__(self, r, r_prime, k, final, alpha, correction):
        """
        Initialize LibDecoder instance.

        :param r: Dimensionality of the output feature space (latent space).
        :type r: int
        :param r_prime: Dimensionality of the intermediate feature space.
        :type r_prime: int
        :param k: List of integers representing the number of categories for each correction variable.
                  None values indicate continuous variables.
        :type k: List[int]
        :param final: Number of final output features (library size).
        :type final: int
        :param alpha: Negative slope for the LeakyReLU activation function.
        :type alpha: float
        :param correction: Flag indicating whether to apply correction using categorical variables.
        :type correction: bool
        """
        super(LibDecoder, self).__init__()

        k_sans_none = [elem for elem in k if elem is not None]  #categorical variables
        k_none = len(k) - len(k_sans_none)

        # add up number of levels over all cat correction variables and use r per each; continous variables are tacked on.
        if correction is True: 
            adjusted_input_dim = (1 + len(k_sans_none)) * r + k_none
        else:
            adjusted_input_dim = r

        self.lib_decode_size_factor = nn.Linear(in_features=adjusted_input_dim, out_features=r_prime)
        self.lib_decode_size_factor_2 = nn.Linear(in_features=r_prime, out_features=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)

        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=sz, embedding_dim=r) for sz in k_sans_none])

        self.apply(self.init_weights)

    def prepare_latent_space_with_korrection_vars(self, K, k, lat_space):
        r"""
        :param K: List of correction variable tensors.
        :type K: List[torch.Tensor]
        :param k: List of integers representing the number of categories for each correction variable.
                None values indicate continuous variables.
        :type k: List[int]
        :param lat_space: Latent space representation tensor.
        :type lat_space: torch.Tensor

        :return: A tuple containing:
                - `h`: The modified latent space tensor with correction variables incorporated.
                - `total_cat`: The sum of the embedded categorical variables, or None if no categorical variables are present.
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]
        """

        continuous_indices = [i for i, x in enumerate(k) if x is None]
        categorical_indices = [i for i, x in enumerate(k) if x is not None]

        h = lat_space
        if len(continuous_indices)>0:
            #h = torch.cat((h, torch.stack(K[continuous_indices], dim=1)), dim=1) 
            h = torch.cat((h, torch.stack([K[i] for i in continuous_indices], dim=1)), dim=1)

        if len(categorical_indices)> 0:
            total_cat = None
            for idx in categorical_indices:
                holder = self.embeddings[idx](K[idx])
                if total_cat is None:
                    total_cat = holder
                else:
                    total_cat += holder
                h = torch.cat((h, holder), dim = 1)
        return h, total_cat

    def decode(self, Z_L, K, k, correction):
        """
        Perform the decoding operation for latent variables and correction variables.

        :param Z_L: Latent variables tensor.
        :type Z_L: torch.Tensor
        :param K: List of correction variable tensors.
        :type K: List[torch.Tensor]
        :param k: List of integers representing the number of categories for each correction variable.
                None values indicate continuous variables.
        :type k: List[int]
        :param correction: Flag indicating whether to apply correction using the correction variables.
        :type correction: bool

        :return: Decoded output tensor representing the reconstructed library size.
        :rtype: torch.Tensor
        """

        if correction is True:
            h, _ = self.prepare_latent_space_with_korrection_vars(K, k, Z_L)
        else:
            h = Z_L

        h = self.leaky_relu(self.lib_decode_size_factor(h))
        h= self.leaky_relu(self.lib_decode_size_factor_2(h))
        return h

    def init_weights(self, m):
        """
        Initialize weights of Linear layers using Xavier initialization

        :param m: A PyTorch module instance.
        :type m: nn.Module

        :return: None
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class VAE(nn.Module):
    r"""
    Variational Autoencoder (VAE)  to capture the latent structure
    Prepares encoders/decoders/linear layers of the VAE

    :param N: Number of genes.
    :type N: int
    :param M: Number of samples.
    :type M: int
    :param ks: List of integers representing the number of categories for each correction variable.
    :type ks: List[int]
    :param log: Logger for outputting information during model operations.
    :type log: logging.Logger
    :param configs: Configuration dictionary containing parameters for the VAE.
    :type configs: dict

    """

    def __init__(self, N, M, ks, log, configs):
        r"""
        Initializing VAE instance

        Args:
            N (int): Number of genes.
            M (int): Number of samples.
            ks (list): List of ks (number of correction variables).
            log (logging.Logger): Logger for outputting information during model operations.
            configs (dict): Configuration dictionary containing parameters like epsilon, calT (number of tasks), 
        """
        super(VAE, self).__init__()
        self.configs = configs
        self.log = log
        self.epsilon = self.configs['epsilon']
        self.calT = self.configs['calT']

        r_prime, r, h, alpha, dropout, model_type, correction = \
            self.configs['r_prime'],\
            self.configs['r'],\
            self.configs['h'],\
            self.configs['alpha'],\
            self.configs['dropout'],\
            self.configs['type'],\
            self.configs['correct_vars']

        if self.configs['infer_lib_size'] is True:
            self.encoders_L = nn.ModuleList(
                [LibEncoder(k=ks[_], r_prime=r_prime, r=r, alpha=alpha, dropout=dropout, correction=correction)
                 for _ in range(self.calT)])
            self.decoders_L = nn.ModuleList(
                [LibDecoder(r_prime=r_prime, r=r, k=ks[_], alpha=alpha, final=1, correction=correction)
                 for _ in range(self.calT)])

        if model_type=='full':
            self.encoders_A = nn.ModuleList([Encoder(
                in_channels=N, k=ks[_],
                r_prime=r_prime, r=r, h=h, alpha=alpha, dropout=dropout, model_type=self.configs['type'], correction=correction)
                for _ in range(self.calT)])
            self.decoders_A = nn.ModuleList([Decoder(
                r_prime=r_prime, r=r, k=ks[_], alpha=alpha, dropout=dropout, final=M, model_type=self.configs['type'], correction=correction)
                for _ in range(self.calT)])   
            
            # separating edge matrix and expression matrix latent spaces
            self.encoders_X = nn.ModuleList([Encoder(
                in_channels=N, k=ks[_],
                r_prime=r_prime, r=r, h=h, alpha=alpha, dropout=dropout, model_type=self.configs['type'], correction=correction) for _ in range(self.calT)]) 
            
            self.decoders_X = nn.ModuleList([Decoder(
                r_prime=r_prime, r=r, k=ks[_], alpha=alpha, dropout=dropout, final=M, model_type=self.configs['type'], correction=correction)
                for _ in range(self.calT)])    

            self.C1_full = nn.Linear(in_features=r * self.calT, out_features=r)
            self.C2_full = nn.Linear(in_features=r + len(ks[0]), out_features=r_prime)
     
        elif model_type=='single':
            self.encoders_A = nn.ModuleList([Encoder(
                in_channels=N, k=ks[0],
                r_prime=r_prime, r=r, h=h, alpha=alpha, dropout=dropout, model_type=self.configs['type'], correction=correction)])
            self.decoders_A = nn.ModuleList([Decoder(
                r_prime=r_prime, r=r, k=ks[0], alpha=alpha, dropout=dropout, final=M, model_type=self.configs['type'], correction=correction)])
            
            self.encoders_X = nn.ModuleList([Encoder(
                in_channels=N, k=ks[0],
                r_prime=r_prime, r=r, h=h, alpha=alpha, dropout=dropout, model_type=self.configs['type'], correction=correction)])

            self.C1_single = nn.Linear(in_features=r + len(ks[0]), out_features=r_prime)
            self.C2_single = nn.Linear(in_features=r_prime, out_features=r_prime)

        elif model_type=='simple':
            # we can assume that calT=1
            self.encoders_simple = nn.ModuleList([Encoder(
                in_channels=N, k=ks[0],
                r_prime=r_prime, r=r, h=h, alpha=alpha, dropout=dropout, model_type=self.configs['type'], correction=correction)])
            self.decoders_simple = nn.ModuleList([Decoder(
                 r_prime=r_prime, r=r, k=ks[0], alpha=alpha, dropout=dropout, final=r_prime, model_type=self.configs['type'], correction=correction
            )])

        self.pi_layer = nn.Linear(in_features=r_prime, out_features=N)
        self.omega_layer = nn.Linear(in_features=r_prime, out_features=N)
        self.theta_layer = nn.Linear(in_features=r_prime, out_features=N)

        self.dropout = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        self.apply(self.init_weights_vae)

        if self.configs['use_pretrain']:
            self.load_pretrained_model()

    def init_weights_vae(self, m):
        r"""
        Initialize weights of Linear layers using Xavier initialization

        :param m: A PyTorch module instance.
        :type m: nn.Module

        :return: None
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def reparameterize(self, mu, logvar):
        r"""
        Reparameterization method to sample from a Gaussian distribution

        :param mu: Mean of the Gaussian distribution.
        :type mu: torch.Tensor
        :param logvar: Natural log of the variance of the Gaussian distribution.
        :type logvar: torch.Tensor

        :return: Sampled latent variable `z`.
        :rtype: torch.Tensor
        """
        std = torch.exp(0.5 * logvar)
        eps1 = torch.randn_like(std)
        z = mu + eps1 * std
        return z

    def encode_reparameterization(self, Xs, Ks, ks, edges):
        r"""
        Encodes and reparameterizes the input data to produce latent structures 
        for the library size (Z_L) and network structure of sample-sample interactions (Z_A).

        :param Xs: List of zeroed expression matrices, one for each mini-batch.
        :type Xs: List[torch.Tensor]
        :param Ks: List of one-hot encoded matrices corresponding to batch or other categorical variables, one for each mini-batch.
        :type Ks: List[torch.Tensor]
        :param ks: List of integers representing the number of categories for each correction variable.
        :type ks: List[int]
        :param edges: List of adjacency matrices or edge lists representing sample-sample interactions, one for each mini-batch.
        :type edges: List[torch.Tensor]

        :return: A dictionary containing two sub-dictionaries:
                - `latent_spaces`: Contains the mean (`mu`) and log variance (`logvar`) for the library size, network interactions, expression matrix, and simple models for each mini-batch.
                - `latent_variables`: Contains the reparameterized latent variables `Z_L`, `Z_A`, `Z_X`, and `Z_simple` for each mini-batch.
        :rtype: Dict[str, Dict[str, List[torch.Tensor]]]
        """

        latent_variables = {'Z_Ls': [], 'Z_As': [],  'Z_Xs': [], 'Z_simples': []}
        latent_spaces = {
            'mu_L': [], 'logvar_L': [], 'lib_normal': [],
            'mu_X': [], 'logvar_X': [],
            'mu_A': [], 'logvar_A': [],
            'mu_simple': [], 'logvar_simple' : []
        }

        cuda_device_num = self.configs['cuda_device_num']
        device = torch.device(f'cuda:{cuda_device_num}' if torch.cuda.is_available() and not self.configs["no_cuda"] else 'cpu')
        cv, scv = self.configs['clamp_value'], self.configs['small_clamp_value']
        correction = self.configs['correct_vars']

        for i in range(self.calT):
            # embedding adjustment variables (an embedding for library and expression separately)            

            if self.configs['infer_lib_size'] is True:
                mu_L, logvar_L = self.encoders_L[i].encode(
                    torch.log(torch.sum(Xs[i], dim=1) + 1), Ks[i], ks[i], correction=correction)  
                                   
                logvar_L = torch.clamp(logvar_L, min=-scv, max=scv)
                check_for_nans(logvar_L.exp(), "decode logvar_L")

                latent_spaces['mu_L'].append(mu_L)
                latent_spaces['logvar_L'].append(logvar_L)
                latent_variables['Z_Ls'].append(self.reparameterize(mu_L, logvar_L))

            # end of infer library size

            if self.configs['type'] in ('full', 'single'):

                mu_A, logvar_A = self.encoders_A[i].encode(
                    X=Xs[i],
                    ejs=edges[i].to(device),
                    K=Ks[i], k=ks[i],correction=correction)

                mu_A = torch.clamp(mu_A, -cv, cv)
                logvar_A = torch.clamp(logvar_A, -scv, scv)

                latent_spaces['mu_A'].append(mu_A)
                latent_spaces['logvar_A'].append(logvar_A)
                latent_variables['Z_As'].append(self.reparameterize(mu_A, logvar_A))

                mu_X, logvar_X = self.encoders_X[i].encode(
                    X=Xs[i],
                    ejs=edges[i].to(device),
                    K=Ks[i], k=ks[i],correction=correction)

                mu_X = torch.clamp(mu_X, -cv, cv)
                logvar_X = torch.clamp(logvar_X, -scv, scv)

                latent_spaces['mu_X'].append(mu_X)
                latent_spaces['logvar_X'].append(logvar_X)
                latent_variables['Z_Xs'].append(self.reparameterize(mu_X, logvar_X))

            
            elif self.configs['type']=='simple':
                mu_simple, logvar_simple = self.encoders_simple[i].encode(
                    X=Xs[i],
                    ejs=None,
                    K=Ks[i],
                    k=ks[i], correction=correction)

                mu_simple = torch.clamp(mu_simple, -cv, cv)
                logvar_simple = torch.clamp(logvar_simple, -scv, scv)

                latent_spaces['mu_simple'].append(mu_simple)
                latent_spaces['logvar_simple'].append(logvar_simple)
                latent_variables['Z_simples'].append(self.reparameterize(mu_simple, logvar_simple))
                
        return {
            'latent_spaces': latent_spaces,
            'latent_variables': latent_variables
        }

    def decode(self, latent_vars, Ks, ks, edges):
        r"""
        Decodes the latent variables and combines them with correction variables 
        to calculate ZINB parameters for each type of decoder configuration.

        :param latent_vars: Dictionary containing reparameterized latent spaces computed by the `encode_reparameterization()` method.
                        It includes keys like 'Z_Ls' for library sizes and 'Z_As' for sample-sample interactions.
        :type latent_vars: Dict[str, List[torch.Tensor]]
        :param Ks: List of one-hot encoded matrices corresponding to batch or other categorical variables, one for each mini-batch.
        :type Ks: List[torch.Tensor]
        :param ks: List of integers representing the number of categories for each correction variable.
        :type ks: List[int]
        :param edges: List of adjacency matrices or edge lists representing sample-sample interactions, one for each mini-batch.
        :type edges: List[torch.Tensor]

        :return: A dictionary containing:
                - `DLs`: List of decoded library sizes.
                - `DAs`: List of decoded sample-sample interactions.
                - `distributional_parameters`: Dictionary containing the ZINB distributional parameters (`pi`, `omega`, and `theta`).
        :rtype: Dict[str, Union[List[torch.Tensor], Dict[str, torch.Tensor]]]
        """
        cuda_device_num = self.configs['cuda_device_num']
        device = torch.device(f'cuda:{cuda_device_num}' if torch.cuda.is_available() else 'cpu')
        correction = self.configs['correct_vars']
        K_stack = torch.stack(Ks[0], dim=1) 

        DLs, DAs = [], []
        for i in range(self.calT):

            if self.configs['infer_lib_size'] is True:
                tmp_L = self.decoders_L[i].decode(latent_vars['Z_Ls'][i],  Ks[i], ks[i], correction=correction)
                DLs.append(tmp_L)
            # Edges get decoded here
            if self.configs['type'] in ('full', 'single'):
                tmp_A = self.decoders_A[i].decode(Z=latent_vars['Z_As'][i], ejs=edges[i].to(device), K=Ks[i], k=ks[i], correction=correction)
                DAs.append(tmp_A)
                
            elif self.configs['type']=='simple':
                tmp_S = self.decoders_simple[i].decode(latent_vars['Z_simples'][i], ejs=None, K=Ks[i], k=ks[i], correction=correction)
          
        if self.configs['type']=='full':
            #h = torch.cat(latent_vars['Z_As'], dim=1)

            h = torch.cat(latent_vars['Z_Xs'], dim=1)
            h = self.leaky_relu(self.C1_full(h))        

            # to handle addition of correction vars   
            h = torch.cat((h, K_stack), dim=1)

            h = self.leaky_relu(self.C2_full(h))


        elif self.configs['type']=='single':
            #h = torch.cat(latent_vars['Z_As'], dim=1)
            h = torch.cat(latent_vars['Z_Xs'], dim=1)

            # to handle addition of correction vars  
            
            h = torch.cat((h, K_stack), dim=1)
            h = self.dropout(self.leaky_relu(self.C1_single(h)))
            h = self.dropout(self.leaky_relu(self.C2_single(h)))

        elif self.configs['type']=='simple':
            h = self.leaky_relu(tmp_S)

        pi = self.pi_layer(h)
        omega = self.softmax(self.omega_layer(h))

        if self.configs['theta_transform']:
            theta = self.theta_layer(h)
        else:
            theta = self.softplus(self.theta_layer(h))

        res = {'DLs': DLs, 'DAs': DAs}
        res['distributional_parameters'] = {'pi': pi, 'omega': omega, 'theta': theta}
        return res

    # FORWARD FUNCTION
    def forward(self, batch): 
        r"""
        Processes a batch of data through the VAE, performing encoding,
        reparameterization, and decoding steps to generate the outputs used for model training or inference.

        :param batch: A dictionary containing tensors that represent different parts of the data batch. Expected keys are:
        
            - 'X_batches': Zeroed expression matrices of the minibatch.
            
            - 'R_batches': Raw expression matrices of the minibatch.
            
            - 'K_batches': Correction variables.
            
            - 'k_batches': Levels of correction variables.
            
            - 'idx_batches': Indices of samples in the minibatch.
            
            - 'ej_batches': Graph edges in each minibatch (used if the model includes graph data).
            
        :type batch: Dict[str, List[torch.Tensor]]

        :return: A dictionary containing various outputs from the forward pass of the model, including:
            
            - 'latent_spaces': The latent spaces derived from the encoder.
            
            - 'latent_variables': Reparameterized latent variables.
            
            - 'X_hat': Predicted data samples (e.g., reconstructed expression levels).
            
            - 'DAs': Decoded activations from the model.
            
            - 'DLs': Decoded library size factors.
            
            - 'lib_size_factors': Library size factors computed post-decoding.
            
            - 'px_dispersion': Dispersion parameters of the distribution.
            
            - 'px_omega': Mu parameters of the distribution.
            
            - 'distributional_parameters': Parameters such as pi, omega, theta used in the distribution.
            
        :rtype: Dict[str, Union[torch.distributions.Distribution, List[torch.Tensor], Dict[str, torch.Tensor]]]
        """

        Xs_batch = batch['X_batches']
        Ks_batch = batch['K_batches']
        ks_batch = batch['k_batches']
        edges= batch['ej_batches']

        lcv = self.configs['lib_clamp_value']
            
        encode_outputs = self.encode_reparameterization(Xs_batch, Ks_batch, ks_batch, edges)
        decode_outputs = self.decode(encode_outputs['latent_variables'], Ks_batch, ks_batch, edges)

        lib_size_factors = []
        for i in range(self.calT):
            if self.configs['infer_lib_size'] is True:
                sf_clamped = torch.clamp(decode_outputs['DLs'][i], min=-lcv, max=lcv)
                lib_size_factors.append(torch.exp(sf_clamped))
            else:  # infer lib size is false
                lib_size_factors.append(
                    Xs_batch[i].sum(dim=1).unsqueeze(1))

        check_for_nans(decode_outputs['distributional_parameters']['omega'], "forward omega: ")
        check_for_nans(lib_size_factors[0].t(), "forward library size factors[0]: ")
        check_for_nans(decode_outputs['distributional_parameters']['theta'], "forward theta: ")
        
        # optional: theta transformation
        if self.configs['theta_transform']:
            clamped_log_theta = torch.clamp(decode_outputs['distributional_parameters']['theta'], min=-lcv, max=+lcv)
            px_dispersion = torch.exp(clamped_log_theta)
        else:
            px_dispersion = decode_outputs['distributional_parameters']['theta']
    
    
        # the 0th element of Ks[0] always contains batch info if batch exists.
        # added: check to make sure batch exist
        if self.configs['dispersion']=='gene-batch' and self.configs['batch_present']:
            batch_one_hot = one_hot_encode(Ks_batch[0][0])
            batch_t =  torch.transpose(batch_one_hot, 0, 1).to(torch.float32)  
            tmp = linear(batch_t, torch.transpose(px_dispersion, 0, 1))
            
            # divide tmp by the number of samples in each batch 
            num_ones_in_onehot = torch.sum(batch_t == 1, dim=1)  # Sum one-hot to get number of samples per batch
            num_samples_per_batch = num_ones_in_onehot.unsqueeze(1)
            tmp = tmp / num_samples_per_batch

            # broadcast thetas
            gb_px_dispersion = torch.zeros(px_dispersion.shape).to(torch.device(self.configs['cuda_device_num']))
            for i in range(batch_one_hot.shape[0]):
                batch_index = torch.argmax(batch_one_hot[i, :])  
                gb_px_dispersion[i, :] = tmp[batch_index, :]  

            px_dispersion = gb_px_dispersion

        px_omega = decode_outputs['distributional_parameters']['omega']

        if self.configs['model_likelihood'] == 'ZINB':
            check_for_nans(decode_outputs['distributional_parameters']['pi'], "forward pi: ")
            X_hat = ZeroInflatedNegativeBinomial(
                mu=px_omega * lib_size_factors[0],
                theta=px_dispersion,
                zi_logits=decode_outputs['distributional_parameters']['pi'])
        elif self.configs['model_likelihood'] == 'NB': 
            X_hat = NegativeBinomial(
                mu=px_omega * lib_size_factors[0],
                theta=px_dispersion)

        generative_outputs = {
            'latent_spaces': encode_outputs['latent_spaces'],
            'latent_variables': encode_outputs['latent_variables'],
            'X_hat': X_hat,
            'DAs': decode_outputs['DAs'],
            'distributional_parameters': decode_outputs['distributional_parameters'],
            'DLs': decode_outputs['DLs'],
            'lib_size_factors': lib_size_factors,
            'px_dispersion' : px_dispersion,
            'px_omega': px_omega}

        return generative_outputs


    def load_pretrained_model(self):
        """
        Loads a pre-trained model state into this model instance. It loads the model's state 
        dictionary, updates the current model instance's parameters, and sets the model to evaluation mode. 

        :raises Exception: If there are any issues accessing the folder or loading the model file.
        """
        
        cuda_device_num = self.configs['cuda_device_num']
        device = torch.device(f'cuda:{cuda_device_num}' if torch.cuda.is_available() and not self.configs["no_cuda"] else 'cpu')

        try:
            check_folder_access(self.configs['pretrain_model_path'])
        except Exception as e:
            print(e)

        copy_file(os.path.join(self.configs['pretrain_model_path'], self.configs['output_path'], 'pretrain.pth'))

        state_dict = torch.load(self.configs['pretrain_model_path'], map_location=torch.device(device))
        model_state_dict = state_dict['model_state_dict']
        self.load_state_dict(model_state_dict)
        torch.set_printoptions(profile="full")
        self.eval()

        # Freeze/unfreeze layers based on specific conditions
        self.set_parameter_requires_grad()

        # get optimizer state as well
        self.optimizer_state_dict = state_dict['optimizer_state_dict']


    def set_parameter_requires_grad(self):
        """
        Sets the `requires_grad` to enable/disable the training of specific layers.

        Usage:
            This method is typically called after model initialization or loading a pre-trained model to prepare
            the model for fine-tuning or full training, depending on the experiment's requirements.
        """
        for name, param in self.named_parameters():
            if self.configs['type'] == 'simple':
                if any(keyword in name for keyword in ["omega", "theta", "pi_layer"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = True
            elif self.configs['type'] == 'single':
                # IS NOT YET IMPLEMENTED, so we'll make everything adjustable for now.
                param.requires_grad = True
            elif self.configs['type'] == 'full':
                # IS NOT YET IMPLEMENTED
                param.requires_grad = True
   
            


    # ------------------------  Functions related to computing Loss   ------------------------


    def remove_ghost_samples(
            self, adj_Rs_batch, idx_batch, dataset,
            adjusted_generative_outputs):
        """
        Removes ghost samples from tensors, distributions and lists of 
        vectors and matrices

        :param adj_Rs_batch: Expression tensor of the current minibatch of samples.
        :type adj_Rs_batch: torch.Tensor
        :param idx_batch: List of indices of samples in the minibatch.
        :type idx_batch: List[int]
        :param dataset: Dataset object created in `_data_loader.py`.
        :type dataset: Dataset
        :param adjusted_generative_outputs: Output dictionary from the VAE containing tensors, distributions, and lists of vectors and matrices.
        :type adjusted_generative_outputs: Dict[str, Union[torch.distributions.Distribution, List[torch.Tensor], Dict[str, torch.Tensor]]]

        :return: None
        """

        def mask_and_delete(target, idxs_master_order):
            """
            Filters the target object by removing elements at specified indices. This function
            can handle different types of objects like tensors, distributions, and lists of tensors.

            :param target: The object to filter. This can be a single tensor, a list of distributions, or a list of tensors.
            :type target: Union[torch.Tensor, List[torch.distributions.Distribution], List[torch.Tensor]]
            :param idxs_master_order: List of indices of the elements to be removed from the target.
                                    Each sublist should correspond to the batch or sub-batch indices in the target object.
            :type idxs_master_order: List[List[int]]

            :return: The modified target with elements at the specified indices removed.
                    The type of the returned object matches the input type (tensor, list of distributions, or list of tensors).
            :rtype: Union[torch.Tensor, List[torch.distributions.Distribution], List[torch.Tensor]]

            :raises ValueError: If the input target type is not supported.
            """

            # case where it's an expression matrix
            if isinstance(target, torch.Tensor):
                mask = torch.ones(target.shape[0], dtype=torch.bool)
                mask[idxs_master_order[0]] = False
                target = target[mask, :]
                return (target)

            # case where it's a distribution
            if isinstance(target[0], torch.distributions.normal.Normal):
                mask = [torch.ones(len(target[i].loc), dtype=torch.bool)
                        for i in range(self.calT)]
                for i in range(self.calT):
                    mask[i][idxs_master_order[i]] = False
                target = [Normal(
                    loc=target[i].loc[mask[i]], scale=target[i].scale[mask[i]])
                          for i in range(self.calT)]
                return (target)

            # case where it's a list of matrices
            if isinstance(target[0], torch.Tensor) and target[0].shape[1] > 1:
                mask = [torch.ones(target[i].size(0), dtype=torch.bool)
                        for i in range(self.calT)]
                for i in range(self.calT):
                    mask[i][idxs_master_order[i]] = False
                target = [target[i][mask[i], :]
                          for i in range(self.calT)]
                return (target)



            # otherwise assume its just a simple list of vectors
            mask = [torch.ones(target[i].size(0), dtype=torch.bool)
                    for i in range(self.calT)]
            for i in range(self.calT):
                mask[i][idxs_master_order[i]] = False
            target = [target[i][mask[i]]
                      for i in range(self.calT)]
            return (target)
        
        ###### end of mask_and_delete

        ghost_indices = [dataset.get_ghost_indices(i) for i in range(self.calT)]
        idx_batch = idx_batch[0].cpu().numpy()
        to_remove = [list(set(ghost_indices[i]) & set(idx_batch))
                     for i in range(self.calT)]
        #idxs_master_order = [[
        #    torch.tensor(np.where(idx_batch == g)[0][0]) for g in to_remove[i]]
        #    for i in range(self.calT)]
        idxs_master_order = [[
            (np.where(idx_batch == g)[0][0]) for g in to_remove[i]]
            for i in range(self.calT)]
        

        adjusted_generative_outputs['lib_size_factors'] =\
            mask_and_delete(
                adjusted_generative_outputs['lib_size_factors'],
                idxs_master_order)

        if self.configs['infer_lib_size'] is True:
            adjusted_generative_outputs['latent_spaces']['mu_L'] = \
                    mask_and_delete(
                        adjusted_generative_outputs['latent_spaces']['mu_L'],
                        idxs_master_order)
            adjusted_generative_outputs['latent_spaces']['logvar_L'] = \
                    mask_and_delete(
                        adjusted_generative_outputs['latent_spaces']['logvar_L'],
                        idxs_master_order)           
        
        adj_Rs_batch = mask_and_delete(adj_Rs_batch, idxs_master_order)

        adjusted_generative_outputs['px_omega'] = \
                    mask_and_delete(
                        adjusted_generative_outputs['px_omega'], idxs_master_order)
        
        # mu_A, logvar_A, mu_X and logvar_X also need ghost samples removed
        # adjusted_generative_outputs['latent_spaces']['mu_X'] = \
        #             mask_and_delete(
        #                 adjusted_generative_outputs['latent_spaces']['mu_X'], idxs_master_order)
        # adjusted_generative_outputs['latent_spaces']['logvar_X'] = \
        #             mask_and_delete(
        #                 adjusted_generative_outputs['latent_spaces']['logvar_X'], idxs_master_order)
        # adjusted_generative_outputs['latent_spaces']['mu_A'] = \
        #             mask_and_delete(
        #                 adjusted_generative_outputs['latent_spaces']['mu_A'], idxs_master_order)
        # adjusted_generative_outputs['latent_spaces']['logvar_A'] = \
        #             mask_and_delete(
        #                 adjusted_generative_outputs['latent_spaces']['logvar_A'], idxs_master_order)


        adjusted_generative_outputs['X_hat'].mu = \
                    mask_and_delete(
                        adjusted_generative_outputs['X_hat'].mu, idxs_master_order)
        adjusted_generative_outputs['X_hat'].theta = \
                    mask_and_delete(
                        adjusted_generative_outputs['X_hat'].theta, idxs_master_order)

        # logits only exists for ZINB
        if self.configs['model_likelihood'] == "ZINB":
            adjusted_generative_outputs['X_hat'].zi_logits = \
                    mask_and_delete(
                        adjusted_generative_outputs['X_hat'].zi_logits,
                        idxs_master_order)

        # passing indices to downstream filters
        return adjusted_generative_outputs, adj_Rs_batch

    # ------------------------------ End of Remove Ghost Samples  ------------

    #################
    def batch_centroid_loss(self, counts, Ks):
        """
        Computes a loss based on the Euclidean distance between centroids of each experimental batch in the latent space.

        :param counts: List of tensors containing the latent representations per batch. Each element corresponds to a
                   different tissue or condition and has shape [num_samples, latent_dim].
        :type counts: List[torch.Tensor]
        :param Ks: List of tensors where the first column indicates batch membership for each sample in `counts`.
                Each tensor corresponds to a different tissue or condition and has shape [num_samples, num_batches].
        :type Ks: List[torch.Tensor]

        :return: A tensor containing the mean of the upper triangular non-zero Euclidean distances between batch
                centroids for each tissue or condition. Each element in the tensor corresponds to the computed
                distance for one of the tissues or conditions.
        :rtype: torch.Tensor
        """
        batch_correction_terms = []
        for tissue in range(self.calT):
            current_corr = Ks[tissue]
            
            # batch information is set in the first column (if only batch provided, `current_corr` is a vector not a matrix)
            #if (current_corr.ndim == 1):
            #    current_batch = current_corr # if a vector, it is batch
            #else:
            #    current_batch = current_corr[:, 0] # if a matrix, get batch from first column
            current_batch = current_corr[0].cpu().squeeze()
            num_batches = len(np.unique(current_batch))  
        
            batch_centroids = []
            for batch_num in range(num_batches):
                #all_indices = torch.nonzero(current_one_hot[:, batch_num] == 1).squeeze()
                all_indices = torch.nonzero(current_batch == batch_num).squeeze()

                batch_samples = counts[tissue][all_indices]
                batch_centroids.append(batch_samples.mean(dim=0))

            # Compute pairwise distances between batch centroids
            centroids_tensor = torch.stack(batch_centroids)  # Shape: [num_batches, latent_dim]
            diff = centroids_tensor.unsqueeze(1) - centroids_tensor.unsqueeze(0)
            distances = torch.norm(diff, dim=2).triu(1)  # Upper triangular part excluding the diagonal
            batch_correction_terms.append(distances[distances.nonzero(as_tuple=True)].mean())

        return torch.stack(batch_correction_terms)


        # ------------------------------ End of Compute Batch Correction  --------

    def loss(self, Rs_batch, idx_batch, adj_batch, dataset, prefix, epoch, generative_outputs, losses, log):
        """
        Calculates and records various losses during training or validation.

        :param Rs_batch: Raw expression matrices for the current minibatch, where each tensor corresponds to a batch from a specific condition or tissue.
        :type Rs_batch: List[torch.Tensor]
        :param idx_batch: List of indices corresponding to samples in the current minibatch.
        :type idx_batch: List[int]
        :param adj_batch: Adjacency matrices for samples in the minibatch, applicable for models considering sample-sample interactions.
        :type adj_batch: List[torch.Tensor]
        :param dataset: Dataset object providing access to dataset properties and helper methods.
        :type dataset: FFPE_Dataset
        :param prefix: Indicates the phase of the model ('train' or 'val') during which the loss is being computed.
        :type prefix: str
        :param epoch: The current epoch number in the training/validation process.
        :type epoch: int
        :param generative_outputs: Outputs from the forward pass of the VAE model including latent variables and other intermediate data.
        :type generative_outputs: Dict[str, Any]
        :param losses: Dictionary to record and update the computed losses over training epochs.
        :type losses: Dict[str, float]
        :param log: Logger object for logging the computed losses.
        :type log: Logger

        :return: The average loss computed across different metrics for the current minibatch.
        :rtype: torch.Tensor
        """

        model_type = self.configs['type']

        cuda_device_num = self.configs['cuda_device_num']
        device = torch.device(f'cuda:{cuda_device_num}' if torch.cuda.is_available() and not self.configs["no_cuda"] else 'cpu')
        idx_batch = [idx.to('cpu') for idx in idx_batch]

        adj_generative_outputs = generative_outputs.copy()
        adj_Rs_batch = Rs_batch.copy()

        # setup Epoch Delay Variables
        delay_conditions = {
            'delay_kl_lib': self.configs['delay_kl_lib'],
            'delay_kl_As': self.configs['delay_kl_As'],
            'delay_kl_simple' : self.configs['delay_kl_simple'],
            'delay_recon_As': self.configs['delay_recon_As'],
            'delay_recon_lib': self.configs['delay_recon_lib'],
            'delay_recon_expr': self.configs['delay_recon_expr'],
            'delay_centroid_batch': self.configs['delay_centroid_batch']
        }

        # this makes "(delay_flags['delay_kl_lib'])" true if we haven't reached
        # its epoch; this way we don't need "epoch > x" everywhere
        delay_flags = {
            condition: epoch < delay for condition,
            delay in delay_conditions.items()}

        if self.calT > 1:
            # ghosts shouldn't be considered the loss functions (except its ok for
            # the adjacency graphs); only activated when multiple tissues are present
            adj_generative_outputs, adj_Rs_batch = self.remove_ghost_samples(
                        adj_Rs_batch, idx_batch, dataset, adj_generative_outputs)

        # D_KL LOSSES

        if self.configs['infer_lib_size'] is True and\
                self.configs['batch_centroid_loss'] is True and\
                delay_flags['delay_centroid_batch'] is False:
            
            isolated_tensors = [[tensor[idx_batch].unsqueeze(0) for tensor in K] for K in dataset.Ks]
            
            batch_centroid_penalty = self.batch_centroid_loss(
                            adj_generative_outputs['latent_variables']['Z_Ls'], isolated_tensors)

        else:
            batch_centroid_penalty = torch.zeros(self.calT, dtype=torch.float32)

        # Where Graph KL Divergence is computed
        def calculate_graph_D_kl(mu_values, logvar_values):
            """
            Calculates the Kullback-Leibler (KL) divergence for each pair of mean (mu) and log-variance (logvar)
            tensors, and returns a stacked tensor of KL divergence values. This term helps regularize the latent
            space by measuring how closely the approximate posterior matches the prior distribution.

            :param mu_values: A list of mean tensors, each representing the latent distribution mean for a particular sample.
            :type mu_values: List[torch.Tensor]
            :param logvar_values: A list of log-variance tensors, each representing the latent distribution log-variance for a particular sample.
            :type logvar_values: List[torch.Tensor]

            :return: A tensor containing the KL divergence values for each sample.
            :rtype: torch.Tensor
            """

            def kl_divergence(mu, logvar):
                # # # return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kls = [kl_divergence(mu, logvar) for mu, logvar in zip(
                                        mu_values, logvar_values)]

            return torch.stack(kls)

        # Libraries
        if self.configs['infer_lib_size'] is True and delay_flags['delay_kl_lib'] is False:
            ave_lib_kl = calculate_graph_D_kl(
                adj_generative_outputs['latent_spaces']['mu_L'],
                adj_generative_outputs['latent_spaces']['logvar_L'])
        else:
            ave_lib_kl = torch.zeros(
                self.calT,
                dtype=torch.float32)

        if model_type in ('full', 'single'):
            if delay_flags['delay_kl_As'] is False:   
                ave_As_kl = calculate_graph_D_kl(            
                    mu_values=adj_generative_outputs['latent_spaces']['mu_A'],
                    logvar_values=adj_generative_outputs[
                                'latent_spaces']['logvar_A'])
                
                # Testing: As (edges) and Xs (expression) being separate latent spaces
                ave_Xs_kl = calculate_graph_D_kl(            
                    mu_values=adj_generative_outputs['latent_spaces']['mu_X'],
                    logvar_values=adj_generative_outputs[
                                'latent_spaces']['logvar_X'])
            else:
                ave_As_kl = torch.zeros(self.calT,dtype=torch.float32)
                ave_Xs_kl = torch.zeros(self.calT,dtype=torch.float32)

        elif model_type=='simple':
            if delay_flags['delay_kl_simple'] is False:
                ave_simple_kl = calculate_graph_D_kl(
                    mu_values=adj_generative_outputs['latent_spaces']['mu_simple'],
                    logvar_values=adj_generative_outputs['latent_spaces']['logvar_simple'])                
            else:
                ave_simple_kl = torch.zeros(self.calT, dtype=torch.float32)
       
        
        # ---------- RECONSTRUCTION LOSSES ----------

        def calculate_library_mse(lib_pred, lib_orig):
            """
            Calculate the Mean Squared Error (MSE) loss between the predicted
            and true library sizes

            :param lib_pred: Predicted library sizes.
            :type lib_pred: torch.Tensor
            :param lib_orig: Target (actual) library sizes.
            :type lib_orig: torch.Tensor

            :return: Scalar tensor representing the MSE loss.
            :rtype: torch.Tensor
            """
            lib_loss_fn = nn.MSELoss()
            lib_pred = lib_pred.squeeze()  # to make torch shape [mini_batch_size]
            return lib_loss_fn(lib_pred.float(), lib_orig.float()).float()

        if self.configs['infer_lib_size'] is True and delay_flags['delay_recon_lib'] is False:
            ave_recon_loss_lib = [calculate_library_mse(
                torch.log(adj_generative_outputs['lib_size_factors'][i]),
                torch.log(adj_Rs_batch[i].sum(dim=1) + 1))
                    for i in range(self.calT)]
            ave_recon_loss_lib = torch.stack(ave_recon_loss_lib)
        else:
            ave_recon_loss_lib = torch.zeros(self.calT, dtype=torch.float32)

        def calculate_graph_loss(graphs_orig, graphs_hat, idx_batch, datset, all=True):
            """
            Calculates the binary cross-entropy loss between original and reconstructed graph adjacency matrices.

            :param graphs_orig: Original adjacency matrices, one for each graph in the minibatch.
            :type graphs_orig: List[torch.Tensor]
            :param graphs_hat: Reconstructed adjacency matrices, corresponding to `graphs_orig`.
            :type graphs_hat: List[torch.Tensor]
            :param idx_batch: Indices of the minibatch samples, used to filter and map entries when `all` is False.
            :type idx_batch: List[torch.Tensor]
            :param datset: Dataset object that may contain additional information or utilities such as ghost indices or mask indices for processing adjacency matrices.
            :type datset: Dataset
            :param all: If True, the loss is calculated across all elements of each graph. If False, the loss calculation is masked to consider only specific elements according to the dataset's specifications. Defaults to True.
            :type all: bool, optional

            :return: A tensor containing the computed binary cross-entropy loss for each graph in the minibatch. The losses are stacked into a single tensor.
            :rtype: torch.Tensor
            """

            # all must be true if masking is off or if lambda_edges is zero
            if self.configs['masking_strategy'] is None or self.configs['lambda_edges'] <= 0:
                all = True

            g_orig = graphs_orig.copy()
            g_hat = graphs_hat.copy()
            graph_losses = []
            
            g_orig = [g.to(device) for g in g_orig]
            g_hat = [g.to(device) for g in g_hat]
            
            if all:
                # code to add weight to the edges for the BCE calculation
                weights = torch.where(
                    g_orig[0] == 1,
                    self.configs['pos_edge_weight'],  # Weight for positive class
                    self.configs['neg_edge_weight']   # negative class gets zero consideration
                )

                graph_losses = [F.binary_cross_entropy(
                    G_h, G_orig.float(), reduction='sum', weight=weights)
                    for G_h, G_orig in zip(g_hat, g_orig)]

            else: 
                for i in range(self.calT):

                    # only return values from 'tmp' that are in mask index
                    masked_idx = datset.As_mask_indices[i].cpu().numpy()
                    row_id = idx_batch[i].cpu().numpy()
                    
                    if self.calT > 1:
                        # filter ghost indices [check this when testing full]
                        ghost_indices = dataset.get_ghost_indices(i)
                        filtered_rowID = [value for value in row_id
                                        if value not in ghost_indices]
                    else:
                        filtered_rowID = [value for value in row_id]

                    # the next 4 lines is a bit faster than above but still takes a while
                    filtered_rowID_set = set(filtered_rowID)

                    # Apply the set for membership checks
                    mask_row = np.array([item in filtered_rowID_set for item in masked_idx[0]])
                    mask_col = np.array([item in filtered_rowID_set for item in masked_idx[1]])

                    mask_both = mask_row & mask_col
                    
                    filtered_masked_idx = masked_idx[:, mask_both]
                    # Step 2: Translate global indices to local indices within the minibatch
                    # Map batch indices to range(0, len(batch_idx))
                    index_map = {idx: i for i, idx in enumerate(filtered_rowID)}
                    local_indices = np.array([np.vectorize(index_map.get)(filtered_masked_idx[i]) for i in range(2)])
                    
                    # Access the elements in g_orig and g_hat using the local indices
                    g_orig_mask = g_orig[i][local_indices[0], local_indices[1]]
                    g_hat_mask = g_hat[i][local_indices[0], local_indices[1]]

                    graph_loss = F.binary_cross_entropy(g_hat_mask, g_orig_mask.float()).mean()
                    graph_losses.append(graph_loss)


            return torch.stack(graph_losses)

        if model_type in ('full', 'single'):
            if delay_flags['delay_recon_As'] is False:
                ave_reconstruction_loss_DAs = calculate_graph_loss(
                    adj_batch, adj_generative_outputs['DAs'], idx_batch, dataset)
            else:
                ave_reconstruction_loss_DAs = torch.zeros(
                    self.calT, dtype=torch.float32)



        def calculate_expression_loss(X_hat, adj_R_batch):
            """
            Calculate the negative log-likelihood (NLL) between the predicted
            and true gene expressions
    
            :param X_hat: Decoded expression of the minibatch, represented as a list of tensors.
            :type X_hat: List[torch.Tensor]
            :param adj_R_batch: Expression of the current minibatch of samples.
            :type adj_R_batch: torch.Tensor

            :return: Tensor containing the NLL loss for each gene in each sample of the minibatch.
            :rtype: torch.Tensor
    
            .. note::
                - Ghost samples are eliminated from the calculation.
            """
            return X_hat.log_prob(adj_R_batch)

        def js_divergence(X):
            """
            Computes the Jensen–Shannon divergence between a given probability distribution X
            and a corresponding uniform distribution of the same length.

            :param X: Probability distribution to compare against the uniform distribution. It can be either a PyTorch tensor or a NumPy array.
            :type X: Union[torch.Tensor, np.ndarray]

            :return: The Jensen-Shannon divergence value between the input distribution and the uniform distribution.
            :rtype: float
            """
            if torch.is_tensor(X):
                X = X.clone().detach().cpu().numpy()
       
            n = len(X)
            uniform_distribution = np.full(n, 1 / n)
            m = 0.5 * (X + uniform_distribution)
            js_div = 0.5 * (entropy(X, m) + entropy(uniform_distribution, m))
            return js_div
        
        def mean_js_divergence(Y):
            """
            Computes the mean Jensen–Shannon (JS) divergence across multiple distributions.

            :param Y: A list of probability distributions. Each element of `Y` represents a single distribution and should be a NumPy array.
            :type Y: List[np.ndarray]

            :return: The mean JS divergence value across all distributions in `Y`.
            :rtype: float
            """
            js_divs = []
            for row in Y:
                js_div = js_divergence(row)
                js_divs.append(js_div)
            return np.mean(js_divs)

        if delay_flags['delay_recon_expr'] is False:

            ave_recon_loss_X = calculate_expression_loss(
                adj_generative_outputs['X_hat'],
                adj_Rs_batch[0])

            # rather than take the mean of the losses, we sum losses across all samples first
            # so we get M' summed losses, which then gets averaged
            ave_recon_loss_X = -1 * ave_recon_loss_X.sum(-1) 

            lsf = torch.sum(adj_Rs_batch[0], dim=1).view(-1, 1)
            original_omega = adj_Rs_batch[0]/lsf

            tmp2 = torch.tensor(dataset.anndatas_orig[0].layers['truth'][idx_batch[0],:]).to(torch.device(f'cuda:{cuda_device_num}' if torch.cuda.is_available() and not self.configs["no_cuda"] else 'cpu'))

            js_div_orig = mean_js_divergence(original_omega)
            js_div_learn= mean_js_divergence(adj_generative_outputs['px_omega'])
            js_flat = mean_js_divergence(torch.full_like(original_omega, 1/original_omega.shape[1]))

            mumu = adj_generative_outputs['px_omega'] * adj_generative_outputs['lib_size_factors'][0]

            log.info(f"--------------++------ LOSS INFORMATION ------------------++--------")
            log.info(f"X_hat mu:\n {adj_generative_outputs['X_hat'].mu[0:3, 0:3]}")
            log.info(f"lib size factors:\n {adj_generative_outputs['lib_size_factors'][0][0:3]}")
            log.info(f"omega:\n {adj_generative_outputs['px_omega'][0:3, 0:3]}")
            log.info(f"omega * lib_size:\n {mumu[0:3]}")
            log.info(f"Original Rs:\n {adj_Rs_batch[0][0:3, 0:3]}")
            log.info(f"Generative mu:\n {tmp2[0:3, 0:3]}")

            log.info(f"X_hat theta:\n {adj_generative_outputs['X_hat'].theta[0:3, 0:3]}")

            if self.configs['model_likelihood']=='ZINB':
                log.info(f"X_hat pi:\n {adj_generative_outputs['X_hat'].zi_probs()[0:3, 0:3]}")
                js_pi_learn = mean_js_divergence(adj_generative_outputs['X_hat'].zi_probs())
                log.info(f"JS-divergence pi learn {js_pi_learn}")

            log.info(f"JS-divergence omega orig {js_div_orig} learn {js_div_learn} flat{js_flat}")
            log.info(f"average loss when compared to Xs: {ave_recon_loss_X}")
            log.info(f"-------------------------------- END --------------------------------")
      
        
        else:
            ave_recon_loss_X = torch.zeros(1, dtype=torch.float32)

        # ---------- ADJUST LOSSES ----------       

        def adjust_loss(param, raw_kl):
            """
            Adjust losses by multiplying it with given scalars
    
            :param param: A list of scalar values used to adjust the loss values.
            :type param: List[float]
            :param raw_kl: A list of tensors representing loss values to be adjusted.
            :type raw_kl: List[torch.Tensor]

            :return: A stacked tensor containing the adjusted loss values.
            :rtype: torch.Tensor
            """
            adjusted_loss = []
            for tensor, scalar in zip(raw_kl, param):
                result = tensor * scalar
                adjusted_loss.append(result)
            return torch.stack(adjusted_loss)

        # we set ave_adj_Libs_kl to zero if "infer_lib_size" is false;
        ave_adj_Libs_kl = adjust_loss(
            self.configs['DL_KL_weight'], ave_lib_kl)
        ave_adj_recon_loss_lib = adjust_loss(
            self.configs['lib_recon_weight'], ave_recon_loss_lib)
        ave_adj_batch_correction = adjust_loss(
            self.configs['batch_centroid_weight'], batch_centroid_penalty)
        ave_adj_recon_loss_expr = self.configs['X_recon_weight'] \
            * ave_recon_loss_X
        
        if model_type in ('full', 'single'):
            ave_adj_As_kl = adjust_loss(
                self.configs['DA_KL_weight'], ave_As_kl)
            ave_adj_Xs_kl = adjust_loss(
                self.configs['X_KL_weight'], ave_Xs_kl)
            ave_adj_recon_loss_DAs = adjust_loss(
                self.configs['DA_recon_weight'], ave_reconstruction_loss_DAs)
        elif model_type=='simple':
            ave_adj_simple_kl = adjust_loss(
                self.configs['simple_KL_weight'], ave_simple_kl)

        # ---------- CALCULATE JOINT LOSS ----------
        weight_conditions = [
            (delay_flags['delay_kl_lib'],
             self.configs['DL_KL_weight']),
            (delay_flags['delay_kl_As'],
             self.configs['DA_KL_weight']),
            (delay_flags['delay_kl_simple'],
             self.configs['simple_KL_weight']),
            (delay_flags['delay_recon_As'],
             self.configs['DA_recon_weight']),
            (delay_flags['delay_recon_lib'],
             self.configs['lib_recon_weight']),
            (delay_flags['delay_centroid_batch'],
             self.configs['batch_centroid_weight']),
            (delay_flags['delay_recon_expr'],
             self.configs['X_recon_weight'])
        ]

        # reset how weights were handled
        total_weight = 0
        for condition, weights in weight_conditions:
            # not because condition is True is the delay is occurring
            if not condition:
                if isinstance(weights, (int, float)):
                    total_weight += weights
                else:
                    total_weight += sum(weights)    
        
        if model_type in ('full', 'single'):

            average_loss = torch.mean(ave_adj_Libs_kl.cpu().sum() +
                        ave_adj_As_kl.cpu().mean() + 
                        ave_adj_Xs_kl.cpu().mean() + 
                        ave_adj_recon_loss_DAs.cpu().sum() +
                        ave_adj_batch_correction.cpu().sum() +
                        ave_adj_recon_loss_expr.cpu()
                        ) / total_weight
            
        elif model_type=='simple':
            # # #  Mean of the Sum of Losses 
            average_loss = torch.mean(ave_adj_Libs_kl.cpu() +
                        ave_adj_simple_kl.cpu() + 
                        ave_adj_recon_loss_lib.cpu() +
                        ave_adj_batch_correction.cpu() +
                        ave_adj_recon_loss_expr.cpu()
                        ) / total_weight
            

        # SEND TO STATISTICS LOG AND LOG FILE

        losses[prefix + '_' + 'average_loss'].append(
            average_loss.item())
        losses[prefix + '_' + 'kl_lib'].append(
            np.mean(ave_adj_Libs_kl.detach().cpu().numpy()))
        losses[prefix + '_' + 'recon_lib'].append(
            np.mean(ave_adj_recon_loss_lib.detach().cpu().numpy()))
        losses[prefix + '_' + 'centroid_batch'].append(
            np.mean(ave_adj_batch_correction.detach().cpu().numpy()))
        losses[prefix + '_' + 'recon_expr'].append(
            # # # ave_adj_recon_loss_expr.item())
            ave_adj_recon_loss_expr.mean().item())
        
        if model_type in ('full', 'single'):
            losses[prefix + '_' + 'kl_edge'].append(
                np.mean(ave_adj_As_kl.detach().cpu().numpy()))
            losses[prefix + '_' + 'kl_expr'].append(
                np.mean(ave_adj_Xs_kl.detach().cpu().numpy()))
            losses[prefix + '_' + 'recon_da'].append(
                np.mean(ave_adj_recon_loss_DAs.detach().cpu().numpy()))
        elif model_type=='simple':
            losses[prefix + '_' + 'kl_simple'].append(
                np.mean(ave_adj_simple_kl.detach().cpu().numpy()))            
            
        # Determine the maximum width of each column
            
        if model_type in ('full', 'single'):
            data = [
                ['Loss Function', 'Loss Value(s)'],
                ["Adj KL Lib:", np.mean(ave_adj_Libs_kl.detach().cpu().numpy())],
                # # # ["Adj KL Gene:", ave_adj_As_kl.detach().cpu().numpy()],
                ["Adj KL Gene:", np.mean(ave_adj_As_kl.detach().cpu().numpy())],
                ["Adj KL X:", np.mean(ave_adj_Xs_kl.detach().cpu().numpy())],
                ["Adj Recon Lib:", ave_adj_recon_loss_lib.detach().cpu().numpy()],
                ["Adj Recon DA:", ave_adj_recon_loss_DAs.detach().cpu().numpy()],
                ["Adj Batch Correction:",
                ave_adj_batch_correction.detach().cpu().numpy()],
                ["Adj Recon Express:",
                "{:.3e}".format(ave_adj_recon_loss_expr.mean().item())]
            ]
        elif model_type=='simple':
            data = [
                ['Loss Function', 'Loss Value(s)'],
                ["Adj KL Lib:", np.mean(ave_adj_Libs_kl.detach().cpu().numpy())],
                # # # ["Adj KL simple:", ave_adj_simple_kl.detach().cpu().numpy()],
                ["Adj KL simple:", np.mean(ave_adj_simple_kl.detach().cpu().numpy())],
                ["Adj Recon Lib:", ave_adj_recon_loss_lib.detach().cpu().numpy()],
                ["Adj Batch Correction:",
                ave_adj_batch_correction.detach().cpu().numpy()],
                ["Adj Recon Express:",
                # # # "{:.3e}".format(ave_adj_recon_loss_expr.item())]
                "{:.3e}".format(ave_adj_recon_loss_expr.mean().item())]
            ]       
      
        print_loss_table(data, self.log)

        return average_loss
