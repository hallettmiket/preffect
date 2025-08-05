import torch
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
from torch.distributions import Normal
import shutil
import torch.nn.functional as F
import umap
import math
import numbers

from _error import ( PreffectError )

input_log = logging.getLogger('input')

def logging_tensor(log, tensor, msg):
    """
    Logs the details of a tensor or a list of tensors, including its size,
    dtype, minimum, maximum values, and checks for NaNs or infinities.

    :param log: The logging object used for logging the tensor details.
    :type log: logging.Logger
    :param tensor: The tensor or list of tensors to be logged.
    :type tensor: torch.Tensor or list[torch.Tensor]
    :param msg: The message to be printed before logging the tensor details.
    :type msg: str

    """
    log.info("\n" + msg)
    if isinstance(tensor, list):
        for i in range(len(tensor)):
            Tt = tensor[i]
            if not isinstance(Tt, torch.Tensor):
                # log.info("Not a tensor")
                return
            log.info(
                "Size: "
                + str(Tt.size())
                + " dtype: "
                + str(Tt.dtype)
                + " Min: "
                + str(torch.min(Tt))
                + "Max: "
                + str(torch.max(Tt))
            )

            log.info(
                "Exist NaNs? "
                + str(torch.any(torch.isnan(Tt)).item())
                + " Exist infinities? "
                + str(torch.any(torch.isinf(Tt)).item())
            )
            log.info(str(Tt))
    else:
        if not isinstance(tensor, torch.Tensor):
            # log.info("Not a tensor")
            return
        Tt = tensor
        log.info("Size: " + str(Tt.size()) + " dtype: " + str(Tt.dtype))
        if Tt.numel() > 0:
            log.info(" Min: " + str(torch.min(Tt)) + "Max: " + str(torch.max(Tt)))
            log.info(
                "Exist NaNs? "
                + str(torch.any(torch.isnan(Tt)).item())
                + " Exist infinities? "
                + str(torch.any(torch.isinf(Tt)).item())
            )
            log.info(str(Tt))


def logging_helper(dic, log):
    """
    Logs tensor values along with their corresponding keys from a dictionary.

    :param dic: The dictionary containing keys and tensor values to be logged.
    :type dic: dict
    :param log: Logger used for outputting tensor information.
    :type log: logging.Logger
    """
    for key, value in dic.items():
        logging_tensor(log, value, "\t ******  - " + str(key))
    
def multi_logging_tensor(things):
    """
    Logs details of multiple tensors from a list of lists, where each sublist contains parameters 
    intended for the `logging_tensor` function. This allows for batch logging of tensor details for efficient monitoring 
    of tensor states across different stages or components of a model.

    :param things: A list of tuples, where each tuple contains three elements:
        
        1. log (logging.Logger): The logger object used for logging.
        2. tensor (torch.Tensor or list[torch.Tensor]): The tensor(s) to be logged.
        3. msg (str): A descriptive message that precedes the tensor details in the log.
    :type things: list of tuple

    """
    if len(things)>0:
        for thing in things:
            logging_tensor(*thing)


def _indices_to_tensor(zero_indices, shape):
    """
    Helper function to convert indices to a tensor efficiently and avoid a warning

    :param zero_indices: An array of flat indices.
    :type zero_indices: numpy.ndarray
    :param shape: The shape of the original tensor.
    :type shape: tuple of int

    :return: A tensor containing the coordinate arrays corresponding to the flat indices.
    :rtype: torch.Tensor
    """
    zero_indices_2d = np.unravel_index(zero_indices, shape)
    all_indices_np = np.array(zero_indices_2d)
    return torch.from_numpy(all_indices_np)

def check_for_nans(obj, msg):
    """
    Checks for NaN values in a PyTorch tensor and raises an error if found.

    :param obj: The tensor to check for NaN values.
    :type obj: torch.Tensor
    :param msg: The error message to display if NaN values are found.
    :type msg: str

    :raises ValueError: If NaN values are found in the `obj` tensor.
    """
    if torch.isnan(obj).any():
        print(obj.shape)
        print(obj)
        nan_positions = torch.nonzero(torch.isnan(obj))
        print(f"NaN positions (Row, Column): \n{nan_positions}")
        raise ValueError(msg)


def target_specific_one_hot(df, cat_var, target_value):
    """
    One-hot encodes a specific column in a DataFrame for a target value. It provides a one-hot encoded representation
    where only the target value is active.

    :param df: The input DataFrame containing the categorical variable.
    :type df: pandas.DataFrame
    :param cat_var: The column name to be one-hot encoded.
    :type cat_var: str
    :param target_value: The specific category value in the column to encode as 1; all others are set to 0.
    :type target_value: int or str

    :return: A tensor with the one-hot encoded column where only the target value is 1, and all other category values are 0.
    :rtype: torch.Tensor
    """
    
    n_unique = df[cat_var].nunique()

    # loop ensures same order every time
    # cat_map is a dictionary that indicates which columns are which variable, e.g {"batch1": 0, "batch2": 1}
    unique_labels = sorted(set(df[cat_var]))  # Extract unique labels and sort them to maintain order between datasets
    cat_map = {label: idx for idx, label in enumerate(unique_labels)}


    label_indices = torch.tensor([cat_map[label] for label in df[cat_var]])

    # convert adjustment value to how it is converted (so if batch starts at 1, it will convert to 0)
    batch_adjust = cat_map[target_value]

    indices = torch.full((label_indices.size(0),), batch_adjust, dtype=torch.long)

    # Apply one-hot encoding
    one_hot_encoded_batch = F.one_hot(indices, num_classes=n_unique)

    return one_hot_encoded_batch

def adjusted_categorical_correction_variable(df, cat_var, target_value):
    """
    One-hot encodes a specific column in a DataFrame for a target value. It provides a one-hot encoded representation
    where only the target value is active.

    :param df: The input DataFrame containing the categorical variable.
    :type df: pandas.DataFrame
    :param cat_var: The column name of the categorical variable to be encoded.
    :type cat_var: str
    :param target_value: The specific category value to be assigned a fixed index.
    :type target_value: int or str

    :return: A tuple containing:
        
        - **indices** (torch.Tensor): The adjusted indices for the categorical variable.
        
        - **named_cat_map** (dict): A dictionary mapping the categorical variable name to a dictionary of category-to-index mappings.
        
        - **num_unique** (int): The number of unique categories in the categorical variable.
    :rtype: tuple
    """
    named_cat_map = {}
    #cat_map = {label: idx for idx, label in enumerate(set(df[cat_var]))} # this needs to be changed to the OG model's cat_map.
    # loop ensures same order every time
    # cat_map is a dictionary that indicates which columns are which variable, e.g {"batch1": 0, "batch2": 1}
    unique_labels = sorted(set(df[cat_var]))  # Extract unique labels and sort them to maintain order between datasets
    cat_map = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = torch.tensor([cat_map[label] for label in df[cat_var]])
    # convert adjustment value to how it is converted (so if batch starts at 1, it will convert to 0)
    named_cat_map[cat_var] = cat_map
    batch_adjust = cat_map[target_value]
    indices = torch.full((label_indices.size(0),), batch_adjust, dtype=torch.long)
    
    return indices, named_cat_map, len(unique_labels)

def categorical_correction_variable(df, cat_var):
    r"""
    Prepare a categorical variable for one-hot encoding.

    This function prepares a categorical variable in a DataFrame for one-hot encoding. It returns the indices, a mapping of
    categories to indices, and the number of unique categories.

    :param df: The input DataFrame containing the categorical variable.
    :type df: pandas.DataFrame
    :param cat_var: The column name of the categorical variable to be encoded.
    :type cat_var: str

    :return: A tuple containing:
        
        - **indices** (torch.Tensor): The indices for the categorical variable.
        
        - **named_cat_map** (dict): A dictionary mapping the categorical variable name to a dictionary of category-to-index mappings.
        
        - **num_unique** (int): The number of unique categories in the categorical variable.
    :rtype: tuple
    """
    named_cat_map = {}

    unique_labels = sorted(set(df[cat_var]))  # Extract unique labels and sort them to maintain order between datasets
    cat_map = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = torch.tensor([cat_map[label] for label in df[cat_var]])
    named_cat_map[cat_var] = cat_map
    
    return label_indices, named_cat_map, len(unique_labels)

def one_hot_encode(column_tensor):
    r"""
    Perform one-hot encoding on a tensor.

    This function takes a tensor representing a categorical variable and performs one-hot encoding on it. If the categorical
    variable is invariant (i.e., has only one unique value), it returns a tensor of ones with the same number of rows as the
    input tensor and one column. Otherwise, it applies one-hot encoding using PyTorch's `F.one_hot` function.

    :param column_tensor: The tensor representing the categorical variable to be one-hot encoded.
    :type column_tensor: torch.Tensor

    :return: The one-hot encoded tensor.
    :rtype: torch.Tensor
    """
    num_classes = column_tensor.max().item() + 1
    
    # needed conditional if categorical variable is invariant
    if num_classes == 1:
        one_hot_encoded = torch.ones(column_tensor.size(0), 1, device=column_tensor.device, dtype=torch.int64)
    else: 
        one_hot_encoded = F.one_hot(column_tensor, num_classes=num_classes)

    return one_hot_encoded

def selective_one_hot(df, cat_var):
    """
    One-hot encodes a specific column in a DataFrame and removes all other columns.

    Args:
        df (DataFrame): The input DataFrame.
        cat_var (str): The column name to be one-hot encoded.

    Returns:
        DataFrame: A DataFrame containing only the one-hot encoded column of cat_var.
    """
    named_cat_map = {}

    if cat_var in df.columns:
        n_unique = df[cat_var].nunique()

        # torch one-hot function expects integers, so I'm converting input to integers
        # cat_map is a dictionary that indicates which columns are which variable, e.g {"batch1": 0, "batch2": 1}
        unique_labels = sorted(set(df[cat_var]))  # Extract unique labels and sort them to maintain order between datasets
        cat_map = {label: idx for idx, label in enumerate(unique_labels)}


        label_indices = torch.tensor([cat_map[label] for label in df[cat_var]])

        # making a map of integer conversion, e.g. "batch_1" -> 0, "batch_2" -> 1.

        named_cat_map[cat_var] = cat_map

        onehot = F.one_hot(label_indices, num_classes=n_unique)
        #return pd.get_dummies(df[cat_var], prefix=cat_var)
        return onehot, named_cat_map
        
    else:
        raise ValueError(f"Column {cat_var} not found in DataFrame.")


def plot_progression_all(losses, epoch, x_dim=2, y_dim=4, override=False, file_path=None, draw_to_screen=False):
    """
    Plots the progression of multiple types of loss metrics during training and validation across epochs.

    :type losses: dict
    :param epoch: The current epoch number for which the losses are being plotted.
    :type epoch: int
    :param x_dim: The number of rows in the subplot grid. Default is 2.
    :type x_dim: int, optional
    :param y_dim: The number of columns in the subplot grid. Default is 4.
    :type y_dim: int, optional
    :param override: If True, overrides the default subplot grid dimensions and sets them based on the number of unique plots. Default is False.
    :type override: bool, optional
    :param file_path: The file path where the plot should be saved. If not specified, the plot is saved as 'loss_progression_epoch_{epoch}.pdf'.
    :type file_path: str, optional
    """

    if file_path is None:
        file_path = f"loss_progression_epoch_{epoch}.pdf"

    filtered_losses = {k: v for k, v in losses.items() if v and not all(item == 0 for item in v)}
    num_unique_plots = math.ceil(len(filtered_losses)/2)
    if not override:
        x_dim=1
        y_dim=num_unique_plots

    fig, axs = plt.subplots(x_dim, y_dim, figsize=(12, 8))
    if x_dim == 1 or y_dim == 1:
        axs = axs.reshape(x_dim, y_dim)  # This line changes axs to 2D array shape for uniform access later

    prefix, prefix2 = 'train', 'val'


    for i, (key, value) in enumerate(filtered_losses.items()):
        if 'train' in key:
            new_key = key.replace(prefix, prefix2)
            simple_key = key.replace(prefix+'_', '')

            # ignoring the first few losses as they can be abnormally high and skew plots
            filtered_losses[key] = [None if i < 5 else val for i, val in enumerate(filtered_losses[key])]
            filtered_losses[new_key] = [None if i < 5 else val for i, val in enumerate(filtered_losses[new_key])]

            x_loc, y_loc = divmod(i, y_dim)
            axs[x_loc, y_loc].plot(filtered_losses[key], marker="o", color="blue", label="Train")
            axs[x_loc, y_loc].plot(filtered_losses[new_key], marker="o", color="green", label="Validate")
            axs[x_loc, y_loc].set_title(simple_key)
            axs[x_loc, y_loc].legend()
            if (draw_to_screen):
                axs[x_loc, y_loc].set_xlabel(f"Num. minibatches * Epoch")
            else:
                axs[x_loc, y_loc].set_xlabel(f"Num minibatches * Epoch (total epochs: {epoch})")
            axs[x_loc, y_loc].set_ylabel("Log Loss")
            axs[x_loc, y_loc].set_yscale("log")

    plt.tight_layout()
    if (draw_to_screen):
        plt.show()
    plt.suptitle(f"Loss functions: Epoch {epoch}", fontsize=16, y=1.05)
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free up memory


# this function no longer appears to be used
def plot_progression(target, title):
    """
    Plots the progression of average loss across mini-batches and epochs.

    :param target: A list or NumPy array containing average loss values at each mini-batch or epoch.
    :type target: list or numpy.ndarray
    :param title: The title to display on the plot.
    :type title: str
    """
    plt.plot(target, marker="o")
    plt.xlabel("Mini-Batch * Epoch")
    plt.ylabel("Average Loss")
    plt.title(title)
    plt.grid(True)
    plt.yscale("log")
    plt.show()


def print_loss_table(data, log):
    """
    Prints a well-formatted table for loss metrics or other numerical data.

    :param data: The data to be printed. Each sublist is considered a row in the table. The first row is
                 assumed to be the header. Cells can contain numerical values (int, float), NumPy arrays,
                 or strings.
    :type data: list of lists
    :param log: The logger object used for printing the table.
    :type log: logging.Logger
    """
    np.set_printoptions(formatter={"float": "{: .2e}".format})
    max_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]
    header = " | ".join(f"{cell:{width}}" for cell, width in zip(data[0], max_widths))
    log.info("")
    log.info(f"{header}")
    log.info(f"{'-' * len(header)}")

    for row in data[1:]:
        row_str = " | ".join(
            (
                f"{cell:.2e}"
                if isinstance(cell, (int, float))
                else (
                    f'{", ".join(f"{x:.2e}" for x in cell)}'
                    if isinstance(cell, np.ndarray)
                    else f"{cell:{width}}"
                )
            )
            for cell, width in zip(row, max_widths)
        )
        log.info(f"{row_str}")


def reparameterize_gaussian(mu, var):
    """
    Reparameterizes a Gaussian distribution and samples from it.

    :param mu: The mean of the Gaussian distribution.
    :type mu: torch.Tensor
    :param var: The variance of the Gaussian distribution.
    :type var: torch.Tensor

    :return: A sample from the Gaussian distribution parameterized by `mu` and `var`, with the same shape
             as `mu` and `var`.
    :rtype: torch.Tensor
    """
    return Normal(mu, var.sqrt()).rsample()




def ZINB_expected_value(mu, logits, distribution):
    """
    Calculate the expected value from a Zero-Inflated Negative Binomial (ZINB) or Negative Binomial (NB) distribution.

    :param mu: The mean of the negative binomial distribution.
    :type mu: torch.Tensor
    :param logits: The logits for the zero-inflation component.
    :type logits: torch.Tensor
    :param distribution: The type of distribution, either "ZINB" or "NB".
    :type distribution: str

    :return: The expected values computed from the NB/ZINB distribution.
    :rtype: torch.Tensor
    """
    if distribution == "ZINB":
        pi_matrix = 1 / (1 + torch.exp(-logits))
        if torch.any(pi_matrix > 1):
            print("PI MATRIX TOO LARGE")
    else:
        pi_matrix = 0

    # Calculate the expected value for each entry
    expected_value_matrix = (1 - pi_matrix) * mu

    return expected_value_matrix


def torch_mtx_unbatching(mtx, idx_batches, dataset, device):
    """
    Checks if list contains unique integers

    :param mtx: A list of mini-batched torch matrices.
    :type mtx: torch.Tensor
    :param idx_batches: A list of mini-batch indices, where each element is a tuple containing a tensor of indices.
    :type idx_batches: list of tuple of torch.Tensor
    :param dataset: The dataset object used to retrieve ghost indices.
    :type dataset: Dataset
    :param device: The device on which to perform the computations.
    :type device: torch.device

    :return: The rearranged matrix with rows sorted based on the indices.
    :rtype: torch.Tensor
    """
    # provide a list of mini-batched torch matrices and the list of mini-batch indices
    # this function will re-arrange them by row (default) or by column
    tensor_batches = [batch[0] for batch in idx_batches]
    idx_full = torch.cat(tensor_batches, dim=0).to(
        device
    )  # idx_batches is a list of vectors

    if dataset.calT > 1:
        ghost_indices = torch.tensor(dataset.get_ghost_indices(0))
        mask = torch.isin(idx_full, ghost_indices)
        idx_full = idx_full.clone().detach()[~mask].to(device)

    # Reorder the matrix rows by the sorted indices
    sorted_indices = torch.argsort(idx_full)
    mtx_full_rearr = mtx.index_select(0, sorted_indices)

    return mtx_full_rearr


def check_int_and_uniq(file_nums):
    """
    Checks if list contains unique integers.

    :param file_nums: A list of values from filenames after the phrase "tau".
    :type file_nums: list of int

    :raises ValueError: If any element in the list is not an integer or if integers are not unique.
    """
    if not all(isinstance(x, int) for x in file_nums):
        raise ValueError(
            "File names must be labelled with integers, e.g. tau_1, tau_2, etc."
        )

    if len(file_nums) != len(set(file_nums)):
        raise ValueError(
            "Multiple files in input folder with same tau value (e.g. two tau_1s). They must be unique."
        )


def calculate_and_sort_by_iqr(
    expr, gene_names=None, force_pam50=False, force_bc_genes=False
):
    """
    Function to calculate IQR in a table by column, then sort the table

    :param expr: A table of expression data where columns represent a gene and rows are samples.
    :type expr: numpy.ndarray
    :param gene_names: Name of genes (only needed if `force_pam50` is True).
    :type gene_names: list of str, optional
    :param force_pam50: Flag indicating whether to force the use of PAM50 genes (not used in the function).
    :type force_pam50: bool, optional
    :param force_bc_genes: Flag indicating whether to force the use of BC genes (not used in the function).
    :type force_bc_genes: bool, optional

    :return: Sorted indices for the columns of the input table based on IQR in descending order.
    :rtype: numpy.ndarray
    """

    # sub-function to calculate IQR; ignores NaNs
    # should it be ignored though? Or should they be considered the mean?
    def calculate_iqr(row):
        return np.nanpercentile(row, 75) - np.nanpercentile(row, 25)

    expr_df = pd.DataFrame(expr)
    # Apply the IQR function across the columns
    iqr_values = expr_df.apply(calculate_iqr, axis=0)

    # Get indices of sorted columns based on IQR in descending order
    sorted_indices = iqr_values.argsort()[::-1]

    return sorted_indices

def To(dev, objects):
    """
    Transfers each object in the provided dictionary to the specified device using PyTorch's `.to()` method.

    :param dev: The target device to which the tensors or models should be moved. It can be a `torch.device` object
                or a string specifying the device (e.g., 'cpu', 'cuda').
    :type dev: torch.device or str
    :param objects: A dictionary where each key-value pair consists of a name (key) and a PyTorch tensor, model, or
                    a list of tensors/models (value) [[6]][[7]][[8]].
    :type objects: dict

    :return: A dictionary with the same keys as the input, but with each tensor or model moved to the specified device.
    :rtype: dict
    """
    # Initialize an empty dictionary to store the modified objects
    modified_objects = {}

    # Iterate over the key-value pairs in the objects dictionary
    for key, value in objects.items():
        # If the value is a list, apply the .to(dev) to each item in the list
        if isinstance(value, list):
            modified_objects[key] = [mini_obj.to(dev) if isinstance(mini_obj, torch.Tensor) else mini_obj for mini_obj in value]
        else:
            # If the value is a single object, apply .to(dev) directly
            modified_objects[key] = value.to(dev)

    return modified_objects


# Function to ensure directory exists and has the right permissions
def ensure_directory(directory_path):
    """
    Ensures that a directory exists at the specified path. If the directory does not exist, it is created with specific
    permissions. If it already exists, the function does nothing.

    :param directory_path: The filesystem path where the directory should exist.
    :type directory_path: str
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)  # exist_ok=True to avoid raising an error if the directory exists
        os.chmod(directory_path, 0o775)  # Sets the directory to be read, write, and execute by owner and group, and readable by others
    else: 
        pass


def check_folder_access(path):
    """
    Checks if the specified folder path exists, is accessible, and adheres to the required format.

    :param path: The filesystem path of the directory to check. This path should be absolute or relative to the current
                 working directory.
    :type path: str

    :raises FileNotFoundError: If the directory does not exist at the specified path.
    :raises PermissionError: If the directory exists but does not have read permissions enabled.
    :raises ValueError: If the directory path does not end with a '/'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")
    
    if not os.access(path, os.R_OK):
        raise PermissionError(f"The path {path} is not readable.")
        
    if not path.endswith('/'):
        raise ValueError(f"The path {path} must end with a '/'.")


def copy_file(source_path, destination_path):
    """
    Copies a file from the source path to the destination path.

    :param source_path: The path to the source file that needs to be copied.
    :type source_path: str
    :param destination_path: The path to the destination where the file should be copied.
    :type destination_path: str

    :raises IOError: If an error occurs during the file copying process, such as the source file not
                     existing, no permission to read the file, or issues with the destination path.
    :raises Exception: If any other unforeseen exception occurs during the copying process.
    """
    try:
        shutil.copy(source_path, destination_path)
        print(f"File copied successfully from {source_path} to {destination_path}")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def set_seeds(config_seed):
    """
    Sets the seed for generating random numbers to ensure reproducibility across various random number generators. 

    :param config_seed: The seed value to use for all random number generators. If `None`, no seed is set.
    :type config_seed: int, optional
    """
    if config_seed is None:
        return

    import random
    torch.manual_seed(config_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config_seed)
    random.seed(config_seed)
    np.random.seed(config_seed)


def sanity_check_on_configs(preffect_con=None, train_ds_con=None, valid_ds_con=None):
    """
    Performs a series of assertions on configuration variables to ensure they meet
    predefined criteria necessary for PREFFECT to function.

    :param preffect_con: The configuration dictionary for the PREFFECT system. Defaults to None.
    :type preffect_con: dict, optional
    :param train_ds_con: The configuration dictionary for the training dataset. Defaults to None.
    :type train_ds_con: dict, optional
    :param valid_ds_con: The configuration dictionary for the validation dataset. Defaults to None.
    :type valid_ds_con: dict, optional

    :raises PreffectE: If any of the configuration variables do not pass their respective checks, indicating
                       a critical mismatch in the expected environment setup.
    """

    if preffect_con is None or train_ds_con is None or valid_ds_con is None:
        raise PreffectError('Must pass to sanity check the configs for the parent Preffect object and the train and validation confi')

    if preffect_con['dispersion']=='gene-batch':
        assert preffect_con['batch_present'], "If you want to infer batch specific dispersion for genes, then batch must be included in meta-data for each tissue."

    if (preffect_con['vars_to_correct'] is None):
       preffect_con['vars_to_correct'] = []

    if len(preffect_con['vars_to_correct']) == 0:
        preffect_con['correct_vars'] = False

    if not preffect_con['correct_vars']:
        assert not preffect_con['adjust_vars'], "You can not adjust a variable that first hasn't be included as a correction variable during training."
    
    if preffect_con['task']=='train':
        assert not preffect_con['adjust_vars'], "You can not adjust a variable during training. It can only be used for correction."

    if not preffect_con['batch_present']:
       preffect_con['batch_centroid_loss'] = False

    # we expect r_prime to be >= r
    assert preffect_con['r_prime'] > preffect_con['r'], (
        "We expect r_prime to be larger than r"
    )

    if preffect_con['correct_vars']:
        correction_types = {x[1] for x in preffect_con['vars_to_correct']}
        correction_vars = {x[0] for x in preffect_con['vars_to_correct']}

        assert correction_types.issubset({'categorical','continuous'}), (
            "Incorrect type for correction variable. Must be either continuous or categorical.")
        
        # this variable must be a string or number, not a list or tensor
        assert isinstance(preffect_con['set_NA_to_unique_corr'], (str, numbers.Number)), "set_NA_to_unique_corr must be a string or number"


        if preffect_con['adjust_vars']:
            assert 'batch' in correction_vars, "If you are adjusting for batch, the batch has to first be included in training as a correction variable."

            # ensure "adjust_to_batch_level" is a number
            assert isinstance(preffect_con['adjust_to_batch_level'], (int, float)), "adjust_to_batch_level is not a number"
            assert preffect_con['adjust_to_batch_level'] >= 0, "'adjust_to_batch_level' is negative."

    assert preffect_con['model_likelihood'] in {'NB', 'ZINB'}, "Set model_likelihood ot either NB or ZINB."

    if preffect_con['select_samples'] != float('inf'):
        assert preffect_con['select_samples'] > 0, f"Number of selected samples must be positive. Currently {preffect_con['select_samples']}"

    if preffect_con['calT']==1 and not preffect_con['adj_exist']:
        preffect_con['type'] = 'simple'
    elif preffect_con['calT']==1 and preffect_con['adj_exist'] and preffect_con['type'] == 'full':
        preffect_con['type'] = 'single'        
    
    # simple and single models must have calT = 1 only
    if preffect_con['calT']>=1 and preffect_con['type'] != 'full':
        preffect_con['calT'] = 1   

    if not preffect_con['correct_vars']:
        preffect_con['dispersion']='gene-sample'
        preffect_con['vars_to_correct'] = []
        

    # make sure weights are positive numbers
    for key, weights in preffect_con.items():
        if key.endswith('_weight'):
            if isinstance(weights, list):
                assert all(isinstance(weight, (int, float)) and weight > 0 for weight in weights), f"{key} cshould be a positive number"
            else:
                assert isinstance(weights, (int, float)) and weights > 0, f"{key} should be a positive number"

    # Check consistency of user defined delay parameters (during loss calculations)
    delay_of_1_found = False
    for parameter, value in preffect_con.items():
        if parameter.startswith('delay_'):
            if value == 1:
                delay_of_1_found = True
                break  # Stop checking
    if not delay_of_1_found:
        raise ValueError("At least one of the 'delay_' parameters should have a value of 1.")

    if train_ds_con['N'] != valid_ds_con['N']:
        raise ValueError(f"Number of transcripts must be the same between training and validation datasets: {train_ds_con['N']}, {valid_ds_con['N']}")

    return


def update_composite_configs(configs):
    r"""
    Update the composite configuration dictionary with derived paths.

    This function takes a configuration dictionary `configs` and updates it with derived paths for logs,
    inference results, and output files. It ensures that the `output_path` ends with a forward slash ('/') and
    constructs the derived paths by joining the `output_path` with the respective subdirectories using `os.path.join`
    and `os.sep`.

    :param configs: The configuration dictionary to be updated. If None, the function returns None.
    :type configs: dict, optional

    :return: The updated configuration dictionary with derived paths, or None if the input `configs` is None.
    :rtype: dict, optional
    """
    if configs is None:
        return None
    if not configs['output_path'].endswith('/'):
        configs['output_path'] += '/'

    configs['log_path']        = os.path.join(configs['output_path'], 'logs') + os.sep             #    for logs
    configs['inference_path']  = os.path.join(configs['output_path'], 'inference') + os.sep       #    for results of inference under difference adjustments (e.g. adjust_to_batch_level)
    configs['results_path']    = os.path.join(configs['output_path'], 'results') + os.sep          #    for pdfs and csv files

    return configs


def umap_draw_latent(results_from_forward, batch_info):
    r"""
    Visualize the latent space using UMAP dimensionality reduction.

    This function takes the latent variables from the forward pass results and applies UMAP (Uniform Manifold
    Approximation and Projection) dimensionality reduction to visualize the latent space. It plots
    the reduced latent space in a 2D scatter plot, where each point represents a sample and is colored based
    on the corresponding batch information.

    :param results_from_forward: A dictionary containing the results from the forward pass, including the latent variables.
    :type results_from_forward: dict
    :param batch_info: An array containing the batch information for each sample.
    :type batch_info: numpy.ndarray
    """
    # to investigate batch adjustment to latent space - will remove
    reducer = umap.UMAP(metric="cosine", n_neighbors=30, random_state=42, n_jobs=1, verbose=False, low_memory=True)
    embedding = reducer.fit_transform(results_from_forward['latent_variables']['Z_simples'][0].cpu().detach().numpy())
                            
    embedding_df = pd.DataFrame(embedding, columns=["0", "1"])

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        embedding_df["0"].values, embedding_df["1"].values,
        c=batch_info, s=3
    )
    ax.minorticks_off()
    ax.set_title("Z_Simple")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(False)
    plt.show()
