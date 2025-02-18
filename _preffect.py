import os
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

from _utils import (
    plot_progression_all,
    logging_tensor,
    To,
    set_seeds,
    sanity_check_on_configs
)
from _data_loader import FFPE_dataset
from _model import VAE
from _error import ( PreffectError )

class Preffect:
    def __init__(self,  forward_log, existing_session=False, configs=None):
        """
        Initializes a new Preffect instance by either preparing a new session or 
        restoring from an existing session based on the `existing_session` flag.

        Args:
            forward_log (logging.Logger): Logger used for outputting information 
                during the setup and operational phases.
            existing_session (bool, optional): A flag to indicate whether to restore 
                an existing session. Defaults to False.
            configs (dict, optional): Configuration parameters used to set up the 
                session. This includes paths, model settings, and other operational
                parameters. If not provided, defaults are assumed or an error is raised.

        Raises:
            Exception: Exceptions can be raised depending on the failure modes in either 
                `prep_from_existing_session` or `prep_new_session`. This might include
                file not found, access violations, or configuration errors.
        """
        if existing_session:
            self.prep_from_existing_session(configs)
        else:
            self.prep_new_session(configs, forward_log)

    

    def prep_new_session(self, configs, forward_log):        
        """
        Prepares a new session for training by setting up datasets, model, optimizer, 
        and other configurations necessary for training and validation.

        Args:
            configs (dict): Configuration parameters including dataset paths, model settings, 
                            training parameters, and device settings.
            forward_log (logging.Logger): Logger used for recording operational messages 
                                        and errors during the setup process.

        Raises:
            Exception: Raises an exception if there are misconfigurations, file path errors, 
                    or other issues during the initialization of datasets or model components.
        """
        self.configs = configs.copy()
        set_seeds(configs['seed'])
        self.inference_dict = {}
        self.Pis = None
        
        self.train_dataset = FFPE_dataset(self.configs.copy(), learning_type = "train", parent = self)
        self.train_dataset.to(self.configs['cuda_device'])
        self.validation_dataset = FFPE_dataset(self.configs.copy(), learning_type ="validation", parent = self)
        self.validation_dataset.to(self.configs['cuda_device'])

        # set overall config "adj_exist" to True if adjacency matrix found during data loading
        if self.configs['adj_exist'] is False:
            if self.train_dataset.configs['adj_exist'] and self.validation_dataset.configs['adj_exist']:
                self.configs['adj_exist'] = True

        sanity_check_on_configs(
            preffect_con=self.configs, 
            train_ds_con=self.train_dataset.configs, 
            valid_ds_con=self.validation_dataset.configs) 

        self.model = VAE(N=self.train_dataset.N, M=self.configs['mini_batch_size'], 
                         ks=self.train_dataset.ks, 
                         log=forward_log,
                         configs=self.configs).to(self.configs['cuda_device'])



        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr = self.configs['lr'],
                                        weight_decay=self.configs['weight_decay']) 

        if self.configs['use_pretrain']:
            self.optimizer.load_state_dict = self.model.optimizer_state_dict
            
        self.batches = self.train_dataset.prep_batch_iterator()
        self.batches_val = self.validation_dataset.prep_batch_iterator()

        self.keys = ['train_average_loss', 'train_kl_lib', 'train_kl_edge', 'train_kl_expr', 'train_kl_simple',
                'train_recon_da',  'train_recon_lib', 'train_recon_expr', 'train_centroid_batch', 
                'val_average_loss', 'val_kl_lib', 'val_kl_edge', 'val_kl_expr', 'val_kl_sample', 'val_kl_simple',
                'val_recon_da',  'val_recon_lib', 'val_recon_expr', 'val_centroid_batch']
        self.losses = {key : [] for key in self.keys}



    def prep_from_existing_session(self, configs):
        """
        Restores the Preffect object from a saved session file.

        Args:
            configs (dict): Configuration dictionary containing necessary paths and 
                            settings, which should include keys for 'output_path' and 
                            'endogenous_file_name' to construct the file path.
            

        Raises:
            Exception: An error occurred while trying to restore the session from the 
                    file, which could be due to missing file, corrupted data, or 
                    incompatible configurations.
        """
        try:
            loaded_session = torch.load(os.path.join(configs['output_path'], configs['input_existing_session'] + '.pth'))
            self.configs = loaded_session.configs

            self.train_dataset = loaded_session.train_dataset
            self.train_dataset.to(self.configs['cuda_device'])
            self.validation_dataset = loaded_session.validation_dataset
            self.validation_dataset.to(self.configs['cuda_device'])
           
            self.model = loaded_session.model
            self.model.load_state_dict(loaded_session.configs['model_state_dict'].copy())
            self.model.to(self.configs['cuda_device'])
            self.optimizer = loaded_session.optimizer
            
            if loaded_session.configs['use_pretrain']:
                self.optimizer.load_state_dict(loaded_session.configs['optimizer_state_dict'].copy())

            self.batches = loaded_session.batches
            self.batches_val = loaded_session.batches_val
            self.keys = loaded_session.keys
            self.losses = loaded_session.losses
            self.Pis = loaded_session.Pis
            self.inference_dict = loaded_session.inference_dict

            self.configs.pop('model_state_dict', None)
            self.configs.pop('optimizer_state_dict', None)  # both of these objects are very big.

        except Exception as e:
            raise PreffectError(f'Failed to restore the training session: {str(e)}')

    def find_inference_in_register(self, ir_name):
        """
        Retrieves an inference object from the inference dictionary based on its name.

        This method searches the internal dictionary that stores inference objects and returns
        the object associated with the given name if it exists. If no such object exists, it returns None.

        Args:
            ir_name (str): The name of the inference object to retrieve.

        Returns:
            The inference object associated with `ir_name` if it exists in the dictionary; otherwise, None.
        """
        if ir_name in self.inference_dict:
            return self.inference_dict[ir_name]
        else:
            return None

    def save(self, fname=None):
        """
        Saves the FFPE_dataset and trained model to a file on disk.

        Args:
            fname (str, optional): The filename under which to save the model. If not specified,
                                the model is saved under the 'nick_name' specified in the
                                configuration concatenated with '.pth'. The file is saved
                                in the directory specified by 'output_path' in the configuration.
        """

        if fname is None:
            fname = self.configs['input_existing_session']

        if self.configs['save_training_session']:
            self.configs['model_state_dict'] = self.model.state_dict()
            self.configs['optimizer_state_dict'] = self.optimizer.state_dict()
            fname = os.path.join(self.configs['output_path'], fname + ".pth")
            torch.save(self, fname)

    def extract_batch(self, batches, idx):
        """
        Extracts a single batch from the given batches dictionary based on the specified index.

        Args:
            batches (dict): A dictionary where each key maps to a list or array of batched data.
            idx (int): The index of the data to extract from each batch list in the batches dictionary.

        Returns:
            dict: A dictionary containing the extracted data for each key in the input batches dictionary.
                The keys remain the same, but the values are the data items at the specified index.
        """
        batch = {}
        for key, value in batches.items():
            batch[key] = value[idx]
        return batch        

    def train(self, forward_log):
        """
        This function initializes the model, prepares data loaders for training and validation sets, 
        and performs  training of VAE model for FFPE dataset.

        Args:
            forward_log (Logger): From _logger_config.py

        Raises:
            AssertionError: If the configuration sanity checks fail from _config.py

        Returns:
            None. 
        """


        start_time = time.time()
        writer = SummaryWriter()
        for epoch in tqdm(range(self.configs['epochs']), desc="Training Progress"):
            train_loss_per_epoch = 0
            self.model.train()

            for batch_idx in range(len(self.batches['X_batches'])):

                batch = To(self.configs['cuda_device'], self.extract_batch(self.batches, batch_idx))

                logging_tensor(forward_log, batch['X_batches'], 'batch X for epoch: ' + str(epoch) + ' batch: ' + str(batch_idx))
                logging_tensor(forward_log, batch['idx_batches'], 'batch indices for epoch: ' + str(epoch) + ' batch: ' + str(batch_idx))

                results_from_forward = self.model(batch)

                if self.configs['task'] == 'inference':
                    Rs_batch_input = batch['R_batches']
                else:
                    Rs_batch_input = batch['X_batches']

                loss = self.model.loss( 
                                #Rs_batch=batch['R_batches'],
                                Rs_batch=Rs_batch_input,
                                idx_batch=batch['idx_batches'], 
                                adj_batch=batch['adj_batches'],
                                dataset=self.train_dataset, prefix="train", 
                                epoch = (epoch + 1),
                                generative_outputs=results_from_forward,
                                losses=self.losses,
                                log = forward_log
                            )

                self.optimizer.zero_grad()
                loss.backward()

                # Trimming testing
                if self.configs['gradient_clip']:
                    max_norm = 10
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                self.optimizer.step()

                train_loss_per_epoch += loss.item()
 
                if not batch_idx % self.configs['logging_interval']:
                    forward_log.info('\nTraining Epoch: %03d/%03d | Batch %04d/%04d' % (epoch+1, self.configs['epochs'], 
                                                                                        (batch_idx + 1), 
                                                                                        len(self.batches['X_batches'])))

            forward_log.info(f"\nAverage training loss per epoch {train_loss_per_epoch / len(self.batches['X_batches'])}")
            writer.add_scalar("Loss/train", train_loss_per_epoch / len(self.batches['X_batches']), epoch)

            # -------------------- VALIDATION --------------------
     
            val_loss_per_epoch = 0
            self.model.eval()
            with torch.set_grad_enabled(False): 
                for batch_idx in range(len(self.batches_val['X_batches'])):
                    batch = To(self.configs['cuda_device'], self.extract_batch(self.batches_val, batch_idx))

                    results_from_forward = self.model(batch)

                    if self.configs['task'] == 'inference':
                        Rs_batch_input = batch['R_batches']
                    else:
                        Rs_batch_input = batch['X_batches']

                    loss = self.model.loss(
                                    #Rs_batch=batch['R_batches'], 
                                    Rs_batch=Rs_batch_input, 
                                    idx_batch=batch['idx_batches'], 
                                    adj_batch=batch['adj_batches'],
                                    dataset=self.validation_dataset, prefix="val", 
                                    epoch = (epoch + 1),
                                    generative_outputs=results_from_forward,
                                    losses=self.losses,
                                    log = forward_log)
                    
                    val_loss_per_epoch += loss.item()

                    if not batch_idx % self.configs['logging_interval']:
                        forward_log.info('Validation Epoch: %03d/%03d | Batch %04d/%04d' % (epoch+1, self.configs['epochs'], (batch_idx + 1), len(batch['X_batches'])))

                # draw fewer progression plots (to avoid overloading VScode)
                if (epoch + 1) % 10 == 0:
                    plot_progression_all(losses=self.losses, epoch=epoch+1, file_path=os.path.join(self.configs['results_path'], "losses.pdf"), override=True)
            
            forward_log.info(f"\nAverage validation loss per epoch {val_loss_per_epoch / len(batch['X_batches'])}")
            writer.add_scalar("Loss/valid ", val_loss_per_epoch / len(batch['X_batches']), epoch )
            writer.flush()
            forward_log.info('\nTime elapsed: %.2f min' % ((time.time() - start_time)/60))
        forward_log.info('\nTotal Training Time: %.2f min' % ((time.time() - start_time)/60))
        writer.close()
