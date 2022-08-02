# Note: Some of this code was adapted from braindecode ShallowFBCSP implementation
# See https://braindecode.org/ for information on authors/licensing

import logging
import pickle

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

import torch as th
from torch import nn
from torch.nn import init
from torch.utils.data import (
    Dataset, 
    IterableDataset,
    DataLoader, 
    TensorDataset,
    Subset
)

from typing import Iterable, Optional, Union, Callable, Tuple, Dict, Any, List

logger = logging.getLogger( __name__ )

def square( x: th.Tensor ) -> th.Tensor:
    return x * x

def safe_log( x: th.Tensor, eps: float = 1e-6 ) -> th.Tensor:
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return th.log( th.clamp( x, min = eps ) )

class Expression( th.nn.Module ):
    """
    Compute given expression on forward pass.
    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    expression_fn: Callable

    def __init__( self, expression_fn ):
        super( Expression, self ).__init__()
        self.expression_fn = expression_fn

    def forward( self, *args ):
        return self.expression_fn( *args )

    def __repr__( self ):
        if hasattr( self.expression_fn, "func" ) and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str( self.expression_fn.kwargs ) # noqa
            )
        elif hasattr( self.expression_fn, "__name__" ):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr( self.expression_fn )
        return f'{ self.__class__.__name__ }(expression={ str( expression_str ) })'
    
class Ensure4d( nn.Module ):
    def forward( self, x: th.Tensor ):
        while( len( x.shape ) < 4 ):
            x = x.unsqueeze( -1 )
        return x

def to_dense_prediction_model( 
    model: nn.Module, 
    axis: Union[ int, Tuple[ int, int ] ] = ( 2, 3 ) 
) -> None:
    """
    Transform a sequential model with strides to a model that outputs
    dense predictions by removing the strides and instead inserting dilations.
    Modifies model in-place.
    Parameters
    ----------
    model: torch.nn.Module
        Model which modules will be modified
    axis: int or (int,int)
        Axis to transform (in terms of intermediate output axes)
        can either be 2, 3, or (2,3).
    Notes
    -----
    Does not yet work correctly for average pooling.
    Prior to version 0.1.7, there had been a bug that could move strides
    backwards one layer.
    """
    if not hasattr( axis, "__len__" ):
        axis = [ axis ]
    assert all( [ ax in [ 2, 3 ] for ax in axis ] ), "Only 2 and 3 allowed for axis"
    axis = np.array( axis ) - 2
    stride_so_far = np.array( [ 1, 1 ] )
    for module in model.modules():
        if hasattr( module, "dilation" ):
            assert module.dilation == 1 or ( module.dilation == ( 1, 1 ) ), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?"
            )
            new_dilation = [ 1, 1 ]
            for ax in axis:
                new_dilation[ ax ] = int( stride_so_far[ ax ] )
            module.dilation = tuple( new_dilation )
        if hasattr( module, "stride" ):
            if not hasattr( module.stride, "__len__" ):
                module.stride = ( module.stride, module.stride )
            stride_so_far *= np.array( module.stride )
            new_stride = list( module.stride )
            for ax in axis:
                new_stride[ ax ] = 1
            module.stride = tuple( new_stride )

def dummy_output( model: nn.Module, in_chans: int, input_window_samples: int ) -> th.Tensor:
    with th.no_grad():
        dummy_input = th.ones(
            1, in_chans, input_window_samples,
            dtype = next( model.parameters() ).dtype,
            device = next( model.parameters() ).device,
        )
        output: th.Tensor = model( dummy_input )
    return output

# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output( x: th.Tensor ) -> th.Tensor:
    assert x.size()[ 3 ] == 1
    x = x[ :, :, :, 0 ]
    return x

def _transpose_time_to_spat( x: th.Tensor ) -> th.Tensor:
    return x.permute( 0, 3, 2, 1 )

def ensure_dataset( dataset: Union[ Tuple[ npt.ArrayLike, Iterable[ int ] ], Dataset ] ) -> Dataset:
    if not isinstance( dataset, Dataset ):
        dataset = TensorDataset( th.tensor( dataset[0] ), th.tensor( dataset[1] ) )
    return dataset

def balance_dataset( 
    dataset: Union[ Tuple[ npt.ArrayLike, Iterable[ int ] ], Dataset ], 
    ratios: Optional[ Iterable[ float ] ] = None, 
    drop_extra: bool = True,
    shuffle: bool = True,
    generator: Optional[ np.random.Generator ] = None
) -> Union[ Dataset, Tuple[ Dataset, ... ] ]:
    """
    If no split ratios are defined, and drop_extra is True,
    this function balances dataset by class examples
    """

    if ratios is None: ratios = [ 1.0 ]
    ratios: List[ float ] = [ 0.0 ] + [ v for v in ratios ]
    if not np.allclose( np.sum( ratios ), 1.0 ):
        raise ValueError( "Values in ratios must sum to 1.0" )

    dataset = ensure_dataset( dataset )

    if isinstance( dataset, IterableDataset ):
        raise ValueError( "Cannot split iterable dataset" )

    labels = np.array( [ label for _, label in dataset ] )
    classes, counts = np.unique( labels, return_counts = True )
    min_class_count = np.min( counts )

    class_indices = [ np.where( labels == cl )[0] for cl in classes ]

    if shuffle:
        if generator is None:
            generator = np.random.default_rng()
        for index_arr in class_indices:
            generator.shuffle( index_arr )

    class_indices = np.array( [ arr[:min_class_count] for arr in class_indices ] )
    extra_indices = np.array( [ arr[min_class_count:] for arr in class_indices ] ).flatten()

    bounds = np.cumsum( ratios ) * min_class_count
    bounds[-1] = min_class_count # account for round-off error
    bounds = bounds.round().astype( int )
    bounds = [ slice( *bound ) for bound in zip( bounds[:-1], bounds[1:] ) ]

    group_membership = [ class_indices[ :, bound ].flatten() for bound in bounds ]

    if not drop_extra:
        group_membership[-1] = np.concatenate( [ group_membership[-1], extra_indices ] )

    outputs = [ Subset( dataset, np.sort( indices ) ) for indices in group_membership ]
    return tuple( outputs ) if len( outputs ) > 1 else outputs[0]


@dataclass
class ShallowFBCSPParameters:
    """
    Default parameters are fine-tuned for EEG classification of spectral modulation
    between 8 and 112 Hz on time-series multi-channel EEG data sampled at ~250 Hz
    Recommend training on 4 second windows (input_time_length = 1000 samples)

    If doing "cropped training", inferencing can happen on smaller temporal windows
    If not doing cropped training, inferencing must happen on same window size as training.
    """
    # IO Parameters
    in_chans: int
    n_classes: int

    # Training information
    input_time_length: int # samples
    single_precision: bool = False # If True, layers use float32 precision, else float64
    # Cropped training requires changes to the network upon construction (dilation)
    cropped_training: bool = False # If True, operate on half-sized compute windows

    # First step -- Temporal Convolution (think FIR filtering)
    n_filters_time: int = 40
    filter_time_length: int = 25 # samples (think FIR filter order)

    # Second step -- Common Spatial Pattern (spatial weighting (no conv))
    n_filters_spat: int = 40
    split_first_layer: bool = True # If False, smash temporal and spatial conv layers together

    # Third step -- Nonlinearities
    #   First Nonlinearity: Batch normalization centers data
    batch_norm: bool = True
    batch_norm_alpha: float = 0.1

    #   Second Nonlinearity: Conv Nonlinearity.  'square' extracts filter output power
    conv_nonlin: Optional[ str ] = 'square' # || safe_log; No nonlin if None

    # Fourth step - Temporal pooling.  Aggregates and decimates spectral power
    pool_time_length: int = 75 # samples (think low pass filter on spectral content)
    pool_time_stride: int = 15 # samples (think decimation of spectral content)
    pool_mode: str = 'mean' # || 'max'

    # Fifth Step - Pool Nonlinearity.  'safe_log' makes spectral power normally distributed
    pool_nonlin: Optional[ str ] = 'safe_log' # || square; No nonlin if None
    
    # Sixth Step - Dropout layer for training resilient network and convergence
    drop_prob: float = 0.5

    # Seventh step -- Dense layer to output class. No parameters
    # Eighth step -- LogSoftmax - Output to probabilities. No parameters

@dataclass
class EpochInfo:
    train_loss: float
    test_loss: float
    test_accuracy: float
    lr: float

@dataclass
class ShallowFBCSPTrainingParameters:
    learning_rate: float = 0.0001
    annealing_epochs: int = 30
    batch_size: int = 32
    weight_decay: float = 0.0
    pin_memory: bool = False
    loss_fn: nn.Module = field( default_factory = nn.NLLLoss )

@dataclass
class ShallowFBCSPCheckpoint:
    params: ShallowFBCSPParameters
    model_state: Dict[ str, Any ]

class ShallowFBCSPNet:
    """
    Shallow ConvNet model from [2]_.

    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    params: ShallowFBCSPParameters
    _model: th.nn.Module

    def __init__( 
        self, 
        params: ShallowFBCSPParameters, 
        model_state: Optional[ Dict[ str, Any ] ] = None,
        device: str = 'cpu'
    ) -> None:

        self.params = params

        dtype = th.float32 if params.single_precision else th.float64

        pool_class = dict( 
            max = nn.MaxPool2d, 
            mean = nn.AvgPool2d 
        )[ params.pool_mode ]

        nonlin_dict = dict( 
            square = square,
            safe_log = safe_log 
        )
        
        self._model = nn.Sequential()
        self._model.add_module( "ensuredims", Ensure4d() )

        if params.split_first_layer:
            self._model.add_module( "dimshuffle", 
                Expression( _transpose_time_to_spat ) 
            )

            self._model.add_module( "conv_time", 
                nn.Conv2d(
                    1,
                    params.n_filters_time,
                    ( params.filter_time_length, 1 ),
                    stride = 1,
                    dtype = dtype
                ),
            )
            self._model.add_module( "conv_spat",
                nn.Conv2d(
                    params.n_filters_time,
                    params.n_filters_spat,
                    ( 1, params.in_chans ),
                    stride = 1,
                    bias = not params.batch_norm,
                    dtype = dtype
                ),
            )
            n_filters_conv = params.n_filters_spat
        else:
            self._model.add_module( "conv_time",
                nn.Conv2d(
                    params.in_chans,
                    params.n_filters_time,
                    ( params.filter_time_length, 1 ),
                    stride = 1,
                    bias = not params.batch_norm,
                    dtype = dtype
                ),
            )
            n_filters_conv = params.n_filters_time

        if params.batch_norm:
            self._model.add_module( "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, 
                    momentum = params.batch_norm_alpha, 
                    affine = True,
                    dtype = dtype
                ),
            )

        if params.conv_nonlin:
            self._model.add_module( "conv_nonlin", Expression( 
                nonlin_dict[ params.conv_nonlin ] 
            ) )

        self._model.add_module( "pool",
            pool_class(
                kernel_size = ( params.pool_time_length, 1 ),
                stride = ( params.pool_time_stride, 1 ),
            ),
        )

        if params.pool_nonlin:
            self._model.add_module( "pool_nonlin", Expression( 
                nonlin_dict[ params.pool_nonlin ] 
            ) )

        self._model.add_module( "drop", nn.Dropout( p = params.drop_prob ) )

        output = dummy_output( self._model, params.in_chans, params.input_time_length )
        n_out_time = output.shape[2]
        if params.cropped_training: n_out_time = int( n_out_time // 2 )

        self._model.add_module( "conv_classifier",
            nn.Conv2d(
                n_filters_conv,
                params.n_classes,
                ( n_out_time, 1 ),
                bias = True,
                dtype = dtype
            ),
        )

        self._model.add_module( "softmax", nn.LogSoftmax( dim = 1 ) )
        self._model.add_module( "squeeze", Expression( _squeeze_final_output ) )

        if params.cropped_training:
            to_dense_prediction_model( self._model )

        if model_state is None: self.reset_model()
        else: self._model.load_state_dict( model_state )

        self._model.to( th.device( device ) )

    def reset_model( self ) -> None:
        """
        (Re)Initialize model weights to appropriate starting values
        """
        # Initialization, xavier is same as in paper...
        init.xavier_uniform_( self._model.conv_time.weight, gain = 1 )

        # maybe no bias in case of no split layer and batch norm
        if self.params.split_first_layer or ( not self.batch_norm ):
            init.constant_( self._model.conv_time.bias, 0 )
        if self.params.split_first_layer:
            init.xavier_uniform_( self._model.conv_spat.weight, gain = 1 )
            if not self.params.batch_norm:
                init.constant_( self._model.conv_spat.bias, 0 )
        if self.params.batch_norm:
            init.constant_( self._model.bnorm.weight, 1 )
            init.constant_( self._model.bnorm.bias, 0 )
        init.xavier_uniform_( self._model.conv_classifier.weight, gain = 1 )
        init.constant_( self._model.conv_classifier.bias, 0 )

    def __repr__( self ) -> str:
        rep = super().__repr__()
        model_rep = f'Model: {self._model.__repr__()}'
        model_parameters = filter( lambda p: p.requires_grad, self._model.parameters() )
        params = sum( [ np.prod( p.size() ) for p in model_parameters ] )
        example_param = next( self._model.parameters() )
        device = example_param.device
        dtype = example_param.dtype
        param_rep = f'Model has {params} trainable parameters on device: {device}'
        if self.params.cropped_training:
            param_rep = param_rep + ' (cropped training)'
        stride_rep = f'When segmenting temporal windows -- use optimal temporal stride of {self.optimal_temporal_stride} samples'
        in_rep = f'Model input: ( batch x {self.params.in_chans} ch x ' + \
            f'{self.params.input_time_length} time points, {dtype=} )'
        out = dummy_output( self._model, self.params.in_chans, self.params.input_time_length )
        out_rep = f'Model output: ( batch x {out.shape[1]} classes x {out.shape[2]} crops, dtype={out.dtype} )'
        return '\n'.join( [ rep, model_rep, param_rep, stride_rep, in_rep, out_rep ] )

    @classmethod
    def from_checkpoint_file( cls, checkpoint_file: Path, **kwargs ) -> "ShallowFBCSPNet":
        with open( checkpoint_file, 'rb' ) as checkpoint_f:
            checkpoint: ShallowFBCSPCheckpoint = pickle.load( checkpoint_f )
        return cls.from_checkpoint( checkpoint, **kwargs )
        
    @classmethod
    def from_checkpoint( cls, checkpoint: ShallowFBCSPCheckpoint, **kwargs ) -> "ShallowFBCSPNet":
        return cls( params = checkpoint.params, model_state = checkpoint.model_state, **kwargs )

    def save_checkpoint_file( self, checkpoint_file: Path ) -> None:
        with open( checkpoint_file, 'wb' ) as checkpoint_f:
            pickle.dump( self.checkpoint, checkpoint_f )

    @property
    def model( self ) -> nn.Module:
        return self._model

    @property
    def checkpoint( self ) -> ShallowFBCSPCheckpoint:
        return ShallowFBCSPCheckpoint(
            params = self.params,
            model_state = self._model.state_dict()
        )

    @property
    def min_input_time_len( self ) -> int:
        """ The minimum temporal input size required for the model """
        if self.params.cropped_training:
            return self.params.input_time_length - self.optimal_temporal_stride
        else: return self.params.input_time_length

    @property
    def optimal_temporal_stride( self ) -> int:
        """
        If performing cropped training, it can help to window data using the 
        "temporal receptive field" of the model.  This property returns the 
        size of the temporal receptive field in samples, which would be a helpful
        stride when windowing data for use in training.
        """
        if self.params.cropped_training:
            output = dummy_output( 
                self._model, 
                self.params.in_chans, 
                self.params.input_time_length 
            )

            return output.shape[2]
        else:
            return self.params.input_time_length

    def train( 
        self, 
        train_data: Union[ Tuple[ npt.ArrayLike, Iterable[ int ] ], Dataset ], 
        test_data: Union[ Tuple[ npt.ArrayLike, Iterable[ int ] ], Dataset, float ],
        training_params: Optional[ ShallowFBCSPTrainingParameters ] = None,
        max_epochs: int = 30,
        balance_datasets: bool = True,
        progress: bool = False,
        epoch_callback: Optional[ Callable[ [ EpochInfo ], None ] ] = None,
    ) -> List[ EpochInfo ]:
        """
        Input is ( batch (trial) x channel x time ) Standardized multichannel timeseries ( N(0,1) across window )
        """

        train_data = ensure_dataset( train_data )

        # Make sure test data is a dataset
        if isinstance( test_data, float ):
            ratios = ( 1.0 - test_data, test_data )
            train_data, test_data = balance_dataset( train_data, ratios, drop_extra = False )
        elif not isinstance( test_data, Dataset ):
            test_data = ensure_dataset( train_data )

        # Rebalance subsets
        if balance_datasets:
            train_data = balance_dataset( train_data )
            test_data = balance_dataset( test_data )

        if training_params is None:
            training_params = ShallowFBCSPTrainingParameters(
                annealing_epochs = max_epochs
            )

        training = ShallowFBCSPTraining(
            net = self,
            train = train_data,
            test = test_data,
            params = training_params
        )

        training_log: List[ EpochInfo ] = []
        epoch_itr = range( max_epochs )

        if progress:
            try:
                from tqdm.autonotebook import tqdm
                epoch_itr = tqdm( epoch_itr )
            except ImportError:
                logger.warn( 'Attempted to provide progress bar, tqdm not installed' )

        for _ in epoch_itr:
            training_log.append( training.run_epoch() )
            if epoch_callback is not None:
                epoch_callback( training_log[-1] )

        return training_log

    def inference( self, data: Union[ npt.ArrayLike, th.Tensor ], probs: bool = True ) -> np.ndarray:
        """
        Input is ( batch (trial) x channel x time ) Standardized multichannel timeseries ( N(0,1) across window )
            * If data.shape == 2, data is assumed to be channel x time input, and will automatically be converted
            to a 1 x channel x time array for inferencing.

        Output is ( batch (trial) x class ) probabilities.
            If probs = False, output is log-probabilities
        """
        if not isinstance( data, th.Tensor ):
            data = th.tensor( data, dtype = th.float32 if self.params.single_precision else th.float64 )
        if len( data.shape ) == 2:
            # Assume we put in a single "trial" of data
            data = data[ None, ... ]

        device = next( self._model.parameters() ).device

        self._model.eval()
        with th.no_grad():
            output: th.Tensor = self._model( data.to( device ) )
            if probs: output = output.exp()
            output = output.mean( axis = 2 )

        output = output.cpu().numpy()

        return output

    def confusion( self, dataset: Union[ Tuple[ npt.ArrayLike, Iterable[ int ] ], Dataset ] ) -> np.ndarray:
        dataset = ensure_dataset( dataset )

        test_feats, test_labels = next( iter( DataLoader( dataset, batch_size = len( dataset ) ) ) )
        decode = self.inference( test_feats ).argmax( axis = 1 )

        classes = np.unique( test_labels )
        confusion = np.zeros( ( len( classes ), len( classes ) ) )

        for true_idx, true_class in enumerate( classes ):
            class_trials = np.where( np.array( test_labels ) == true_class )[0]
            for pred_idx, pred_class in enumerate( classes ):
                num_preds: np.ndarray = ( decode[ class_trials ] == pred_class )
                confusion[ true_idx, pred_idx ] = num_preds.sum() / len( class_trials ) # .sum().item()
        
        return confusion    

@dataclass
class ShallowFBCSPTraining:
    net: ShallowFBCSPNet
    train: Dataset
    test: Dataset

    params: ShallowFBCSPTrainingParameters = field(
        default_factory = ShallowFBCSPTrainingParameters
    )

    device: th.device = field( init = False )
    optimizer: th.optim.Optimizer = field( init = False )
    scheduler: th.optim.lr_scheduler._LRScheduler = field( init = False )

    def __post_init__( self ):

        self.device = next( self.net.model.parameters() ).device

        self.optimizer = th.optim.AdamW( 
            self.net.model.parameters(), 
            lr = self.params.learning_rate, 
            weight_decay = self.params.weight_decay 
        )
            
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR( 
            self.optimizer, T_max = self.params.annealing_epochs / 1 
        )

    def run_epoch( self ) -> EpochInfo:

        self.net.model.train()
        train_loss_batches = []
        for train_feats, train_labels in DataLoader(
            self.train,
            batch_size = self.params.batch_size, 
            pin_memory = self.params.pin_memory,
        ):
            pred: th.Tensor = self.net.model( train_feats.to( self.device ) )
            pred = pred.mean( axis = 2 )
            loss: th.Tensor = self.params.loss_fn( pred, train_labels.to( self.device ) )
            train_loss_batches.append( loss.cpu().item() )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

        accuracy = 0
        test_loss_batches = []
        self.net.model.eval()
        with th.no_grad():
            for test_feats, test_labels in DataLoader(
                self.test, 
                batch_size = self.params.batch_size, 
                pin_memory = self.params.pin_memory
            ):
                output: th.Tensor = self.net.model( test_feats.to( self.device ) )
                output = output.mean( axis = 2 )
                loss: th.Tensor = self.params.loss_fn( output, test_labels.to( self.device ) )
                test_loss_batches.append( loss.cpu().item() )
                accuracy += ( output.argmax( axis = 1 ).cpu() == test_labels ).sum().item()

        info = EpochInfo(
            train_loss = np.mean( train_loss_batches ),
            test_loss = np.mean( test_loss_batches ),
            test_accuracy = accuracy / len( self.test ),
            lr = self.scheduler.get_last_lr()[0]
        ) 

        return info

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    def plot_train_info_mpl( train_info: List[ EpochInfo ], ax: Axes ) -> None:
        ax.plot( [ e.train_loss for e in train_info ], label = 'Train Loss' )
        ax.plot( [ e.test_loss for e in train_info ], label = 'Test Loss' )
        ax.plot( [ e.test_accuracy for e in train_info ], label = 'Test Accuracy' )
        ax.plot( [ e.lr for e in train_info ], label = 'Learning Rate' )
        ax.legend()
        ax.set_yscale( 'log' )
        ax.set_xlabel( 'Epoch' )
        ax.axhline( 1, color = 'k' )
        ax.set_title( 'Training Curves' )

    def plot_confusion_mpl( confusion: np.ndarray, ax: Axes, add_colorbar: bool = True ) -> None:
        n_classes = confusion.shape[0]

        corners = np.arange( n_classes + 1 ) - 0.5
        im = ax.pcolormesh( 
            corners, corners, confusion, alpha = 0.5,
            cmap = plt.cm.Blues, vmin = 0.0, vmax = 1.0
        )

        for row_idx, row in enumerate( confusion ):
            for col_idx, freq in enumerate( row ):
                ax.annotate( 
                    f'{freq:0.2f}', ( col_idx, row_idx ), 
                    ha = 'center', va = 'center' 
                )

        ax.set_aspect( 'equal' )
        ax.set_xticks( np.arange( n_classes ) )
        ax.set_yticks( np.arange( n_classes ) )
        ax.set_ylabel( 'True Class' )
        ax.set_xlabel( 'Predicted Class' )
        ax.invert_yaxis()
        if add_colorbar: 
            ax.get_figure().colorbar( im )
        ax.set_title( 'Classifier Confusion' )

except ImportError:
    pass     