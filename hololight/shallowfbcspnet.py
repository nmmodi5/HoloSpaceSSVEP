# Note: This code was adapted from braindecode ShallowFBCSP implementation
# See https://braindecode.org/ for information on authors/licensing

import logging
from multiprocessing import dummy
import pickle

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

import torch as th
from torch import nn
from torch.nn import init
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split

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


class FBCSPDataset( Dataset ):
    """
    ShallowFBCSPNet has a built-in training procedure that will only function with 
    Map Datasets, and requires all trial labels (as integer classes) preloaded
    in order to perform balanced batching
    """
    
    # An FBCSPDataset always has a preloaded set of class labels
    classes: List[ int ]

    def __init__( self ) -> None:
        self.classes = list()

    def __len__( self ) -> int:
        return len( self.classes )

    def __getitem__( self, idx: int ) -> Tuple[ th.tensor, int ]:  
        return ( self.get_trial( idx ), self.classes[ idx ] )

    # This interface exists for lazy-loading of the trial data at idx
    def get_trial( self, idx: int ) -> th.Tensor:
        raise NotImplementedError

class PreloadedFBCSPDataset( FBCSPDataset ):
    
    # A preloaded dataset just stores the data in a tensor
    data: th.Tensor

    def __init__( self, 
        data: npt.ArrayLike, 
        labels: Iterable[ int ] 
    ) -> None:
        super().__init__()
        self.data = th.tensor( data )
        self.classes = [ int( l ) for l in labels ]
        assert self.data.shape[0] == len( self.classes )

    def get_trial( self, idx: int ) -> th.Tensor:  
        return self.data[ idx, ... ]


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
    epoch_idx: int
    train_loss: float
    test_loss: float
    test_accuracy: float
    lr: float

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
    model: th.nn.Module

    def __init__( 
        self, 
        params: ShallowFBCSPParameters, 
        model_state: Optional[ Dict[ str, Any ] ] = None,
        device_str: str = 'cpu',
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
        
        self.model = nn.Sequential()
        self.model.add_module( "ensuredims", Ensure4d() )

        if params.split_first_layer:
            self.model.add_module( "dimshuffle", 
                Expression( _transpose_time_to_spat ) 
            )

            self.model.add_module( "conv_time", 
                nn.Conv2d(
                    1,
                    params.n_filters_time,
                    ( params.filter_time_length, 1 ),
                    stride = 1,
                    dtype = dtype
                ),
            )
            self.model.add_module( "conv_spat",
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
            self.model.add_module( "conv_time",
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
            self.model.add_module( "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, 
                    momentum = params.batch_norm_alpha, 
                    affine = True,
                    dtype = dtype
                ),
            )

        if params.conv_nonlin:
            self.model.add_module( "conv_nonlin", Expression( 
                nonlin_dict[ params.conv_nonlin ] 
            ) )

        self.model.add_module( "pool",
            pool_class(
                kernel_size = ( params.pool_time_length, 1 ),
                stride = ( params.pool_time_stride, 1 ),
            ),
        )

        if params.pool_nonlin:
            self.model.add_module( "pool_nonlin", Expression( 
                nonlin_dict[ params.pool_nonlin ] 
            ) )

        self.model.add_module( "drop", nn.Dropout( p = params.drop_prob ) )

        output = dummy_output( self.model, params.in_chans, params.input_time_length )
        n_out_time = output.shape[2]
        if params.cropped_training: n_out_time = int( n_out_time // 2 )

        self.model.add_module( "conv_classifier",
            nn.Conv2d(
                n_filters_conv,
                params.n_classes,
                ( n_out_time, 1 ),
                bias = True,
                dtype = dtype
            ),
        )

        self.model.add_module( "softmax", nn.LogSoftmax( dim = 1 ) )
        self.model.add_module( "squeeze", Expression( _squeeze_final_output ) )

        if params.cropped_training:
            to_dense_prediction_model( self.model )

        if model_state is None:
            # Initialization, xavier is same as in paper...
            init.xavier_uniform_( self.model.conv_time.weight, gain = 1 )

            # maybe no bias in case of no split layer and batch norm
            if params.split_first_layer or ( not self.batch_norm ):
                init.constant_( self.model.conv_time.bias, 0 )
            if params.split_first_layer:
                init.xavier_uniform_( self.model.conv_spat.weight, gain = 1 )
                if not params.batch_norm:
                    init.constant_( self.model.conv_spat.bias, 0 )
            if params.batch_norm:
                init.constant_( self.model.bnorm.weight, 1 )
                init.constant_( self.model.bnorm.bias, 0 )
            init.xavier_uniform_( self.model.conv_classifier.weight, gain = 1 )
            init.constant_( self.model.conv_classifier.bias, 0 )
        else:
            self.model.load_state_dict( model_state )

        device = th.device( device_str )
        self.model.to( device )


    def __repr__( self ) -> str:
        rep = super().__repr__()
        model_rep = f'Model: {self.model.__repr__()}'
        model_parameters = filter( lambda p: p.requires_grad, self.model.parameters() )
        params = sum( [ np.prod( p.size() ) for p in model_parameters ] )
        example_param = next( self.model.parameters() )
        device = example_param.device
        dtype = example_param.dtype
        param_rep = f'Model has {params} trainable parameters on device: {device}'
        if self.params.cropped_training:
            param_rep = param_rep + ' (cropped training)'
        stride_rep = f'When segmenting temporal windows -- use optimal temporal stride of {self.optimal_temporal_stride} samples'
        in_rep = f'Model input: ( batch x {self.params.in_chans} ch x ' + \
            f'{self.params.input_time_length} time points, {dtype=} )'
        out = dummy_output( self.model, self.params.in_chans, self.params.input_time_length )
        out_rep = f'Model output: ( batch x {out.shape[1]} classes x {out.shape[2]} crops, dtype={out.dtype} )'
        return '\n'.join( [ rep, model_rep, param_rep, stride_rep, in_rep, out_rep ] )

    @classmethod
    def from_checkpoint_file( cls, checkpoint_file: Path, **kwargs ) -> "ShallowFBCSPNet":
        with open( checkpoint_file, 'rb' ) as checkpoint_f:
            obj: ShallowFBCSPCheckpoint = pickle.load( checkpoint_f )
        cls( params = obj.params, model_state = obj.model_state, **kwargs )

    def save_checkpoint_file( self, checkpoint_file: Path ) -> None:
        with open( checkpoint_file, 'wb' ) as checkpoint_f:
            pickle.dump( ShallowFBCSPCheckpoint(
                params = self.params,
                model_state = self.model.state_dict()
            ), checkpoint_f )

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
                self.model, 
                self.params.in_chans, 
                self.params.input_time_length 
            )

            return output.shape[2]
        else:
            return self.params.input_time_length

    def train( 
        self, 
        train_data: Union[ Tuple[ npt.ArrayLike, Iterable[ int ] ], FBCSPDataset ], 
        test_data: Union[ Tuple[ npt.ArrayLike, Iterable[ int ] ], FBCSPDataset ],
        learning_rate: float = 0.0001,
        max_epochs: int = 30,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        progress: bool = False,
        epoch_callback: Optional[ Callable[ [ EpochInfo ], None ] ] = None
    ) -> List[ EpochInfo ]:
        """
        Input is ( batch (trial) x channel x time ) Standardized multichannel timeseries ( N(0,1) across window )
        """

        if not isinstance( train_data, FBCSPDataset ):
            train_data = PreloadedFBCSPDataset( *train_data )

        if not isinstance( test_data, FBCSPDataset ):
            test_data = PreloadedFBCSPDataset( *test_data )

        device = next( self.model.parameters() ).device

        loss_fn = nn.NLLLoss()
        optimizer = th.optim.AdamW( 
            self.model.parameters(), 
            lr = learning_rate, 
            weight_decay = weight_decay 
        )
            
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR( 
            optimizer, T_max = max_epochs / 1 
        )

        training_log: List[ EpochInfo ] = []
        epoch_itr = range( max_epochs )

        if progress:
            try:
                from tqdm.autonotebook import tqdm
                epoch_itr = tqdm( epoch_itr )
            except ImportError:
                logger.warn( 'Attempted to provide progress bar, tqdm not installed' )

        # Calculate weights for class balancing
        classes, counts = np.unique( train_data.classes, return_counts = True )
        weights = { cl.item(): 1.0 / co.item() for cl, co in zip( classes, counts ) }
        weights = [ weights[ lab ] for lab in train_data.classes ]

        for epoch_idx in epoch_itr:

            self.model.train()
            train_loss_batches = []
            for train_feats, train_labels in DataLoader(
                train_data,
                batch_size = batch_size, 
                sampler = WeightedRandomSampler( weights, len( train_data ), replacement = False ),
                pin_memory = True,
            ):
                pred: th.Tensor = self.model( train_feats.to( device ) )
                if self.params.cropped_training:
                    pred = pred.mean( axis = 2 )
                loss: th.Tensor = loss_fn( pred, train_labels.to( device ) )
                train_loss_batches.append( loss.cpu().item() )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            accuracy = 0
            test_loss_batches = []
            self.model.eval()
            with th.no_grad():
                for test_feats, test_labels in DataLoader(
                    test_data, 
                    batch_size = batch_size, 
                    pin_memory = True
                ):
                    output: th.Tensor = self.model( test_feats.to( device ) )
                    if self.params.cropped_training:
                        output = output.mean( axis = 2 )
                    loss: th.Tensor = loss_fn( output, test_labels.to( device ) )
                    test_loss_batches.append( loss.cpu().item() )
                    accuracy += ( output.argmax( axis = 1 ).cpu() == test_labels ).sum().item()

            training_log.append( 
                EpochInfo(
                    epoch_idx = epoch_idx,
                    train_loss = np.mean( train_loss_batches ),
                    test_loss = np.mean( test_loss_batches ),
                    test_accuracy = accuracy / len( test_data ),
                    lr = scheduler.get_last_lr()[0]
                ) 
            )

            if epoch_callback is not None:
                epoch_callback( training_log[-1] )

        self.model.cpu()
        return training_log

    def inference( self, data: npt.ArrayLike, probs: bool = True ) -> np.ndarray:
        """
        Input is ( batch (trial) x channel x time ) Standardized multichannel timeseries ( N(0,1) across window )
            * If data.shape == 2, data is assumed to be channel x time input, and will automatically be converted
            to a 1 x channel x time array for inferencing.

        Output is ( batch (trial) x class ) probabilities.
            If probs = False, output is log-probabilities
        """
        data_input = th.tensor( data )
        if len( data_input.shape ) == 2:
            # Assume we put in a single trial of data
            data_input = data_input[ None, ... ]

        # n_trials, n_ch, n_time = data_input.shape

        device = next( self.model.parameters() ).device

        self.model.eval()
        with th.no_grad():
            output: th.Tensor = self.model( data_input.to( device ) )
            if probs: output = output.exp()
            
            # If we did cropped training, we might get multiple time outputs
            # collapse across time dimension here
            output = output.mean( axis = 2 )

        output = output.cpu().numpy()

        return output
            

        



                
