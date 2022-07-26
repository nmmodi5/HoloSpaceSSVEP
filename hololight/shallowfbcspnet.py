# Note: This code was adapted from braindecode ShallowFBCSP implementation
# See https://braindecode.org/ for information on authors/licensing

import logging

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

import torch as th
from torch import nn
from torch.nn import init
# from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from typing import Optional, Union, Callable, Tuple, Dict, Any

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

class FBCSPDataset( th.utils.data.Dataset ):
    
    X: th.Tensor
    y: th.Tensor

    label_map: Dict[ Any, int ]

    def __init__( self, data: npt.ArrayLike, labels: npt.ArrayLike ) -> None:
        self.X = th.tensor( data )

        self.label_map = { 
            label: idx for idx, label in 
            enumerate( np.unique( labels ).tolist() ) 
        }

        self.y = th.tensor( [ self.label_map[ l ] for l in labels ] )
        assert self.X.shape[0] == self.y.shape[0]

    def __len__( self ) -> int:
        return self.X.shape[0]

    def __getitem__( self, idx: int ) -> Tuple[ th.tensor, int ]:  
        return ( self.X[ idx, ... ], self.y[ idx ] )


@dataclass
class ShallowFBCSPNet:

    """
    Shallow ConvNet model from [2]_.

    Input is ( batch x channel x time ) Standardized multichannel timeseries ( N(0,1) across window )
    Output is ( batch x class x time ) log-probabilities
        To get probabilities, recommend np.exp( output ).mean( dim = 2 ).squeeze()

    Default parameters are fine-tuned for EEG classification of spectral modulation
    between 8 and 112 Hz on time-series multi-channel EEG data sampled at ~250 Hz
    Recommend training on 4 second windows (input_time_length = 1000 samples)

    If doing "cropped training", inferencing can happen on smaller temporal windows
    If not doing cropped training, inferencing must happen on same window size as training.

    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    # IO Parameters
    in_chans: int
    n_classes: int

    # Training information
    input_time_length: int # samples
    cropped_training: bool = False

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

    device: str = 'cpu'
    _device: th.device = field( init = False )
    model: th.nn.Module = field( default_factory = nn.Sequential, init = False )

    single_precision: bool = False
    _dtype: th.dtype = field( init = False )

    optimizer: Optional[ th.optim.Optimizer ] = field( default = None, init = False )

    def __post_init__( self ) -> None:

        self._dtype = th.float32 if self.single_precision else th.float64

        pool_class = dict( 
            max = nn.MaxPool2d, 
            mean = nn.AvgPool2d 
        )[ self.pool_mode ]

        nonlin_dict = dict( 
            square = square,
            safe_log = safe_log 
        )
        
        self.model.add_module( "ensuredims", Ensure4d() )

        if self.split_first_layer:
            self.model.add_module( "dimshuffle", Expression( _transpose_time_to_spat ) )

            self.model.add_module( "conv_time", 
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    ( self.filter_time_length, 1 ),
                    stride = 1,
                    dtype = self._dtype
                ),
            )
            self.model.add_module( "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    ( 1, self.in_chans ),
                    stride = 1,
                    bias = not self.batch_norm,
                    dtype = self._dtype
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            self.model.add_module( "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    ( self.filter_time_length, 1 ),
                    stride = 1,
                    bias = not self.batch_norm,
                    dtype = self._dtype
                ),
            )
            n_filters_conv = self.n_filters_time

        if self.batch_norm:
            self.model.add_module( "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, 
                    momentum = self.batch_norm_alpha, 
                    affine = True,
                    dtype = self._dtype
                ),
            )

        if self.conv_nonlin:
            self.model.add_module( "conv_nonlin", Expression( 
                nonlin_dict[ self.conv_nonlin ] 
            ) )

        self.model.add_module( "pool",
            pool_class(
                kernel_size = ( self.pool_time_length, 1 ),
                stride = ( self.pool_time_stride, 1 ),
            ),
        )

        if self.pool_nonlin:
            self.model.add_module( "pool_nonlin", Expression( 
                nonlin_dict[ self.pool_nonlin ] 
            ) )

        self.model.add_module( "drop", nn.Dropout( p = self.drop_prob ) )

        output = dummy_output( self.model, self.in_chans, self.input_time_length )
        n_out_time = output.shape[2]
        if self.cropped_training: n_out_time = int( n_out_time // 2 )

        self.model.add_module( "conv_classifier",
            nn.Conv2d(
                n_filters_conv,
                self.n_classes,
                ( n_out_time, 1 ),
                bias = True,
                dtype = self._dtype
            ),
        )

        self.model.add_module( "softmax", nn.LogSoftmax( dim = 1 ) )
        self.model.add_module( "squeeze", Expression( _squeeze_final_output ) )

        # Initialization, xavier is same as in paper...
        init.xavier_uniform_( self.model.conv_time.weight, gain = 1 )

        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or ( not self.batch_norm ):
            init.constant_( self.model.conv_time.bias, 0 )
        if self.split_first_layer:
            init.xavier_uniform_( self.model.conv_spat.weight, gain = 1 )
            if not self.batch_norm:
                init.constant_( self.model.conv_spat.bias, 0 )
        if self.batch_norm:
            init.constant_( self.model.bnorm.weight, 1 )
            init.constant_( self.model.bnorm.bias, 0 )
        init.xavier_uniform_( self.model.conv_classifier.weight, gain = 1 )
        init.constant_( self.model.conv_classifier.bias, 0 )

        if self.cropped_training:
            to_dense_prediction_model( self.model )

        self._device = th.device( self.device )
        self.model.to( self._device )

    @classmethod
    def from_checkpoint( cls, checkpoint: Path ) -> "ShallowFBCSPNet":
        ...

    def save_checkpoint( self, checkpoint: Path ) -> None:
        ...

    @property
    def optimal_temporal_stride( self ) -> int:
        if self.cropped_training:
            output = dummy_output( 
                self.model, 
                self.in_chans, 
                self.input_time_length 
            )

            return output.shape[2]
        else:
            return self.input_time_length

    def train( self, train: FBCSPDataset, test: FBCSPDataset ) -> None:

        learning_rate = 0.0001
        max_epochs = 30
        batch_size = 32
        weight_decay = 0.0
        restart_optimizer = False
        progress = False

        # model_parameters = filter( lambda p: p.requires_grad, self.model.parameters() )
        # params = sum( [ np.prod( p.size() ) for p in model_parameters ] )
        # print( f'Model has {params} trainable parameters' )

        loss_fn = nn.NLLLoss()
        if self.optimizer is None or restart_optimizer:
            self.optimizer = th.optim.AdamW( 
                self.model.parameters(), 
                lr = learning_rate, 
                weight_decay = weight_decay 
            )
            
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, T_max = max_epochs / 1 )

        train_loss, test_loss, test_accuracy, lr = [], [], [], []
        epoch_itr = range( max_epochs )

        if progress:
            try:
                from tqdm.autonotebook import tqdm
                epoch_itr = tqdm( epoch_itr )
            except ImportError:
                logger.warn( 'Attempted to provide progress bar, tqdm not installed' )

        # Calculate weights for class balancing
        classes, counts = th.unique( train.y, return_counts = True )
        weights = { cl.item(): 1.0 / co.item() for cl, co in zip( classes, counts ) }
        weights = [ weights[ lab.item() ] for lab in train.y ]

        for epoch in epoch_itr:

            self.model.train()
            train_loss_batches = []
            for train_feats, train_labels in DataLoader(
                train,
                batch_size = batch_size, 
                # drop_last = True,
                sampler = WeightedRandomSampler( weights, len( train ), replacement = False ),
                pin_memory = True,
            ):
                pred = self.model( train_feats.to( self._device ) )
                if self.cropped_training:
                    pred = pred.mean( axis = 2 )
                loss = loss_fn( pred, train_labels.to( self._device ) )
                train_loss_batches.append( loss.cpu().item() )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            scheduler.step()

            lr.append( scheduler.get_last_lr()[0] )
            train_loss.append( np.mean( train_loss_batches ) )

            self.model.eval()
            with th.no_grad():
                accuracy = 0
                test_loss_batches = []
                for test_feats, test_labels in DataLoader(
                    test, 
                    batch_size = batch_size, 
                    pin_memory = True
                ):
                    output = self.model( test_feats.to( self._device ) )
                    if self.cropped_training:
                        output = output.mean( axis = 2 )
                    loss = loss_fn( output, test_labels.to( self._device ) )
                    test_loss_batches.append( loss.cpu().item() )
                    accuracy += ( output.argmax( axis = 1 ).cpu() == test_labels ).sum().item()

                test_loss.append( np.mean( test_loss_batches ) )
                test_accuracy.append( accuracy / len( test ) )
                
