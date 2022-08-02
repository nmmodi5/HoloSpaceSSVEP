import asyncio
import io
import json
import logging

from pathlib import Path
from dataclasses import dataclass, field, replace

import ezmsg.core as ez
from ezmsg.eeg.eegmessage import EEGMessage
from ezmsg.util.messagelogger import (
    MessageLogger, 
    MessageLoggerSettings, 
    MessageDecoder
)

import numpy as np

import torch as th
from torch.utils.data import Dataset, ConcatDataset

from .trainingtask.server import TrainingTaskServer, TrainingTaskServerSettings
from .sampler import Sampler, SamplerSettings, SampleMessage
from .classdecodemessage import ClassDecodeMessage
from .shallowfbcspnet import (
    EpochInfo, 
    ShallowFBCSPCheckpoint,
    ShallowFBCSPNet, 
    ShallowFBCSPParameters,
    ShallowFBCSPTraining,
    ShallowFBCSPTrainingParameters,
    balance_dataset
)

from typing import AsyncGenerator, List, Optional, Tuple, Any

logger = logging.getLogger( __name__ )

class FBCSPDataset( Dataset ):
    """
    Lazy loaded dataset from disk
    All messages in file assumed to be SampleMessages
    All SampleMessages assumed to have same sample dimensions and fs

    It'd probably be better to pickle SampleMessages directly to disk
    and reconstitute them to avoid issues with file format between runs,
    but a part of me worries the pickle format is not very forward-looking
    and the json serialized format may be usable by other programs and
    analyses in the future... -Griff
    """
    file: io.TextIOWrapper

    seekpoints: List[ int ]
    labels: List[ int ]

    dtype: np.dtype

    fs: Optional[ float ] = None
    n_ch: Optional[ int ] = None
    n_time: Optional[ int ] = None

    def __init__( self, filename: Path, single_precision: bool = True ) -> None:
        super().__init__()

        self.dtype = np.float32 if single_precision else np.float64
        self.seekpoints = list()
        self.file = open( filename, 'r' )   
        self.labels = []
  
        while self.file.readable():
            seekpoint = self.file.tell()
            line = self.file.readline()
            if len( line ):
                self.seekpoints.append( seekpoint )
                obj = json.loads( line, cls = MessageDecoder )
                self.labels.append( int( obj[ 'trigger' ][ 'value' ] ) )
                self.fs = float( obj[ 'sample' ][ 'fs' ] )
                data: np.ndarray = obj[ 'sample' ][ 'data' ]
                self.n_ch = data.shape[ obj[ 'sample' ][ 'ch_dim' ] ]
                self.n_time = data.shape[ obj[ 'sample' ][ 'time_dim' ] ]
            else: break

    def __len__( self ) -> int:
        return len( self.labels )

    def __getitem__( self, idx: int ) -> Tuple[ th.Tensor, th.Tensor ]:
        self.file.seek( self.seekpoints[ idx ] )
        obj = json.loads( self.file.readline(), cls = MessageDecoder )
        time_dim: int = obj[ 'sample' ][ 'time_dim' ]
        data: np.ndarray = obj[ 'sample' ][ 'data' ]
        data = np.swapaxes( data.astype( self.dtype ), time_dim, -1 )
        return th.tensor( data ), th.tensor( self.labels[ idx ] )

    def __del__( self ):
        self.file.close()


@dataclass
class FBCSPNetParameters:
    """ Downselected copy of ShallowFBCSPNetParameters with some replacements for physical units """
    # Training information
    cropped_training: bool = True # If True, operate on half-sized compute windows

    # First step -- Temporal Convolution (think FIR filtering)
    n_filters_time: int = 40
    filter_time_dur: float = 0.1 # sec (think FIR filter order once converted to samples)

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
    pool_time_dur: float = 0.3 # sec (think low pass boxcar filter on spectral content)
    pool_time_stride_dur: int = 0.05 # sec (think decimation of spectral content)
    pool_mode: str = 'mean' # || 'max'

    # Fifth Step - Pool Nonlinearity.  'safe_log' makes spectral power normally distributed
    pool_nonlin: Optional[ str ] = 'safe_log' # || square; No nonlin if None
    
    # Sixth Step - Dropout layer for training resilient network and convergence
    drop_prob: float = 0.5

    # Seventh step -- Dense layer to output class. No parameters
    # Eighth step -- LogSoftmax - Output to probabilities. No parameters

class FBCSPTrainMessage( ez.Message ):
    n_epochs: int
    reset: bool = False

    # All the following parameters are ignored unless reset = True
    model_dir: Optional[ Path ] = None
    net_params: Optional[ FBCSPNetParameters ] = None
    train_params: Optional[ ShallowFBCSPTrainingParameters ] = None

class FBCSPTrainingSettings( ez.Settings ):
    model_dir: Optional[ Path ] = None
    device: str = 'cpu'
    single_precision: bool = True
    default_network_params: FBCSPNetParameters = field( 
        default_factory = FBCSPNetParameters
    )
    default_train_params: ShallowFBCSPTrainingParameters = field(
        default_factory = ShallowFBCSPTrainingParameters
    )

class FBCSPTrainingState( ez.State ):
    context: Optional[ ShallowFBCSPTraining ] = None

class FBCSPTraining( ez.Unit ):
    SETTINGS: FBCSPTrainingSettings
    STATE: FBCSPTrainingState

    INPUT_TRAIN = ez.InputStream( FBCSPTrainMessage )
    OUTPUT_CHECKPOINT = ez.OutputStream( ShallowFBCSPCheckpoint )
    OUTPUT_EPOCH = ez.OutputStream( EpochInfo )
    OUTPUT_CONFUSION = ez.OutputStream( np.ndarray )

    @ez.subscriber( INPUT_TRAIN )
    @ez.publisher( OUTPUT_CHECKPOINT )
    @ez.publisher( OUTPUT_EPOCH )
    @ez.publisher( OUTPUT_CONFUSION )
    async def train( self, msg: FBCSPTrainMessage ) -> AsyncGenerator:

        if self.STATE.context is None or msg.reset:
            # Create a new model context
            model_dir = self.SETTINGS.model_dir
            if msg.model_dir is not None: 
                model_dir = msg.model_dir 
            if model_dir is None:
                logger.warn( 'Cancelling FBCSP Training -- No directory set' )
                return

            # Load datasets
            datasets: List[ FBCSPDataset ] = list()
            all_labels: List[ int ] = list()
            for fname in model_dir.glob( '*.txt' ):
                dataset = FBCSPDataset( fname, 
                    single_precision = self.SETTINGS.single_precision )
                if len( dataset ) == 0: continue

                datasets.append( dataset )
                in_chans = dataset.n_ch
                in_time = dataset.n_time
                fs = dataset.fs
                all_labels = all_labels + dataset.labels

            if len( datasets ) == 0:
                logger.warn( 'Cancelling FBCSP Training -- No datasets available' )
                return 

            datasets = ConcatDataset( datasets )
            train, test = balance_dataset( datasets, ( 0.8, 0.2 ) )
            
            # Create network
            net_params = self.SETTINGS.default_network_params
            if msg.net_params is not None:
                net_params = msg.net_params 
            net = ShallowFBCSPNet( 
                ShallowFBCSPParameters(
                    in_chans = in_chans,
                    n_classes = len( set( all_labels ) ),
                    input_time_length = in_time,
                    single_precision = self.SETTINGS.single_precision,
                    cropped_training = net_params.cropped_training,
                    n_filters_time = net_params.n_filters_time,
                    filter_time_length = int( net_params.filter_time_dur * fs ),
                    n_filters_spat = net_params.n_filters_spat,
                    split_first_layer = net_params.split_first_layer,
                    batch_norm = net_params.batch_norm,
                    batch_norm_alpha = net_params.batch_norm_alpha,
                    conv_nonlin = net_params.conv_nonlin,
                    pool_time_length = int( net_params.pool_time_dur * fs ),
                    pool_time_stride = int( net_params.pool_time_stride_dur * fs ),
                    pool_mode = net_params.pool_mode,
                    pool_nonlin = net_params.pool_nonlin,
                    drop_prob = net_params.drop_prob
                ),
                device = self.SETTINGS.device
            )

            train_params = self.SETTINGS.default_train_params
            if msg.train_params is not None:
                train_params = msg.train_params

            self.STATE.context = ShallowFBCSPTraining( net, train, test, train_params )
            yield self.OUTPUT_EPOCH, None

        # Train over many epochs and publish epoch info
        loop = asyncio.get_running_loop()
        for _ in range( msg.n_epochs ):
            info = await loop.run_in_executor( None, self.STATE.context.run_epoch )
            yield self.OUTPUT_EPOCH, info

        # Publish Confusion
        yield self.OUTPUT_CONFUSION, self.STATE.context.net.confusion( self.STATE.context.test )

        # Publish Checkpoint
        yield self.OUTPUT_CHECKPOINT, self.STATE.context.net.checkpoint

##########

class FBCSPInferenceSettings( ez.Settings ):
    checkpoint_file: Optional[ Path ] = None
    inference_device: str = 'cpu'

class FBCSPInferenceState( ez.State ):
    net: Optional[ ShallowFBCSPNet ] = None

class FBCSPInference( ez.Unit ):

    SETTINGS: FBCSPInferenceSettings
    STATE: FBCSPInferenceState

    INPUT_CHECKPOINT = ez.InputStream( ShallowFBCSPCheckpoint )

    INPUT_SIGNAL = ez.InputStream( EEGMessage )
    OUTPUT_DECODE = ez.OutputStream( ClassDecodeMessage )

    def initialize( self ) -> None:
        if self.SETTINGS.checkpoint_file:
            self.STATE.net = ShallowFBCSPNet.from_checkpoint_file( 
                self.SETTINGS.checkpoint_file, 
                device = self.SETTINGS.inference_device 
            )

    @ez.subscriber( INPUT_CHECKPOINT )
    async def on_checkpoint( self, msg: ShallowFBCSPCheckpoint ) -> None:
        self.STATE.net = ShallowFBCSPNet.from_checkpoint( 
            msg, device = self.SETTINGS.inference_device 
        )

    @ez.subscriber( INPUT_SIGNAL )
    @ez.publisher( OUTPUT_DECODE )
    async def decode_signal( self, msg: EEGMessage ) -> AsyncGenerator:
        if self.STATE.net is not None:
            if msg.n_time >= self.STATE.net.min_input_time_len:
                arr = np.swapaxes( msg.data, msg.time_dim, -1 )
                output = self.STATE.net.inference( arr )
                yield self.OUTPUT_DECODE, ClassDecodeMessage( 
                    data = output, 
                    fs = msg.fs / msg.n_time 
                )


import panel
import param

import matplotlib
matplotlib.use( "agg" )

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .shallowfbcspnet import plot_confusion_mpl, plot_train_info_mpl

class TrainingCurves( param.Parameterized ):
    # Reactive Parameters
    train_info: Optional[ List[ EpochInfo ] ] = param.Parameter()

    # Internal plotting functionality
    fig: Figure
    ax: Axes

    def __init__( self, **params ) -> None:
        super().__init__( **params )
        self.fig, self.ax = plt.subplots()

    def view( self ) -> Figure:
        if self.train_info is None:
            self.ax.clear()
        elif self.train_info is not None:
            self.ax.clear()
            plot_train_info_mpl( self.train_info, self.ax )
        self.ax.set_title( 'Train/Test Curves' )
        return self.fig

class ConfusionPlot( param.Parameterized ):
    # Reactive Parameters
    confusion: Optional[ np.ndarray ] = param.Parameter()

    # Internal plotting functionality
    fig: Figure
    ax: Axes

    def __init__( self, **params ) -> None:
        super().__init__( **params )
        self.fig, self.ax = plt.subplots()

    def view( self ) -> Figure:
        self.ax.clear()
        if self.confusion is not None:
            plot_confusion_mpl( self.confusion, self.ax, add_colorbar = False )
        self.ax.set_title( 'Test-set Confusion' )
        return self.fig


class TrainingControlPanel( param.Parameterized ):
    n_epochs: int = param.Integer( default = 10 )
    reset: bool = param.Boolean( default = False )

    # TODO: Add accessability of the remaining parameters
    # Except maybe model_dir, which should probably just be static

    # All the following parameters are ignored unless reset = True
    # model_dir: Optional[ Path ] = None
    # net_params: Optional[ FBCSPNetParameters ] = None
    # train_params: Optional[ ShallowFBCSPTrainingParameters ] = None

class TrainingDashboardSettings( ez.Settings ):
    port: int = 8083

class TrainingDashboardState( ez.State ):
    train_plot: TrainingCurves = field( default_factory = TrainingCurves )
    confusion_plot: ConfusionPlot = field( default_factory = ConfusionPlot )
    control_panel: TrainingControlPanel = field( default_factory = TrainingControlPanel )

    train_queue: "asyncio.Queue[FBCSPTrainMessage]" = field( 
        default_factory = asyncio.Queue 
    )

class TrainingDashboard( ez.Unit ):

    SETTINGS: TrainingDashboardSettings
    STATE: TrainingDashboardState

    INPUT_EPOCH = ez.InputStream( Optional[ EpochInfo ] )
    INPUT_CONFUSION = ez.InputStream( np.ndarray )
    OUTPUT_TRAIN = ez.OutputStream( FBCSPTrainMessage )

    @ez.subscriber( INPUT_EPOCH )
    async def on_epoch( self, msg: Optional[ EpochInfo ] ) -> None:
        if msg is None:
            self.STATE.train_plot.train_info = None
        else:
            info_list = self.STATE.train_plot.train_info
            if info_list is None: info_list = [ msg ]
            else: info_list.append( msg )
            self.STATE.train_plot.train_info = info_list

    @ez.subscriber( INPUT_CONFUSION )
    async def on_confusion( self, msg: np.ndarray ) -> None:
        self.STATE.confusion_plot.confusion = msg

    @ez.publisher( OUTPUT_TRAIN )
    async def start_training( self ) -> AsyncGenerator:
        while True:
            msg = await self.STATE.train_queue.get()
            yield self.OUTPUT_TRAIN, msg
        
    @ez.main
    def serve_dashboard( self ) -> None:

        train_button = panel.widgets.Button( name = 'Train', width = 50 )
        train_button.on_click( lambda _: 
            asyncio.get_running_loop().call_soon_threadsafe(
                self.STATE.train_queue.put_nowait, 
                FBCSPTrainMessage( 
                    n_epochs = self.STATE.control_panel.n_epochs, 
                    reset = self.STATE.control_panel.reset
                )
            )
        )

        # TODO
        deploy_button = panel.widgets.Button( name = 'Deploy', width = 50 )
        deploy_button.on_click( lambda event: print( 'Deploy!' ) )

        # TODO
        save_button = panel.widgets.Button( name = 'Save', width = 50 )
        save_button.on_click( lambda event: print( 'Save!' ) )

        button_row = panel.Row( train_button, deploy_button, save_button )

        dashboard = panel.Row(
            panel.Column(
                self.STATE.train_plot.view,
                self.STATE.confusion_plot.view,
            ),
            panel.Column(
                self.STATE.control_panel,
                button_row,
            ),
            # "<br>\n# [Training Task](https://google.com)",
        )
        panel.serve( dashboard, port = self.SETTINGS.port )


# Dev/Test Fixture

import time

from ezmsg.testing.debuglog import DebugLog
from ezmsg.sigproc.window import Window, WindowSettings
from ezmsg.eeg.eegmessage import EEGMessage

from .plotter import EEGPlotter
from .eegsynth import EEGSynth, EEGSynthSettings
from .preprocessing import Preprocessing, PreprocessingSettings

class SampleSignalModulatorSettings( ez.Settings ):
    signal_amplitude: float = 0.01

class SampleSignalModulatorState( ez.State ):
    classes: List[ str ] = field( default_factory = list )

class SampleSignalModulator( ez.Unit ):

    STATE: SampleSignalModulatorState
    SETTINGS: SampleSignalModulatorSettings

    INPUT_SAMPLE = ez.InputStream( SampleMessage )
    OUTPUT_SAMPLE = ez.OutputStream( SampleMessage )

    OUTPUT_EEG = ez.OutputStream( EEGMessage )

    @ez.subscriber( INPUT_SAMPLE )
    @ez.publisher( OUTPUT_SAMPLE )
    @ez.publisher( OUTPUT_EEG )
    async def on_sample( self, msg: SampleMessage ) -> AsyncGenerator:

        if msg.trigger.value not in self.STATE.classes:
            self.STATE.classes.append( msg.trigger.value )

        assert isinstance( msg.sample, EEGMessage )
        sample: EEGMessage = msg.sample

        ch_idx = min( self.STATE.classes.index( msg.trigger.value ), sample.n_ch )
        arr = np.swapaxes( sample.data, sample.time_dim, 0 )

        sample_time = ( np.arange( sample.n_time ) / sample.fs )
        test_signal = np.sin( 2.0 * np.pi * 20.0 * sample_time )
        test_signal = test_signal * np.hamming( sample.n_time )
        arr[:, ch_idx] = arr[:, ch_idx] + ( test_signal * self.SETTINGS.signal_amplitude )

        sample = replace( sample, data = np.swapaxes( arr, sample.time_dim, 0 ) )
        yield self.OUTPUT_EEG, sample
        yield self.OUTPUT_SAMPLE, replace( msg, sample = sample )

# class PublishOnceSettings( ez.Settings ):
#     msg: Any
#     delay: Optional[ float ] = None 

# class PublishOnce( ez.Unit ):
#     SETTINGS: PublishOnceSettings
#     OUTPUT = ez.OutputStream( Any )

#     @ez.publisher( OUTPUT )
#     async def pub_once( self ) -> AsyncGenerator:
#         if self.SETTINGS.delay:
#             await asyncio.sleep( self.SETTINGS.delay )
#         yield self.OUTPUT, self.SETTINGS.msg


class ShallowFBCSPTrainingTestSystemSettings( ez.Settings ):
    session: Path
    trainingserver_settings: TrainingTaskServerSettings
    train_device: str = 'cpu'
    inference_device: str = 'cpu'
    eeg_settings: EEGSynthSettings = field( 
        default_factory = EEGSynthSettings 
    )
    preproc_settings: PreprocessingSettings = field(
        default_factory = PreprocessingSettings
    )

class ShallowFBCSPTrainingTestSystem( ez.System ):

    SETTINGS: ShallowFBCSPTrainingTestSystemSettings

    EEG = EEGSynth()
    PREPROC = Preprocessing()
    SAMPLER = Sampler()
    INJECTOR = SampleSignalModulator()
    LOGGER = MessageLogger()

    PLOTTER = EEGPlotter()

    TASK_SERVER = TrainingTaskServer()
    FBCSP_TRAINING = FBCSPTraining()
    FBCSP_INFERENCE = FBCSPInference()
    DASHBOARD = TrainingDashboard()
    DEBUG = DebugLog()

    def configure( self ) -> None:
        timestr = time.strftime( '%Y%m%dT%H%M%S' )
        run_fname = self.SETTINGS.session / f'{timestr}.txt'

        self.FBCSP_TRAINING.apply_settings( 
            FBCSPTrainingSettings(
                model_dir = self.SETTINGS.session,
                device = self.SETTINGS.train_device,
            )
        )

        self.FBCSP_INFERENCE.apply_settings(
            FBCSPInferenceSettings(
                checkpoint_file = None,
                inference_device = self.SETTINGS.inference_device
            )
        )

        self.EEG.apply_settings(
            self.SETTINGS.eeg_settings
        )

        self.PREPROC.apply_settings(
            self.SETTINGS.preproc_settings
        )

        self.TASK_SERVER.apply_settings(
            self.SETTINGS.trainingserver_settings
        )

        self.SAMPLER.apply_settings(
            SamplerSettings(
                buffer_dur = 5.0 # sec
            )
        )

        self.LOGGER.apply_settings(
            MessageLoggerSettings(
                output = run_fname
            )
        )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.EEG.OUTPUT_SIGNAL, self.PREPROC.INPUT_SIGNAL ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.SAMPLER.INPUT_SIGNAL ),
            ( self.SAMPLER.OUTPUT_SAMPLE, self.INJECTOR.INPUT_SAMPLE ),
            ( self.INJECTOR.OUTPUT_SAMPLE, self.LOGGER.INPUT_MESSAGE ),

            ( self.INJECTOR.OUTPUT_SAMPLE, self.DEBUG.INPUT ),

            ( self.FBCSP_TRAINING.OUTPUT_CHECKPOINT, self.FBCSP_INFERENCE.INPUT_CHECKPOINT ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.FBCSP_INFERENCE.INPUT_SIGNAL ),
            
            # DEBUG STUFF
            # ( self.PUBONCE.OUTPUT, self.FBCSP_TRAINING.INPUT_TRAIN ),
            # ( self.FBCSP_TRAINING.OUTPUT_EPOCH, self.DEBUG.INPUT ),
            # ( self.FBCSP_TRAINING.OUTPUT_CONFUSION, self.DEBUG.INPUT ),
            # ( self.FBCSP_TRAINING.OUTPUT_CHECKPOINT, self.DEBUG.INPUT ),
            ( self.FBCSP_INFERENCE.OUTPUT_DECODE, self.DEBUG.INPUT ),

            ( self.FBCSP_TRAINING.OUTPUT_EPOCH, self.DASHBOARD.INPUT_EPOCH ),
            ( self.FBCSP_TRAINING.OUTPUT_CONFUSION, self.DASHBOARD.INPUT_CONFUSION ),
            ( self.DASHBOARD.OUTPUT_TRAIN, self.FBCSP_TRAINING.INPUT_TRAIN ),

            ( self.TASK_SERVER.OUTPUT_SAMPLETRIGGER, self.SAMPLER.INPUT_TRIGGER ),

            # Plotter connections
            ( self.INJECTOR.OUTPUT_EEG, self.PLOTTER.INPUT_SIGNAL ),

        )

    def process_components( self ) -> Tuple[ ez.Component, ... ]:
        return (  
            self.PLOTTER, 
            self.TASK_SERVER,
            self.LOGGER,
            self.DASHBOARD,
            self.FBCSP_TRAINING,
            self.FBCSP_INFERENCE
        )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description = 'ShallowFBCSP Dev/Test Environment'
    )

    parser.add_argument(
        '--channels',
        type = int,
        help = "Number of EEG channels to simulate",
        default = 8
    )

    parser.add_argument( 
        '--session',
        type = lambda x: Path( x ),
        help = "Directory to store samples and model checkpoints",
        default = Path( '.' ) / 'test_session'
    )

    parser.add_argument(
        '--cert',
        type = lambda x: Path( x ),
        help = "Certificate file for frontend server",
        default = ( Path( '.' ) / 'cert.pem' ).absolute()
    )

    parser.add_argument(
        '--key',
        type = lambda x: Path( x ),
        help = "Private key for frontend server [Optional -- assumed to be included in --cert file if omitted)",
        default = None
    )

    parser.add_argument(
        '--cacert',
        type = lambda x: Path( x ),
        help = "Certificate for custom authority [Optional]",
        default = None
    )

    args = parser.parse_args()

    channels: int = args.channels
    session: Path = args.session
    cert: Path = args.cert
    key: Optional[ Path ] = args.key
    cacert: Optional[ Path ] = args.cacert

    settings = ShallowFBCSPTrainingTestSystemSettings(
        session = session,
        eeg_settings = EEGSynthSettings(
            fs = 500.0, # Hz
            channels = channels,
            blocksize = 100, # samples per block
            amplitude = 10e-6, # Volts
            dc_offset = 0, # Volts
            alpha = 9.5, # Hz; don't add alpha if None

            # Rate (in Hz) at which to dispatch EEGMessages
            # None => as fast as possible
            # float number => block publish rate in Hz
            # 'realtime' => Use wall-clock to publish EEG at proper rate
            dispatch_rate = 'realtime'
        ),

        preproc_settings = PreprocessingSettings(
            # 1. Bandpass Filter
            bpfilt_order = 5,
            bpfilt_cuton = 5.0, # Hz
            bpfilt_cutoff = 30.0, # Hz

            # 2. Downsample
            downsample_factor = 2, # Downsample factor to reduce sampling rate to ~ 250 Hz

            # 3. Exponentially Weighted Standardization
            ewm_history_dur = 4.0, # sec

            # 4. Sliding Window
            output_window_dur = 4.0, # sec
            output_window_shift = 4.0, # sec
        ),

        trainingserver_settings = TrainingTaskServerSettings(
            cert = cert,
            key = key,
            ca_cert = cacert
        ),
    )

    system = ShallowFBCSPTrainingTestSystem( settings )

    ez.run_system( system )
