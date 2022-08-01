import asyncio
from pathlib import Path
from dataclasses import field, replace

import io
import logging
import json

import ezmsg.core as ez
from ezmsg.eeg.eegmessage import EEGMessage
from ezmsg.util.messagelogger import MessageDecoder

import numpy as np

import torch as th
from torch.utils.data import Dataset, ConcatDataset

from .sampler import SampleMessage
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

from typing import AsyncGenerator, List, Optional, Tuple, Any, Set

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
        data = np.swapaxes( data.astype( self.dtype ), time_dim, 0 )
        return th.tensor( data ), th.tensor( self.labels[ idx ] )

    def __del__( self ):
        self.file.close()

class FBCSPTrainMessage( ez.Message ):
    n_epochs: int
    reset: bool = False

class FBCSPTrainingSettings( ez.Settings ):
    model_dir: Path

class FBCSPTrainingState( ez.State ):
    ...

class FBCSPTraining( ez.Unit ):
    SETTINGS: FBCSPTrainingSettings
    STATE: FBCSPTrainingState

    INPUT_TRAIN = ez.InputStream( FBCSPTrainMessage )
    OUTPUT_CHECKPOINT = ez.OutputStream( ShallowFBCSPCheckpoint )
    OUTPUT_EPOCH = ez.OutputStream( EpochInfo )

    @ez.subscriber( INPUT_TRAIN )
    @ez.publisher( OUTPUT_CHECKPOINT )
    @ez.publisher( OUTPUT_EPOCH )
    async def train( self, msg: Path ) -> AsyncGenerator:

        # Load datasets
        datasets: List[ FBCSPDataset ] = list()
        all_labels: List[ int ] = list()
        for fname in msg.glob( '*.txt' ):
            dataset = FBCSPDataset( fname )
            if len( dataset ) == 0: continue
            datasets.append( dataset )
            in_chans = dataset.n_ch
            in_time = dataset.n_time
            fs = dataset.fs
            all_labels = all_labels + dataset.labels
        datasets = ConcatDataset( datasets )

        # Create network
        net = ShallowFBCSPNet( 
            ShallowFBCSPParameters(
                in_chans = in_chans,
                n_classes = len( set( all_labels ) ),
                input_time_length = in_time,
                single_precision = True,
                cropped_training = True,
                filter_time_length = int( 0.1 * fs ),
                pool_time_length = int( 0.3 * fs ),
                pool_time_stride = int( 0.06 * fs )
            )
        )

        # Create training module
        num_epochs = 30
        train, test = balance_dataset( datasets, ( 0.8, 0.2 ) )
        training = ShallowFBCSPTraining(
            net.model, 
            train, 
            test,
            ShallowFBCSPTrainingParameters(
                annealing_epochs = num_epochs
            )
        )

        # Train over many epochs and publish epoch info
        loop = asyncio.get_running_loop()
        for _ in range( num_epochs ):
            info = await loop.run_in_executor( None, training.run_epoch )
            yield self.OUTPUT_EPOCH, info

        # Publish Checkpoint
        yield self.OUTPUT_CHECKPOINT, net.checkpoint

class FBCSPInferenceSettings( ez.Settings ):
    checkpoint_file: Optional[ Path ] = None
    inference_device: str = 'cpu'

class FBCSPInferenceState( ez.Settings ):
    net: Optional[ ShallowFBCSPNet ] = None

class FBCSPInference( ez.Unit ):

    SETTINGS: FBCSPInferenceSettings
    STATE: FBCSPInferenceState

    INPUT_MODEL = ez.InputStream( ShallowFBCSPCheckpoint )

    INPUT_SIGNAL = ez.InputStream( EEGMessage )
    OUTPUT_DECODE = ez.OutputStream( ClassDecodeMessage )

    @ez.subscriber( INPUT_MODEL )
    async def on_checkpoint( self, msg: ShallowFBCSPCheckpoint ) -> None:
        self.STATE.net = ShallowFBCSPNet.from_checkpoint( 
            msg, device = self.SETTINGS.inference_device 
        )

    @ez.subscriber( INPUT_SIGNAL )
    @ez.publisher( OUTPUT_DECODE )
    async def decode_signal( self, msg: EEGMessage ) -> AsyncGenerator:
        if self.STATE.net is not None:
            if msg.n_time >= self.STATE.net.optimal_temporal_stride:
                arr = np.swapaxes( msg.data, msg.time_dim, 0 )
                output = self.STATE.net.inference( arr )
                yield self.OUTPUT_DECODE, ClassDecodeMessage( data = output )


# Dev/Test Fixture

from pathlib import Path

from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings

from ezmsg.testing.debuglog import DebugLog
from ezmsg.sigproc.window import Window, WindowSettings
from ezmsg.eeg.eegmessage import EEGMessage

from .plotter import EEGPlotter
from .eegsynth import EEGSynth, EEGSynthSettings
from .preprocessing import Preprocessing, PreprocessingSettings
from .training.server import TrainingServer, TrainingServerSettings
from .sampler import Sampler, SamplerSettings

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

class PublishOnceSettings( ez.Settings ):
    msg: Any

class PublishOnce( ez.Unit ):

    SETTINGS: PublishOnceSettings

    OUTPUT = ez.OutputStream( Any )

    @ez.publisher( OUTPUT )
    async def pub_once( self ) -> AsyncGenerator:
        yield self.OUTPUT, self.SETTINGS.msg


class ShallowFBCSPTrainingTestSystemSettings( ez.Settings ):
    shallowfbcsptraining_settings: FBCSPTrainingSettings
    trainingserver_settings: TrainingServerSettings
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

    WINDOW = Window()
    PLOTTER = EEGPlotter()

    TRAIN_SERVER = TrainingServer()
    FBCSP_TRAINING = FBCSPTraining()
    DEBUG = DebugLog()

    PUBONCE = PublishOnce( 
        PublishOnceSettings(
            msg = Path( '.' ) / 'recordings'
        )
    )

    def configure( self ) -> None:
        self.FBCSP_TRAINING.apply_settings( 
            self.SETTINGS.shallowfbcsptraining_settings 
        )

        self.EEG.apply_settings(
            self.SETTINGS.eeg_settings
        )

        self.PREPROC.apply_settings(
            self.SETTINGS.preproc_settings
        )

        self.TRAIN_SERVER.apply_settings(
            self.SETTINGS.trainingserver_settings
        )

        self.WINDOW.apply_settings(
            WindowSettings( 
                window_dur = 4.0, # sec
                window_shift = 1.0 # sec
            )
        )

        self.SAMPLER.apply_settings(
            SamplerSettings(
                buffer_dur = 5.0 # sec
            )
        )

        self.LOGGER.apply_settings(
            MessageLoggerSettings(
                output = Path( '.' ) / 'recordings' / 'traindata.txt'
            )
        )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.EEG.OUTPUT_SIGNAL, self.PREPROC.INPUT_SIGNAL ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.SAMPLER.INPUT_SIGNAL ),
            ( self.SAMPLER.OUTPUT_SAMPLE, self.INJECTOR.INPUT_SAMPLE ),
            # ( self.INJECTOR.OUTPUT_SAMPLE, self.FBCSP_TRAINING.INPUT_SAMPLE ),
            ( self.INJECTOR.OUTPUT_SAMPLE, self.LOGGER.INPUT_MESSAGE ),

            ( self.INJECTOR.OUTPUT_SAMPLE, self.DEBUG.INPUT ),
            ( self.PUBONCE.OUTPUT, self.FBCSP_TRAINING.INPUT_TRAIN ),

            ( self.TRAIN_SERVER.OUTPUT_SAMPLETRIGGER, self.SAMPLER.INPUT_TRIGGER ),

            # Plotter connections
            # ( self.EEG.OUTPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL ),
            # ( self.WINDOW.OUTPUT_SIGNAL, self.PLOTTER.INPUT_SIGNAL ),
            ( self.INJECTOR.OUTPUT_EEG, self.PLOTTER.INPUT_SIGNAL ),

            
        )

    def process_components( self ) -> Tuple[ ez.Component, ... ]:
        return ( 
            self.FBCSP_TRAINING, 
            self.PLOTTER, 
            self.TRAIN_SERVER,
            self.LOGGER
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
    cert: Path = args.cert
    key: Optional[ Path ] = args.key
    cacert: Optional[ Path ] = args.cacert

    settings = ShallowFBCSPTrainingTestSystemSettings(
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
            output_window_dur = 1.0, # sec
            output_window_shift = 1.0, # sec
        ),

        trainingserver_settings = TrainingServerSettings(
            cert = cert,
            key = key,
            ca_cert = cacert
        ),

        shallowfbcsptraining_settings = ModelTrainingSettings(
        )
    )

    system = ShallowFBCSPTrainingTestSystem( settings )

    ez.run_system( system )
