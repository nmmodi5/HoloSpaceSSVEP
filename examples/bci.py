from dataclasses import field
from pathlib import Path

import ezmsg.core as ez

from ezmsg.eeg.openbci import (
    OpenBCISource,
    OpenBCISourceSettings, 
    GainState,
    PowerStatus,
    BiasSetting,
    OpenBCIChannelConfigSettings,
    OpenBCIChannelSetting,
)

from ezmsg.fbcsp.decoder import FBCSP, FBCSPSettings
from ezmsg.fbcsp.samplemapper import SampleMapperSettings
from ezmsg.fbcsp.trainingtask.server import TrainingTaskServerSettings
from ezmsg.fbcsp.tsmessageplot import TSMessagePlot, TSMessagePlotSettings
from ezmsg.fbcsp.panelapplication import Application, ApplicationSettings

from ezmsg.eeg.eegmessage import EEGMessage

from ezmsg.sigproc.decimate import Decimate, DownsampleSettings
from ezmsg.sigproc.butterworthfilter import ButterworthFilter, ButterworthFilterSettings
from ezmsg.sigproc.ewmfilter import EWMFilter, EWMFilterSettings
from ezmsg.sigproc.window import Window, WindowSettings

from typing import Dict, Optional, Any, Tuple

from hololight.demo import HololightDemo, HololightDemoSettings

class PreprocessingSettings( ez.Settings ):
    # 1. Bandpass Filter
    bpfilt_order: int = 5
    bpfilt_cuton: float = 1.0 # Hz
    bpfilt_cutoff: float = 45.0 # Hz

    # X. TODO: Common Average Reference/Spatial Filtering

    # 2. Downsample
    downsample_factor: int = 4 # Downsample factor to reduce sampling rate to ~ 100 Hz

    # 3. Exponentially Weighted Standardization
    ewm_history_dur: float = 2.0 # sec

    # 4. Sliding Window
    output_window_dur: float = 1.0 # sec
    output_window_shift: float = 1.0 # For training, we dont want overlap


class Preprocessing( ez.Collection ):

    SETTINGS: PreprocessingSettings

    INPUT_SIGNAL = ez.InputStream( EEGMessage )
    OUTPUT_SIGNAL = ez.OutputStream( EEGMessage )

    # Subunits
    BPFILT = ButterworthFilter()
    DECIMATE = Decimate()
    EWM = EWMFilter()
    WINDOW = Window()

    def configure( self ) -> None:
        self.BPFILT.apply_settings(
            ButterworthFilterSettings(
                order = self.SETTINGS.bpfilt_order,
                cuton = self.SETTINGS.bpfilt_cuton,
                cutoff = self.SETTINGS.bpfilt_cutoff
            )
        )
        self.DECIMATE.apply_settings( 
            DownsampleSettings(
                factor = self.SETTINGS.downsample_factor
            )
        )

        self.EWM.apply_settings(
            EWMFilterSettings(
                history_dur = self.SETTINGS.ewm_history_dur,
            )
        )

        self.WINDOW.apply_settings(
            WindowSettings(
                window_dur = self.SETTINGS.output_window_dur, # sec
                window_shift = self.SETTINGS.output_window_shift # sec
            )
        )


    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.INPUT_SIGNAL, self.BPFILT.INPUT_SIGNAL ),
            ( self.BPFILT.OUTPUT_SIGNAL, self.DECIMATE.INPUT_SIGNAL ),
            ( self.DECIMATE.OUTPUT_SIGNAL, self.EWM.INPUT_SIGNAL ),
            ( self.EWM.OUTPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL ),
            ( self.WINDOW.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL )
        )

class HololightSystemSettings( ez.Settings ):
    openbcisource_settings: OpenBCISourceSettings
    decoder_settings: FBCSPSettings
    demo_settings: HololightDemoSettings

    preprocessing_settings: PreprocessingSettings = field(
        default_factory = PreprocessingSettings
    )

class HololightSystem( ez.System ):

    SETTINGS: HololightSystemSettings

    SOURCE = OpenBCISource()
    SOURCE_PLOT = TSMessagePlot()
    PREPROC = Preprocessing()
    DECODER = FBCSP()
    HOLOLIGHT = HololightDemo()
    APP = Application()

    def configure( self ) -> None:
        self.SOURCE.apply_settings( self.SETTINGS.openbcisource_settings )
        self.PREPROC.apply_settings( self.SETTINGS.preprocessing_settings )
        self.DECODER.apply_settings( self.SETTINGS.decoder_settings )
        self.HOLOLIGHT.apply_settings( self.SETTINGS.demo_settings )

        self.SOURCE_PLOT.apply_settings( 
            TSMessagePlotSettings(
                name = 'OpenBCI Cyton Source'
            )
        )

        self.APP.apply_settings(
            ApplicationSettings(
                port = 8083,
                name = 'Hololight Dashboard'
            )
        )

        self.APP.panels = self.DECODER.panels()
        self.APP.panels['source'] = self.SOURCE_PLOT.GUI.panel

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.SOURCE.OUTPUT_SIGNAL, self.PREPROC.INPUT_SIGNAL ),
            ( self.SOURCE.OUTPUT_SIGNAL, self.SOURCE_PLOT.INPUT_SIGNAL ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.DECODER.INPUT_SIGNAL ),
            ( self.DECODER.OUTPUT_DECODE, self.HOLOLIGHT.INPUT_DECODE ),
        )

    def process_components( self ) -> Tuple[ ez.Component, ... ]:
        return ( 
            self.HOLOLIGHT, 
            self.SOURCE, 
            self.PREPROC,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = 'Hololight ARBCI Lightbulb Demonstration'
    )

    ## OpenBCI Arguments
    parser.add_argument(
        '--device',
        type = str,
        help = 'Serial port to pull data from',
        default = 'simulator'
    )

    parser.add_argument(
        '--blocksize',
        type = int,
        help = 'Sample block size @ 500 Hz',
        default = 100
    )

    parser.add_argument(
        '--gain',
        type = int,
        help = 'Gain setting for all channels.  Valid settings {1, 2, 4, 6, 8, 12, 24}',
        default = 24
    )

    parser.add_argument(
        '--bias',
        type = str,
        help = 'Include channels in bias calculation. Default: 11111111',
        default = '11111111'
    )

    parser.add_argument(
        '--powerdown',
        type = str,
        help = 'Channels to disconnect/powerdown. Default: 00111111',
        default = '00111111'
    )

    parser.add_argument(
        '--impedance',
        action = 'store_true',
        help = "Enable continuous impedance monitoring",
        default = False
    )

    ## Decoder Arguments
    parser.add_argument( 
        '--session-dir',
        type = lambda x: Path( x ),
        help = "Directory to store samples and model checkpoints",
        default = Path( '.' ) / 'test_session'
    )

    # Demo Settings
    parser.add_argument(
        '--bridge',
        type = str,
        help = 'Hostname for Philips Hue Bridge',
        default = None
    )

    parser.add_argument(
        '--cert',
        type = lambda x: Path( x ),
        help = "Certificate file for frontend server",
    )

    args = parser.parse_args()

    device: str = args.device
    blocksize: int = args.blocksize
    gain: int = args.gain
    bias: str = args.bias
    powerdown: str = args.powerdown
    impedance: bool = args.impedance

    session_dir: Path = args.session_dir

    bridge: Optional[ str ] = args.bridge
    cert: Path = args.cert

    gain_map: Dict[ int, GainState ] = {
        1:  GainState.GAIN_1,
        2:  GainState.GAIN_2,
        4:  GainState.GAIN_4,
        6:  GainState.GAIN_6,
        8:  GainState.GAIN_8,
        12: GainState.GAIN_12,
        24: GainState.GAIN_24
    }

    ch_setting = lambda ch_idx: ( 
        OpenBCIChannelSetting(
            gain = gain_map[ gain ], 
            power = ( PowerStatus.POWER_OFF 
                if powerdown[ch_idx] == '1' 
                else PowerStatus.POWER_ON ),
            bias = ( BiasSetting.INCLUDE   
                if bias[ch_idx] == '1'
                else BiasSetting.REMOVE 
            )
        )
    )

    if not cert.exists():
        raise ValueError( f"Certificate {cert} does not exist" )

    settings = HololightSystemSettings(

        openbcisource_settings = OpenBCISourceSettings(
            device = device,
            blocksize = blocksize,
            impedance = impedance,
            ch_config = OpenBCIChannelConfigSettings(
                ch_setting = tuple( [ 
                    ch_setting( i ) for i in range( 8 ) 
                ] )
            )
        ),

        decoder_settings = FBCSPSettings(
            session_dir = session_dir,
            inferencewindow_settings = WindowSettings(
                window_dur = 3.0,
                window_shift = 1.0
            ),
            samplemapper_settings = SampleMapperSettings(
                # Add a test signal for classification
                test_signal = 1.0 if device == 'simulator' else 0.0 
            ),
            application_port = None,
        ),

        demo_settings = HololightDemoSettings(
            cert = cert,
            bridge_host = bridge
        )
    )

    system = HololightSystem( settings )
    ez.run_system( system )
