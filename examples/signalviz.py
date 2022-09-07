from dataclasses import dataclass, field
from pathlib import Path

import panel
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

from ezmsg.fbcsp.tsplotter import TSMessagePlot
from ezmsg.sigproc.messages import TSMessage

from ezmsg.sigproc.butterworthfilter import ButterworthFilter, ButterworthFilterSettings

from typing import Dict, Optional, Any, Tuple

class PlotServerSettings( ez.Settings ):
    port: int = 8082

class PlotServerState( ez.State ):
    raw_plot: TSMessagePlot = field( default_factory = TSMessagePlot )
    # filt_plot: TSMessagePlot = field( default_factory = TSMessagePlot )
    cur_x: int = 0

class PlotServer( ez.Unit ):

    SETTINGS: PlotServerSettings
    STATE: PlotServerState

    INPUT_SIGNAL_RAW = ez.InputStream( TSMessage )
    # INPUT_SIGNAL_FILT = ez.InputStream( TSMessage )

    @ez.subscriber( INPUT_SIGNAL_RAW )
    async def on_signal( self, msg: TSMessage ) -> None:
        self.STATE.raw_plot.update( msg )
    
    # @ez.subscriber( INPUT_SIGNAL_FILT )
    # async def on_signal( self, msg: TSMessage ) -> None:
    #     self.STATE.filt_plot.update( msg )

    @ez.task
    async def dashboard( self ) -> None:
        panel.serve( 
            dict(
                raw = self.STATE.raw_plot.client_view,
                # filt = self.STATE.filt_plot.client_view
            ), 
            port = self.SETTINGS.port 
        )


class SignalVizSystemSettings( ez.Settings ):
    openbcisource_settings: OpenBCISourceSettings
    server_settings: PlotServerSettings = field(
        default_factory = PlotServerSettings
    )


class SignalVizSystem( ez.System ):

    SETTINGS: SignalVizSystemSettings

    # Subunits
    SOURCE = OpenBCISource()
    BPFILT = ButterworthFilter()
    SERVER = PlotServer()

    def configure( self ) -> None:
        self.SOURCE.apply_settings( 
            self.SETTINGS.openbcisource_settings 
        )

        self.BPFILT.apply_settings(
            ButterworthFilterSettings(
                order = 5,
                cuton = 5,
                cutoff = None
            )
        )

        self.SERVER.apply_settings(
            self.SETTINGS.server_settings
        )


    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.SOURCE.OUTPUT_SIGNAL, self.BPFILT.INPUT_SIGNAL ),
            ( self.SOURCE.OUTPUT_SIGNAL, self.SERVER.INPUT_SIGNAL_RAW ),
            # ( self.BPFILT.OUTPUT_SIGNAL, self.SERVER.INPUT_SIGNAL_FILT ),
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = 'Signal Visualizer for OpenBCI (debug?)'
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

    args = parser.parse_args()

    device: str = args.device
    blocksize: int = args.blocksize
    gain: int = args.gain
    bias: str = args.bias
    powerdown: str = args.powerdown
    impedance: bool = args.impedance

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

    settings = SignalVizSystemSettings(
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
    )

    system = SignalVizSystem( settings )
    ez.run_system( system )
