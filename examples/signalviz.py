from dataclasses import field
import socket
import logging

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

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

from typing import Dict, Optional

class PlotServerSettings( ez.Settings ):
    port: int = 8082

class PlotServerState( ez.State ):
    plot: TSMessagePlot = field( default_factory = TSMessagePlot )

class PlotServer( ez.Unit ):

    SETTINGS: PlotServerSettings
    STATE: PlotServerState

    INPUT_SIGNAL = ez.InputStream( TSMessage )
    
    @ez.subscriber( INPUT_SIGNAL )
    async def on_signal( self, msg: TSMessage ) -> None:
        self.STATE.plot.update( msg )

    @ez.task
    async def dashboard( self ) -> None:
        panel.serve( 
            self.STATE.plot.client,
            port = self.SETTINGS.port,
            websocket_origin = [
                f'localhost:{self.SETTINGS.port}',
                f'{socket.gethostname()}:{self.SETTINGS.port}'
            ],
            show = False
        )


class SignalVizSystemSettings( ez.Settings ):
    openbcisource_settings: OpenBCISourceSettings
    cuton: Optional[ float ] = None
    cutoff: Optional[ float ] = None
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
                cuton = self.SETTINGS.cuton,
                cutoff = self.SETTINGS.cutoff
            )
        )

        self.SERVER.apply_settings(
            self.SETTINGS.server_settings
        )


    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.SOURCE.OUTPUT_SIGNAL, self.BPFILT.INPUT_SIGNAL ),
            ( self.BPFILT.OUTPUT_SIGNAL, self.SERVER.INPUT_SIGNAL ),
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

    parser.add_argument(
        '--cuton',
        type = float,
        help = 'Filter cuton',
        default = None
    )

    parser.add_argument(
        '--cutoff',
        type = float,
        help = 'Filter cutoff',
        default = None
    )

    args = parser.parse_args()

    device: str = args.device
    blocksize: int = args.blocksize
    gain: int = args.gain
    bias: str = args.bias
    powerdown: str = args.powerdown
    impedance: bool = args.impedance
    cuton: Optional[ float ] = args.cuton
    cutoff: Optional[ float ] = args.cutoff

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
        cuton = cuton,
        cutoff = cutoff,
    )

    system = SignalVizSystem( settings )
    ez.run_system( system )
