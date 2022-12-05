import asyncio
import http.server
import ssl
import logging
import traceback

from pathlib import Path

import ezmsg.core as ez
import numpy as np

import websockets
import websockets.server
import websockets.exceptions

from ezmsg.fbcsp.classdecodemessage import ClassDecodeMessage

from typing import AsyncGenerator, Optional, List

logger = logging.getLogger( __name__ )


class HololightDemoSettings( ez.Settings ):
    cert: Path
    host: str = '0.0.0.0'
    port: int = 8081
    ws_port: int = 8082

class HololightDemoState( ez.State ):
    decode_class: Optional[ int ] = None
    new_decoded_op: Optional[ bool ] = None
    start_SSVEP_decode: Optional[ bool ] = None

class HololightDemo( ez.Unit ):

    SETTINGS: HololightDemoSettings
    STATE: HololightDemoState

    INPUT_DECODE = ez.InputStream( ClassDecodeMessage )

    def initialize( self ) -> None:
        try:
            logger.info(f'initialize is empty for now !')
        except:
            logger.warn( f'Failed to connect. Traceback follows:\n{traceback.format_exc()}' )

    @ez.task
    async def start_websocket_server( self ) -> None:

        async def connection( websocket: websockets.server.WebSocketServerProtocol, path ):
            logger.info( 'Client Connected to Websocket Input' )

            try:
                while True:
                    if (not self.STATE.start_SSVEP_decode) :
                        data = await websocket.recv()
                        cmd, value = data.split( ': ' )
                        if cmd == 'COMMAND':
                            if value == 'START_SSVEP_DECODE':
                                self.STATE.start_SSVEP_decode = True
                                logger.info(f'self.STATE.start_SSVEP_decode: {self.STATE.start_SSVEP_decode}')
                        elif cmd == 'STATUS':
                            logger.info(f'STATUS: {value}')
                        else:
                            logger.info( f'Received problematic message from websocket client: {data}')
                    else:
                        if (self.STATE.new_decoded_op):
                            await websocket.send(f'CLASS: {self.STATE.decode_class[0]}')
                            self.STATE.new_decoded_op = None
                            logger.info( f'ws: sent {self.STATE.decode_class[0]}' )
                            self.STATE.start_SSVEP_decode = None

            except ( websockets.exceptions.ConnectionClosed ):
                logger.info( 'Websocket Client Closed Connection' )
            except asyncio.CancelledError:
                logger.info( 'Websocket Client Handler Task Cancelled!' )
            except Exception as e:
                logger.warn( 'Error in websocket server:', e )
            finally:
                logger.info( 'Websocket Client Handler Task Concluded' )

        try:
            ssl_context = ssl.SSLContext( ssl.PROTOCOL_TLS_SERVER ) 
            ssl_context.load_cert_chain( 
                certfile = self.SETTINGS.cert, 
                keyfile = self.SETTINGS.cert 
            )

            server = await websockets.server.serve(
                connection,
                self.SETTINGS.host,
                self.SETTINGS.ws_port,
                ssl = ssl_context
            )

            await server.wait_closed()

        finally:
            logger.info( 'Closing Websocket Server' )

    ## action to be taken on decoding the input
    @ez.subscriber( INPUT_DECODE )
    async def on_decode( self, decode: ClassDecodeMessage ) -> None:
        cur_class = decode.data.argmax( axis = decode.class_dim )
        cur_prob = decode.data[ :, cur_class ]

        self.STATE.decode_class = cur_class
        self.STATE.new_decoded_op = [True]
        #logger.info( f'self.STATE.decode_class {self.STATE.decode_class}' )

    @ez.main
    def serve( self ):

        directory = str( ( Path( __file__ ).parent / 'web' ) )

        class Handler( http.server.SimpleHTTPRequestHandler ):
            def __init__( self, *args, **kwargs ):
                super().__init__( *args, directory = directory, **kwargs )

        address = ( self.SETTINGS.host, self.SETTINGS.port )
        httpd = http.server.HTTPServer( address, Handler )

        httpd.socket = ssl.wrap_socket(
            httpd.socket,
            server_side = True,
            certfile = self.SETTINGS.cert,
            ssl_version = ssl.PROTOCOL_TLS_SERVER
        )

        httpd.serve_forever()


### DEV/TEST APPARATUS

class GenerateDecodeOutput( ez.Unit ):
    ### ClassDecodeMessage gets the decoded message
    OUTPUT_DECODE = ez.OutputStream( ClassDecodeMessage )

    @ez.publisher( OUTPUT_DECODE )
    async def generate( self ) -> AsyncGenerator:
        ## Assuming the actual outputs intended are [8], [10], [12], [15]
        ## output is one hot encoding for each decoded frequency output
        ## Note: In actuality we may have something like [8, 8, 8, 10, 8, 8, 10]
        ## for each intended output and we need to do one-hot encoding of output accordingly
        output = np.array( [ [ 1, 0, 0, 0 ] ] )
        while True:
            out = ( output.astype( float ) * 0.9 ) + 0.05
            out /= out.sum()
            yield self.OUTPUT_DECODE, ClassDecodeMessage( data = out, fs = 0.5 )
            await asyncio.sleep( 2.0 )
            output = np.roll(output, 1)

class HololightTestSystem( ez.System ):

    SETTINGS: HololightDemoSettings

    HOLOLIGHT = HololightDemo()
    DECODE_TEST = GenerateDecodeOutput()

    def configure( self ) -> None:
        return self.HOLOLIGHT.apply_settings( self.SETTINGS )

    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.DECODE_TEST.OUTPUT_DECODE, self.HOLOLIGHT.INPUT_DECODE ),
        )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Hololight Test Script'
    )

    parser.add_argument(
        '--cert',
        type = lambda x: Path( x ),
        help = "Certificate file for frontend server",
        default = ( Path( '.' ) / 'cert.pem' ).absolute()
    )

    args = parser.parse_args()
    cert: Path = args.cert

    settings = HololightDemoSettings(
        cert = cert
    )

    system = HololightTestSystem( settings )
    ez.run_system( system )



