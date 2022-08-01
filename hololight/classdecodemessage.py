from dataclasses import field

from ezmsg.sigproc.timeseriesmessage import TimeSeriesMessage

from typing import (
    Optional,
    List
)

class ClassDecodeMessage( TimeSeriesMessage ):
    """ Class Decode Messages have two dimensions, time and class """

    class_dim: int = field( default = 1, init = False )
    class_names: Optional[ List[ str ] ] = None

    @property
    def n_classes( self ) -> int:
        """ Number of classes in the message """
        return self.shape[ self.class_dim ]