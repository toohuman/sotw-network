# Methods related to serializing nested arrays into a writeable data structure.

import numpy as np

def serialize(data):
    """
    Provided a (nested) list of data, identify the nesting structure and convert
    the data to a 2-D list that can be written to a file. The original data
    structure should be reconstructable.
    """

    levels = 0
    widths = []


