# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

#		LFprepsubbandMP

#	Script to compute the best dedispersion parameters, and carry out the prepsubband PRESTO command using multiprocessing.

#		Author :        Mark Brionne
#		Date :          18/08/2020
#		Version :       2c1

#		Comments :	New version of multiprocessing using a pool.
#				Correct the bug for the period research when the pulsar is unknown by Psrcat.
#				Correct the dmmin argument name for the research of the lowest DM.
#				Change the cjaracter percent to letters.
#				Check if the downsampling is less than the number of samples per block.

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Modules

import numpy as np
import argparse
import os
import pyfits as fi
import sys


# Input parameters

parser = argparse.ArgumentParser( description='Creation of the dedispersion plan used by prepsubband.' )
parser.add_argument( dest='fits' , type=str , help='Name of the FITS file to use.' )

args = parser.parse_args()


# Functions

def FitsInfo ( file ) :

        h0 = fi.getheader( file , 0 )
        h = fi.getheader( file , 1 )
        psr = h0['SRC_NAME']			# Name of the observed pulsar
        bwTot = float( h0['OBSBW'] )            # Total bandwidth of the observation
        fc = float( h0['OBSFREQ'] )             # Central frequency of the observation
        dm = float( h0['CHAN_DM'] )             # DM used by LUPPI to do the coherent dedispersion
        tsamp = float( h['TBIN'] )              # Time sample
        bwCh = float( h['CHAN_BW'] )            # Channel bandwidth
        blkLen = int( h['NSBLK'] )		# Number of samples per block
        nblk = int( h['NAXIS2'] )		# Number of samples per block
        chan_bw = float( h['CHAN_BW'] )


        return psr , fc - bwTot / 2. + bwCh / 2. , fc , fc + bwTot / 2. - bwCh / 2., chan_bw, tsamp , dm , blkLen, nblk


psrName , fmin , fcent , fmax , chan_bw , tsamp , dm0 , blkLen, nblk = FitsInfo( args.fits )

print("PSR name : %s" % psrName)
print("min frequency : %f" % fmin)
print("centre frequency : %f" % fcent)
print("max frequency : %f" % fmax)
print("chanel bandwidth : %f" % chan_bw)
print("time sample : %f" % tsamp)
print("block length : %d" % blkLen)
print("number of block : %d" % nblk)
print("coherent dedispersion : %f" % dm0)





