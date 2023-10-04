# -----------------------------------------------------------------------------------------
#					CalcRfifindParam

#	Script to compute the different parameters to send on input for rfifind.
#		Author :	Mark Brionne
#		Date :		02/07/2020
#		Version :	2d4

#		Comments :	Removing the offset computing.
#				Replacing the tint varible by the number of blocks to use.
#				Correct a bug if nblocks = 0.
#				Extract the KP name from the FITS header for the definition of the start offset.
#				Correct the altazA file research to search different minutes for the filename.


# -----------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------
# MODULES

import numpy as np
import argparse
import pyfits as fi
import os
import sys
from astropy.time import Time
import glob
import datetime


# -----------------------------------------------------------------------------------------
# INPUT PARAMETERS

parser = argparse.ArgumentParser( 'Script to calculate blocks and channels to zap.' )
parser.add_argument( dest='fits' , type=str , help='Name of the FITS file to use.' )
parser.add_argument( '-p' , dest='per' , help='Period of the pulsar in seconds.' )
parser.add_argument( '-n' , dest='nper' , type=float , default=2. , help='Integration time in number of periods which will be used for rfifind in seconds (default = 2).' )
parser.add_argument( '-wi' , dest='widthint' , type=float , default=1.5 , help='Width of analogical beam jumps in seconds (default = 1.5 s).'  )
parser.add_argument( '-wc' , dest='widthchan' , type=float , default=0.5 , help='Width of the frequency interferences in MHz (default = 0.5 MHz).'  )
parser.add_argument( '-fsig' , type=float , default=20. , help='Threshold to apply on the research on power to reject the bad max powers (default = 20.0).' )
parser.add_argument( '-tsig' , type=float , default=3. , help='Threshold to apply on the research on std and median to reject the bad integrations and channels (default = 3.0).' )
parser.add_argument( '-cfrac' , type=float , default=0.3 , help='Fraction of channels inside an integration to reject the entire integration (default = 0.3).' )
parser.add_argument( '-ifrac' , type=float , default=0.3 , help='Fraction of integrations inside an channel to reject the entire channel (default=0.3).' )
parser.add_argument( '-offs' , type=float , default=120.0 , help='Offset in the start time between the observation and the file for other KP than ES03 (default = 120 s).' )
args = parser.parse_args()




# -----------------------------------------------------------------------------------------
# EXTRACT FITS DATA

_file0 = fi.open( args.fits , mode='update' )						# Open the FITS file to update it

tblock = float( _file0[1].header['TBIN'] ) * int( _file0[1].header['NSBLK'] )		# Computing the block time size


# -----------------------------------------------------------------------------------------
# COMPUTING INTEGRATION TIME

try :							# Check if Psrcat know the pulsar period
	float( args.per )
	nblocks = int( round( float( args.per ) * args.nper / tblock ) )			# Computing the rfi time
except ValueError :
	nblocks = 1					# Default number of blocks equal to 1 block

if nblocks == 0 :
	nblocks = 1

# -----------------------------------------------------------------------------------------
# RUN RFIFIND

psrName = args.fits.split('_rescaled')[0]

print('rfifind -psrfits -noclip -blocks {:d} -freqsig {:.1f} -timesig {:.1f} -chanfrac {:.1f} -intfrac {:.1f} -o {:s} {:s}'
	.format( nblocks , args.fsig , args.tsig , args.cfrac , args.ifrac , psrName , args.fits ))
os.system( 'rfifind -psrfits -noclip -blocks {:d} -freqsig {:.1f} -timesig {:.1f} -chanfrac {:.1f} -intfrac {:.1f} -o {:s} {:s}'
	.format( nblocks , args.fsig , args.tsig , args.cfrac , args.ifrac , psrName , args.fits ) )


# -----------------------------------------------------------------------------------------
