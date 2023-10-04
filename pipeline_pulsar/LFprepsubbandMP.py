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
import multiprocessing as mp
import argparse
import os
import subprocess
import pyfits as fi
import sys


# Input parameters

parser = argparse.ArgumentParser( description='Creation of the dedispersion plan used by prepsubband.' )
parser.add_argument( dest='fits' , type=str , help='Name of the FITS file to use.' )
parser.add_argument( '-p' , '--per' , type=float , help='Period of the pulsar (default is the PSRCAT value).' )
#parser.add_argument( '-t' , '--scat' , type=float , help='Scattering constant time at 1 GHz (default is the PSRCAT value if it exists or 5 percent of the period).' )
parser.add_argument( '-i' , '--interv' , default=5.0 , type=float , help='Interval wished around the expected DM value (default = 5.0 pc.cm-3).' )
parser.add_argument( '-s' , '--dmstep' , default=0.0001 , type=float , help='Minimal DM step to use (by default calculated on the total bandwidth with a minimal value of 0.0001 pc.cm-3.' )
parser.add_argument( '-d' , '--dsmin' , default=1 , type=int , help='Minimal downsampling to use (by default calculated relative to the scattering and the period of the pulsar.')
parser.add_argument( '-m' , '--dmmin' , type=float , default=1.0 , help='Minimal DM  to use for the research (default = 1.0 pc.cm-3).' )
parser.add_argument( '-n' , '--nbmax' , type=int , default=100 , help='Maximal number of DM to try for a given downsampling (default = 100).' )
parser.add_argument( '-c' , '--ncpus' , type=int , default=10 , help='Number of cores to use.' )
parser.add_argument( '-k' , '--mask', type=str , help='Name of the mask file to use (default = 10).' )
parser.add_argument( '-o' , '--out', type=str , help='Name of the DD plan outfile to create.' )
parser.add_argument( '-t' , '--threshold' , type=float , default=4.0 , help='Minimal DM  to use for the research (default = 1.0 pc.cm-3).' )

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


def DedispersionCmd ( lowVal , numTry ) :
	global highDM
	if(lowVal+(dDM*numTry) > highDM):
		numTry = ((highDM - lowVal)/dDM)-1

	if (numTry < 0):
		return

	cmdLine = "prepsubband -noclip -psrfits -nobary -dmprec 4 -nsub 192 -dmstep {:.4f} -downsamp {:d} -lodm {:.4f} -numdms {:d} -mask {:s} -o search {:s}".format( dDM , ds , lowVal , numTry , args.mask , args.fits )
	return subprocess.check_output( cmdLine , shell=True )


def psrcatSearch ( psr ) :
	try:
		result = subprocess.check_output( "psrcat {:s} -c 'p0 tau_sc' -o long -nohead -nonumber".format( psr ) , shell=True )
	except:
		per = 0.01
		scatt = None

	if result.startswith( 'WARNING' ) :
		per = 1.0
		scatt = None
	else :
		per = result.split()[0]
		scatt = result.split()[3]

		if per == '*' :
			per = None
		else :
			per = float( per )

		if scatt == '*' :
			scatt = None
		else :
			scatt = float( scatt )

	return per , scatt


# Information extraction

psrName , fmin , fcent , fmax , chan_bw , tsamp , dm0 , blkLen, nblk = FitsInfo( args.fits )

try:
	period , scat = psrcatSearch( psrName )
except:
	period = 0.01
	scat = 0.001



if args.per :
	period = args.per

if dm0 - args.interv < args.dmmin :
	lowDM = args.dmmin
else :
	lowDM = dm0 - args.interv

global highDM
highDM = dm0 + args.interv

alow_dt = 4150.*(((fmin)**-2) - ((fmin+chan_bw)**-2))
DM_max = blkLen*tsamp/alow_dt

if (highDM > DM_max):
	highDM = DM_max-args.dmstep

# Optimal downsampling calculating

if scat :
	scat = scat * ( fcent / 1e3 )**( -4.4 )
else :
	scat = 0.05 * period

ds = int( 2 ** np.round( np.log2( 2 * scat / 30 / tsamp ) ) )

if scat > period :
	ds = int( 2 ** np.round( np.log2( 2 * period / 30 / tsamp ) ) )

if ds < args.dsmin :
	ds = args.dsmin

if ds > blkLen :
	ds = blkLen


# DM step computing

dDM = round( tsamp / 4.15e3 / ( 1/fmin**2 - 1/fmax**2 ) , 4 )	# Computing of the DM step based on the total bandwidth
if args.dmstep > dDM :
	dDMopt = dDM
	dDM = args.dmstep


# Number of DM to try computing

nbTotDMs = round ( ( highDM - lowDM ) / dDM , 0 )
if mp.cpu_count() < args.ncpus :
	nCPUs = mp.cpu_count()
else :
	nCPUs = args.ncpus

if ( nbTotDMs ) > args.nbmax :
	dDM = round( ( highDM - lowDM ) / ( args.nbmax ) , 4 )
	nbTotDMs = round ( ( highDM - lowDM ) / dDM , 0 )



DMperiter = 256
DMperiter = DMperiter - DMperiter%nCPUs


nbTotDMs = nbTotDMs + DMperiter - nbTotDMs%DMperiter
dDM = ( highDM - lowDM )/(nbTotDMs-1)

niter = np.ceil(nbTotDMs/DMperiter)


# Write the DD plan in an ASCII file

if args.out :

	print "\n\tWriting DD plan in the file :\t'{:s}.ddplan'.\n".format( args.out )

	_file0 = open( args.out + '.ddplan' , 'w' )

	_file0.write( "FITS file used :\t{:s}\n".format( args.fits ) )
	_file0.write( "Reference DM used :\t{:.4f} pc.cm-3\n".format( dm0 ) )
	_file0.write( "DM interval used :\t-{:.4f} -> +{:.4f} pc.cm-3\n".format( args.interv , args.interv ) )
	if (args.dmstep > dDM):
		if (dDMopt):
			_file0.write( "\nOptimal DM step :\t{:.4f} pc.cm-3".format( dDMopt ) )
	_file0.write( "\nDM step used :\t\t{:.4f} pc.cm-3\n".format( dDM ) )
	_file0.write( "Downsampling used :\t{:d}\n".format( ds ) )
	_file0.write( "Number of tried DMs :\t{:d}\n".format( int(DMperiter*niter) ) )

	_file0.write( "Scattering value used :\t{:.4f} s\n".format( scat ) )
	_file0.write( "Period value used :\t{:.4f} s\n".format( period ) )
	_file0.write( "Mask file used :\t{:s}\n".format( args.mask ) )
	_file0.write( "\nPrepsubband runs on {:d} CPUs with the following splitting :\n".format( nCPUs ) )

	_file0.write( "\nlow DM\tdDM\tds\tnb DM\n" )



# Prepsuband in multi-processing
loDM = [lowDM - dDM]
print "\n\tRunning prepsubband command.\n"
commande = ('mkdir single_pulse_mark;mkdir single_pulse' )
p = subprocess.Popen(commande, shell=True, stdout=subprocess.PIPE)
output, errors = p.communicate()
print errors,output
for ite in range(int(niter)):
    pool = mp.Pool()
    res = []
    loDM = np.linspace(loDM[-1]+dDM, loDM[-1]+DMperiter*dDM, nCPUs)
    nDMs = DMperiter/nCPUs
    for l in loDM :
        _file0.write( "{:.4f}\t{:.4f}\t{:d}\t{:d}\n".format( float(l) , float(dDM) , int(ds) , int(nDMs) ) )
    print("prepsubband: Iteration %d/%d for dm %.4f to %.4f with %.4f step " %(ite, int(niter), np.min(loDM), np.max(loDM), float(dDM)))
    for l in loDM :
        if (l < highDM):
            res.append( pool.apply_async( DedispersionCmd , ( l , nDMs) ) )
        else:
            continue
    for r in res :
        print r.get()
    print("single_pulse_search: Iteration %d/%d for dm %.4f to %.4f with %.4f step " %(ite, int(niter), np.min(loDM), np.max(loDM), float(dDM)))
    commande = ('python /home/lbondonneau/scripts/pav/psrfits_search/single_pulse_search/single_pulse_search_NenuFAR.py  --nobadblocks --noplot *.dat -o %d -n %d -c %d -t %d; mv *.singlepulse single_pulse_mark/; cp *.inf single_pulse_mark/' % (int(blkLen*nblk), int(nDMs), int(nCPUs), int(args.threshold)))
    p = subprocess.Popen(commande, shell=True, stdout=subprocess.PIPE)
    output, errors = p.communicate()
    commande = ('python /home/lbondonneau/scripts/pav/psrfits_search/single_pulse_search_mp.py --nobadblocks --noplot *.dat -c %d -t %d; mv *.singlepulse single_pulse/; cp *.inf single_pulse/; rm *.dat' % (int(nCPUs), int(args.threshold)))
    p = subprocess.Popen(commande, shell=True, stdout=subprocess.PIPE)
    output, errors = p.communicate()
    print errors,output
    print errors,output
_file0.close()
