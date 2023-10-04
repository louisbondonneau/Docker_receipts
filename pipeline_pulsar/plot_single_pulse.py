#                RESCALE TIME
#    Resize data array extracted from a FITS file to a new number of rows.
#    Used to decrease the time lapse of a block for a involve the rfifind processing.


# MODULES

import numpy as np
import pyfits as fi
import sys
import argparse as arg
import os
from multiprocessing import Pool, TimeoutError
import scipy.stats as stats


# ARGUMENTS LIST

parser = arg.ArgumentParser( description = 'transforme 32 bits data to a flatband 8 bits without scales and offsets.' )

parser.add_argument('-gui', dest='gui', action='store_true', default=False,
                    help="Open the matplotlib graphical user interface")
parser.add_argument( '-f' , dest='fileName' , type=str , help='Name of the FITS file to change.' )
parser.add_argument( '-df' , dest='df' , type=int , default=8, help='integration value in freq.' )
parser.add_argument( '-ds' , dest='ds' , default=False, help='integration value in freq.' )
parser.add_argument( '-chan_threshold' , dest='chan_threshold' , type=float , default=3, help='threshold of peak RFI cleaning (default 3).' )
parser.add_argument( '-time_threshold' , dest='time_threshold' , type=float , default=3, help='threshold of mean RFI cleaning (default 3).' )
parser.add_argument( '-goto_wind' , dest='goto_wind' , type=float , default=1.5, help='time window for the go to (default 3 sec).' )
parser.add_argument( '-u' , dest='dir' , type=str , default='./', help='directory of the output.' )
parser.add_argument( '-snr' , dest='snr' , type=float , default=6.0, help='snr limit.' )
parser.add_argument( '-nbest' , dest='nbest' , type=int , default=10, help='plot the N best pulsations.' )
parser.add_argument('singlepulse_files', nargs='+', help='.singlepulse files from presto')



args = parser.parse_args()


if (args.gui):
    import matplotlib
    import matplotlib.pyplot as plt
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def mad(data, axis='all'):
    if( axis=='all' ):
        data = np.nanmedian( np.abs( data - np.nanmedian(data) ) )
    else:
        if (axis != 0):
            axis_array = range(np.size(np.shape(data)))
            data = data.transpose(np.roll(axis_array, -axis))
        data = np.nanmedian( np.abs(data - np.ones(np.shape(data))*np.nanmedian(data, axis=0)), axis=0)
        if (axis != 0):
            axis_array = range(np.size(np.shape(data)))
            data = data.transpose(np.roll(axis_array, axis))
    return data

def extract_data_array(fileName, minrow=0, maxrow=False):
    data = fi.getdata( fileName , do_not_scale_image_data=True , scale_back=True)            # Extraction of the data arrays
    head = fi.getheader( fileName , 1 , do_not_scale_image_data=True , scale_back=True )
    if not (maxrow):
        nline = int(head[ 'NAXIS2' ])
        maxrow = nline
    old_offset = data.field( 14 ).astype('float32')[minrow:maxrow] 
    old_scale = data.field( 15 ).astype('float32')[minrow:maxrow] 
    old_data = data.field( 16 )[minrow:maxrow, :, :, :, 0]                    # Copy of the old amplitude data array
    ##### calculate constantes
    nline, line_lenght, npol, nchan = np.shape(old_data[:, :, :, :])
    ##### extract values
    old_scale = np.resize(old_scale,(nline, npol, nchan))
    old_offset = np.resize(old_offset,(nline, npol, nchan))
    old_data = np.resize(old_data,(nline, line_lenght, npol, nchan, 1))
    if (np.mean(old_scale) != 1) or (np.sum(old_offset) != 0):
        for bin in range(line_lenght) :
            old_data[:, bin, :, :, 0] = (old_data[:, bin, :, :, 0]*old_scale + old_offset)
    return old_data

def clean_data_intergrated(old_data):
    time_serie = np.nanmean(old_data, axis=3) #(1, line_lenght*nline, npol, nchan, 1)
    std_spectrum = np.std(old_data, axis=1)
    std_freq = 1.48*mad(std_spectrum[0,0,:,0])
    std_time = 1.48*mad(time_serie[0,:,0,0])
    median = np.median(time_serie)
    ind_bad_time = np.where((np.abs(time_serie[0,:,0,0] - median) > std_time*float(args.time_threshold)))

    old_data[0, ind_bad_time[0][:], :, :, 0] = np.nan

    ind_bad_chan = np.where((std_spectrum[0,0,:,0]-np.nanmedian(std_spectrum[0,0,:,0]) > std_freq*float(args.chan_threshold)))

    old_data[0, :, :, ind_bad_chan[0][:], 0] = np.nan

    #
    #print('peak', np.shape(ind_peak))
    #print('mean', np.shape(ind_mean))
    ##print(np.median(old_data), np.std(old_data))
    #for i in range(len(ind_peak[0])):
    #    noise = gaussian_noise(shape=[int(line_lenght), int(npol)], std=1., median=0.)
    #    old_data[int(ind_peak[0][i]), :, :, int(ind_peak[1][i]), 0] = noise
    #for i in range(len(ind_mean[0])):
    #    noise = gaussian_noise(shape=[int(line_lenght), int(npol)], std=1., median=0.)
    #    old_data[int(ind_mean[0][i]), :, :, int(ind_mean[1][i]), 0] = noise
    return old_data

# CHECKING INPUT PARAMETERS

if os.path.isfile( args.fileName ) :        # Checking file existence
    print('\nExtraction of data from {:s}.\n'.format( args.fileName ))
else :
    print('\n{:s} is not a file.\n'.format( args.fileName ))
    sys.exit()


def gaussian_noise(shape, std=1, median=0, noise_min='None', noise_max='None'):
    data = std*np.random.randn(*shape) + median
    cont = 0
    while ((noise_min != 'None') or (noise_max != 'None')) or (cont < 1000):
        cont += 1
        if(noise_min != 'None'):
            min_ind = (data < noise_min)
            size = np.sum(min_ind)
            if (size > 0):
                noise  = std*np.random.randn(*[size]) + median
                data[min_ind] = noise
            else:
                noise_min = 'None'
        if(noise_max != 'None'):
            max_ind = (data >= noise_max)
            size = np.sum(max_ind)
            if (size > 0):
                noise = std*np.random.randn(*[size]) + median
                data[max_ind] = noise
            else:
                noise_max = 'None'
    return data

#test = gaussian_noise(shape=[int(10000000), int(1)], std=42.5, median=85.0 - 1.28, noise_min=0, noise_max=255)
#print(np.median(test))
#print(np.std(test))
#print(mad(test, axis='all'))
##print(np.shape(test))
#import matplotlib
#import matplotlib.pyplot as plt
#plt.hist(test)
#plt.show()
# DATA EXTRACTION OF THE PREVIOUS FITS
def plot_single_pulse(goto, dm, ds=1, df=4, goto_wind=1.5):
    global DATA_mem
    df = int(2**(np.floor(np.log(df)/np.log(2))))
    ds = int(2**(np.floor(np.log(ds)/np.log(2))))
    headObs = fi.getheader( args.fileName , 0 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the observation header
    head = fi.getheader( args.fileName , 1 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the data header
    data = fi.getdata( args.fileName , do_not_scale_image_data=True , scale_back=True )            # Extraction of the data arrays
    
    chan_bw = float(head[ 'CHAN_BW' ])
    NBITS = head[ 'NBITS' ]
    
    DAT_FREQ = data.field( 12 ).astype('float32') 
    freq_vec = DAT_FREQ[0,:]
    minfreq = np.min(DAT_FREQ) - chan_bw/2
    maxfreq = np.max(DAT_FREQ) + chan_bw/2
    TSUBINT = data.field( 0 ).astype('float32') 
    OFFS_SUB = data.field( 1 ).astype('float32') 
    LST_SUB = data.field( 2 ).astype('float32')
    
    if (NBITS == '32'):
        dtype = np.float32
    elif (NBITS == '16'):
        dtype = np.float16
    else:
        dtype = np.uint8
    
    NROW = np.shape(data)[0]
    npol = int(head['NPOL'])
    
    nline = int(head[ 'NAXIS2' ])
    line_lenght = int(head[ 'NSBLK' ])
    npol = int(head['NPOL'])
    nchan = int(head[ 'NCHAN' ])
    tbin = float(head['TBIN'])
    
    length = TSUBINT[0]*nline

    if not (os.path.isfile(args.dir+'tmp_memmapped_'+args.fileName)):
        DATA_mem = np.memmap(args.dir+'tmp_memmapped_'+args.fileName, dtype=np.float16, mode='w+', shape=(nline, line_lenght, npol, nchan, 1))
        DATA_mem = extract_data_array(args.fileName).astype('float16')
    #global DATA
    time_off = 0
    minrow = False
    maxrow = False
    if(goto_wind < ds*tbin*36): goto_wind = ds*tbin*36
    if (goto):
        dt_bin = 0
        if (dm > 0):
            dt = dm*4150*(minfreq**-2 -  maxfreq**-2.)
            dt_bin = np.ceil(dt/tbin)
        minrow = int(np.floor((goto - tbin*dt_bin - goto_wind/2)/tbin/line_lenght))
        maxrow = int(np.ceil((goto + tbin*dt_bin + goto_wind/2)/tbin/line_lenght))
        if(minrow < 0): minrow = 0
        if(maxrow > nline): minrow = nline
        nline = int(maxrow-minrow)
        time_off +=  - minrow*tbin*line_lenght
    
    DATA = DATA_mem[minrow:maxrow, :, :, :, ]
    
    DATA = np.reshape(DATA,(1, line_lenght*nline, npol, nchan, 1))
    line_lenght = line_lenght*nline
    nline = 1
    if (ds>1):
        print('---------time-integration---------')
        DATA = np.resize(DATA,(nline, line_lenght/ds, ds, npol, nchan, 1))
        DATA = np.nanmean(DATA, axis=2)
        DATA = np.resize(DATA,(nline, line_lenght/ds, npol, nchan, 1))
        line_lenght = line_lenght/ds
        tbin = tbin*ds
    
    
    
    print(line_lenght, tbin)
    
    nline = 1
    
    fig = plt.figure(figsize=(5, 4))
    plt.subplots_adjust(top=0.97,
                        bottom=0.12,
                        left=0.12,
                        right=0.980,
                        hspace=0,
                        wspace=0.25)
    
    ax0 = plt.subplot2grid((4, 1), (1, 0), colspan=4, rowspan=3)
    ax1 = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=1, sharex=ax0)
    
    #ax1 = plt.subplot2grid((5, 5), (3, 3), colspan=1, rowspan=1, sharex=ax2, sharey=ax8)
    #DATA = extract_data_array(args.fileName, minrow=False, maxrow=False).astype('uint8')
    DATA = clean_data_intergrated(DATA)
    DATA = DATA[0,:, 0, :, 0]
    print(np.shape(DATA))
    
    if (dm > 0):
        dt = dm*4150*(minfreq**-2 -  maxfreq**-2.)
        dt_bin = np.ceil(dt/tbin)
        print(dt, tbin, dt_bin)
        time_off += int(dt_bin)*tbin
        pad = np.nan*np.zeros([int(dt_bin), int(nchan)])
        DATA = np.concatenate((pad, DATA), axis=0)
        line_lenght += dt_bin
        for ifreq in range(len(freq_vec)):
            dt = dm*4150*(freq_vec[ifreq]**-2 -  maxfreq**-2.)
            dt_bin = np.ceil(dt/tbin)
            DATA[:, ifreq] = np.roll(DATA[:, ifreq], -int(dt_bin))
    
    print(np.shape(DATA))
    
    if (df>1):
        print('---------freq-integration---------')
        DATA = np.resize(DATA,(int(line_lenght), int(nchan/df), int(df)))
        DATA = np.nanmean(DATA, axis=2)
        DATA = np.resize(DATA,(int(line_lenght), int(nchan/df)))
        nchan = int(nchan/df)
        chan_bw = chan_bw*df
        freq_vec = np.resize(freq_vec,(int(nchan/df), int(df)))
        freq_vec = np.nanmean(freq_vec, axis=1)
        #freq_vec = freq_vec[:,0]
    
    #replace nan by noise
    std_val = 1.48*mad(DATA)
    median = np.nanmedian(DATA)
    ind_replace = np.where(np.isnan(DATA) == True)
    DATA[ind_replace] =  std_val*np.random.randn(np.sum(np.isnan(DATA))) + median
    
    print(np.shape(DATA))
    ax0.imshow(np.rot90(DATA), interpolation='none', cmap='afmhot',
                  extent=[-time_off, line_lenght*tbin-time_off, minfreq, maxfreq], aspect='auto')
    ax1.plot(np.linspace(-time_off,line_lenght*tbin-time_off, line_lenght), np.nanmean(DATA, axis=1))
    ax1.axes.get_xaxis().set_visible(False)
    if (goto):
        ax0.set_xlim([goto - goto_wind/2,  goto + goto_wind/2])
    ax0.text(goto + 0.20*goto_wind/2, minfreq+0.10*(maxfreq-minfreq), str(dm)+r" pc cm-3", size=13)
    print("tbin = %.2f sec" % tbin) 
    ax0.set_xlabel('Time [sec]')
    ax0.set_ylabel('Frequency [MHz]')
    ax1.set_ylabel('Amplitude [AU]')
    
    if (goto):
        png_name = args.dir+os.path.basename(args.fileName).split('.')[0]+'_'+str(int(goto))+'sec.png'
    else:
        png_name = args.dir+os.path.basename(args.fileName).split('.')[0]+'.png'
    if (args.gui):
        plt.show()
    plt.savefig(png_name, dpi=128, format='png')

dm_liste = []
sigma_liste = []
goto_liste = []
ds_liste = []
for ifile in range(len(args.singlepulse_files)):
    result = np.loadtxt(args.singlepulse_files[ifile],unpack=True)
    if (np.size(result) == 0):
        continue
    else:
        print(args.singlepulse_files[ifile])
    dm_liste = np.append(dm_liste, result[0])
    sigma_liste = np.append(sigma_liste, result[1])
    goto_liste = np.append(goto_liste, result[2])
    ds_liste = np.append(ds_liste, result[4])




ind = np.where(sigma_liste > args.snr)
dm_liste = dm_liste[ind]
sigma_liste = sigma_liste[ind]
goto_liste = goto_liste[ind]
ds_liste = ds_liste[ind]

print(goto_liste)

if (len(goto_liste) == 0):
    exit(0)

head = fi.getheader( args.fileName , 1 , do_not_scale_image_data=True , scale_back=True )        # Extraction of the data header
print(head)
nline = int(head[ 'NAXIS2' ])
line_lenght = int(head[ 'NSBLK' ])
tbin = float(head['TBIN'])
duration = nline*line_lenght*tbin

best_pulse = goto_liste[np.argmax(sigma_liste)]
time_search = (best_pulse%args.goto_wind) - args.goto_wind

wind_dm_liste = []
wind_sigma_liste = []
wind_goto_liste = []
wind_ds_liste = []

while (time_search < duration+args.goto_wind):
    cand = np.where( (time_search<goto_liste) & (goto_liste<(time_search+args.goto_wind)) )
    if(np.size(cand) > 0):
        cand_dm_liste = dm_liste[cand]
        cand_sigma_liste = sigma_liste[cand]
        cand_goto_liste = goto_liste[cand]
        cand_ds_liste = ds_liste[cand]
        best_cand = np.argmax(cand_sigma_liste)
        wind_dm_liste = np.append(wind_dm_liste, cand_dm_liste[best_cand])
        wind_sigma_liste = np.append(wind_sigma_liste, cand_sigma_liste[best_cand])
        wind_goto_liste = np.append(wind_goto_liste, cand_goto_liste[best_cand])
        wind_ds_liste = np.append(wind_ds_liste, cand_ds_liste[best_cand])
    time_search += args.goto_wind

if (len(wind_sigma_liste) < args.nbest): args.nbest = len(wind_sigma_liste)
nbest = wind_sigma_liste.argsort()[-args.nbest:][::-1]
wind_dm_liste = wind_dm_liste[nbest]
wind_sigma_liste = wind_sigma_liste[nbest]
wind_goto_liste = wind_goto_liste[nbest]
wind_ds_liste = wind_ds_liste[nbest]

if (os.path.isfile(args.dir+'tmp_memmapped_'+args.fileName)):
    os.remove(args.dir+'tmp_memmapped_'+args.fileName)
for icand in range(len(wind_dm_liste)):
    if (args.ds): wind_ds_liste[icand] = int(args.ds)
    plot_single_pulse(wind_goto_liste[icand], wind_dm_liste[icand], ds=wind_ds_liste[icand], df=args.df, goto_wind=args.goto_wind)
os.remove(args.dir+'tmp_memmapped_'+args.fileName)
