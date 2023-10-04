#                RESCALE TIME
#    Resize data array extracted from a FITS file to a new number of rows.
#    Used to decrease the time lapse of a block for a involve the rfifind processing.


# MODULES

import numpy as np
import numpy.ma as ma
import pyfits as fi
import sys
import argparse as arg
import os
from multiprocessing import Pool

# import scipy.stats as stats
# import matplotlib.pyplot as plt


# ARGUMENTS LIST

parser = arg.ArgumentParser(description='transforme 32 bits data to a flatband 8 bits without scales and offsets.')

parser.add_argument('-f', dest='fileName', type=str, help='Name of the FITS file to change.')
parser.add_argument('-o', dest='newFileName', type=str, help='Name of the new FITS file to write.')
parser.add_argument('-j', dest='ncore', help='number of core for the multiprocessing', default=8)
parser.add_argument('-ds', dest='ds', type=int, default=1, help='downsample value.')
parser.add_argument('-subblock', dest='subblock', type=float, help='number of subblock per block (default 1)', default=1)
parser.add_argument('-pscrunch', dest='pscrunch', action='store_true', default=False, help="scrunch the polarisation")
parser.add_argument('-intscales', dest='intscales', action='store_true', default=False,
                    help="used 8 bits scales and offset (replace 32 bits float bu 8 bit uint)")
parser.add_argument('-noscale', dest='noscale', action='store_true', default=False, help="force all scales to 1")
parser.add_argument('-notimevar', dest='notimevar', action='store_true', default=False,
                    help="do not take in count the time dependency of the offset and the scale")
parser.add_argument('-threshold', dest='threshold', type=int, default=6, help='Change the threshold value (default threshold = 6).')
parser.add_argument('-plot', dest='plot', action='store_true', default=False, help="plot statistics")
parser.add_argument('-flat_per_block', dest='flat_per_block', action='store_true', default=False, help="flat each block chan/int independently")
parser.add_argument('-dm_zero', dest='dm_zero', action='store_true', default=False, help="deduce the median per integrations")
parser.add_argument('-clean', dest='clean', action='store_true', default=False, help="clean file replacing RFI with gaussian noise")
parser.add_argument('-clean_with_zero', dest='clean_with_zero', action='store_true', default=False, help="clean file replacing RFI with zeros")
parser.add_argument('-no_dm0_clean', dest='no_dm0_clean', action='store_true', default=False, help="do not clean the integrated dm0 time serie")
parser.add_argument('-rmbase', dest='rmbase', action='store_true', default=False, help="remove the baseline")
parser.add_argument('-mean_threshold', dest='mean_threshold', type=float, default=5, help='threshold of RFI cleaning (default 3).')


args = parser.parse_args()


if(args.clean_with_zero):
    args.clean = True
# if(args.clean):
#     args.flat_per_block = True


if (os.path.dirname(args.newFileName) == '.'):
    args.newFileName = os.getcwd() + '/' + os.path.basename(args.newFileName)

ds = int(2**(round(np.log(args.ds) / np.log(2))))


def mad(data, axis='all'):
    if(axis == 'all'):
        data = np.nanmedian(np.abs(data - np.nanmedian(data)))
    else:
        if (axis != 0):
            axis_array = range(np.size(np.shape(data)))
            data = data.transpose(np.roll(axis_array, -axis))
        data = np.nanmedian(np.abs(data - np.ones(np.shape(data)) * np.nanmedian(data, axis=0)), axis=0)
        if (axis != 0):
            axis_array = range(np.size(np.shape(data)))
            data = data.transpose(np.roll(axis_array, axis))
    return data


def data_to_offsets_and_scales(old_data):
    ds = int(2**(round(np.log(args.ds) / np.log(2))))
    SIGMA = float(args.threshold)
    SIGMA = SIGMA * (2. / 3)
    # calculate constantes
    nline, line_lenght, npol, nchan = np.shape(old_data[:, :, :, :, 0])
    print("nline = %d npol = %d" % (nline, npol))

    # calcul des std et median

    print('---------calculate median_array---------')
    median_array = np.median(old_data, axis=1)  # OFFSET

    if not (args.noscale):
        print('---------calculate std_array---------')
        std_array = np.std(old_data, axis=1)        # SCAL
    else:
        std_array = 0 * median_array

    if (args.notimevar):
        print('---------rermoving time variation in OFFSET and SCAL---------')
        print(np.shape(median_array))
        print(np.shape(std_array))
        mean_median_array = np.median(median_array, axis=0)
        mean_std_array = np.median(std_array, axis=0)
        for line in range(nline):
            median_array[line, :, :, :] = mean_median_array
            std_array[line, :, :, :] = mean_std_array

    OFFSET = median_array - 0.5 * SIGMA * std_array
    # The signal is between median_array-0.5*SIGMA*std and median_array+1.5*SIGMA*std

    SCAL = 2. * SIGMA * std_array / 256.

    if (args.intscales):
        print('---------used intscales---------')
        saturation = np.where(OFFSET > 255)
        SCAL[saturation] = (OFFSET[saturation] - 255 + 2. * SIGMA * std_array[saturation]) / 256.
        SCAL = np.ceil(SCAL)

        OFFSET[OFFSET > 255] = 255
        OFFSET[OFFSET < 0] = 0

        #SCAL = np.ceil(SCAL)
        SCAL[SCAL > 255] = 255
        SCAL[SCAL < 1] = 1

        OFFSET = OFFSET.astype('uint8')  # cast OFFSET matrix in a uint8 matrix
        SCAL = SCAL.astype('uint8')  # cast SCAL matrix in a uint8 matrix

    # some plots
    if (args.plot):
        print('---------make plot flat-std-median.png---------')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(top=0.98,
                            bottom=0.07,
                            left=0.1,
                            right=0.980,
                            hspace=0.215,
                            wspace=0.25)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        for i in range(npol):
            mean_med = np.mean(median_array[:, i, :], axis=0)
            ax1.semilogy(mean_med)
            ax1.set_xlabel('channel number')
            ax1.set_ylabel('median value')
            bins = np.logspace(np.log10(1), np.log10(np.max(mean_med)), 32)
            if(np.max(mean_med) < 10):
                bins = np.logspace(np.log10(1), np.log10(10), 32)
            ax3.hist(mean_med, bins=bins, alpha=0.3, log=True)
            ax3.set_xscale("log")
            ax3.set_xlabel('median value')
            ax3.set_ylabel('number of value')
        for i in range(npol):
            mean_std = np.mean(std_array[:, i, :], axis=0)
            ax2.semilogy(mean_std)
            ax2.set_xlabel('channel number')
            ax2.set_ylabel('standard deviation value')
            bins = np.logspace(np.log10(1), np.log10(np.max(mean_std)), 32)
            if(np.max(mean_std) < 10):
                bins = np.logspace(np.log10(1), np.log10(10), 32)
            ax4.hist(mean_std, bins=bins, alpha=0.3, log=True)
            ax4.set_xscale("log")
            ax4.set_xlabel('std')
            ax4.set_ylabel('number of value')
        plt.savefig('flat-std-median.png')

    # some plots
    if (args.plot):
        print('---------make plot flat-scal-offset.png---------')
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(top=0.98,
                            bottom=0.07,
                            left=0.1,
                            right=0.980,
                            hspace=0.215,
                            wspace=0.25)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        for i in range(npol):
            mean_scal = np.mean(SCAL[:, i, :, 0], axis=0)
            ax2.semilogy(mean_scal)
            if (args.intscales):
                ax2.axhline(256, color="r")
            ax2.set_xlabel('channel number')
            ax2.set_ylabel('scal')
            bins = np.logspace(np.log10(1), np.log10(np.max(mean_scal)), 32)
            if(np.max(mean_scal) < 10):
                bins = np.logspace(np.log10(1), np.log10(10), 32)
            ax4.hist(mean_scal, bins=bins, alpha=0.3, log=True)
            ax4.set_xscale("log")
            ax4.set_xlabel('scal')
            ax4.set_ylabel('number of value')
        for i in range(npol):
            mean_offset = np.mean(OFFSET[:, i, :, 0], axis=0)
            ax1.semilogy(mean_offset)
            if (args.intscales):
                ax1.axhline(256, color="r")
            ax1.set_xlabel('channel number')
            ax1.set_ylabel('offset')
            bins = np.logspace(np.log10(1), np.log10(np.max(mean_offset)), 32)
            if(np.max(mean_offset) < 10):
                bins = np.logspace(np.log10(1), np.log10(10), 32)
            ax3.hist(mean_offset, bins=bins, alpha=0.3, log=True)
            ax3.set_xscale("log")
            ax3.set_xlabel('offset')
            ax3.set_ylabel('number of value')
        plt.savefig('flat-scal-offset.png')
    #

    # some plots
    if (args.plot):
        print('---------make plot data.png---PART1------')
        plt.clf()
        spectrum = np.mean(median_array, axis=0)
        fig = plt.figure(figsize=(8, 8))
        plt.subplots_adjust(top=0.98,
                            bottom=0.07,
                            left=0.1,
                            right=0.980,
                            hspace=0.215,
                            wspace=0.25)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        pol = ['XX', 'YY', 'XY', 'YX']
        for ipol in range(npol):
            ax1.semilogy(spectrum[ipol, :, 0], label=pol[ipol])
        ax1.set_xlabel('channel')
        ax1.set_ylabel('OLD Amplitude (AU)')
        ax1.legend(loc='upper right')
        ax3.hist(np.resize(old_data, len(old_data)), alpha=1, log=True)
        ax3.set_xlabel('OLD values')
        ax3.set_ylabel('number of value')

    print('---------apply offset and scaling---------')
    # apply offset and scalingine*ipol*ichan, nline*npol*nchan, prefix = 'Progress:', suffix = 'Complete', barLength = 50)

    for bin in range(line_lenght):
        old_data[:, bin, :, :, :] = (old_data[:, bin, :, :, :] - OFFSET) / SCAL

    if (args.plot):
        print('---------make plot data.png---PART2------')
        spectrum = np.median(old_data, axis=1)
        spectrum = np.mean(spectrum, axis=0)
        for ipol in range(npol):
            ax2.semilogy(spectrum[ipol, :, 0], label=pol[ipol])
        if (args.intscales):
            ax2.axhline(256, color="r")
        ax2.set_xlabel('channel')
        ax2.set_ylabel('NEW Amplitude (AU)')
        ax2.legend(loc='upper right')
        ax4.hist(np.resize(old_data, len(old_data)), alpha=1, log=True)
        ax4.set_xlabel('NEW values')
        ax4.set_ylabel('number of value')
        plt.savefig('oldDATA_newDATA.png')

    OFFSET = np.resize(OFFSET, (nline, npol, nchan))
    SCAL = np.resize(SCAL, (nline, npol, nchan))

    return (old_data, SCAL, OFFSET)


def extract_data_array(fileName, minrow=False, maxrow=False):
    global DATA
    data = fi.getdata(fileName, do_not_scale_image_data=True, scale_back=True)            # Extraction of the data arrays
    old_offset = data.field(14).astype('float32')[minrow:maxrow]
    old_scale = data.field(15).astype('float32')[minrow:maxrow]
    print(np.shape(DATA))
    old_data = DATA[minrow:maxrow, :, :, :, 0]                    # Copy of the old amplitude data array
    # calculate constantes
    nline, line_lenght, npol, nchan = np.shape(old_data[:, :, :, :])
    # extract values
    old_scale = np.resize(old_scale, (nline, npol, nchan))
    old_offset = np.resize(old_offset, (nline, npol, nchan))
    old_data = np.resize(old_data, (nline, line_lenght, npol, nchan, 1))
    for bin in range(line_lenght):
        old_data[:, bin, :, :, 0] = (old_data[:, bin, :, :, 0] * old_scale + old_offset)
    return old_data


def clean_data(old_data, mask_array, dm_zero=False):
    nline, line_lenght, npol, nchan, i = np.shape(old_data)  # (nline, line_lenght, npol, nchan, 1)
    if (np.sum(mask_array > 0)):
        print('--------generating normal white noise to replace RFI-------')
        std_val = 255. / (float(args.threshold))
        median_val = 255 * (1. / 3)
        RFI_ratio = float(np.sum(~mask_array)) / float(np.size(mask_array))
        print("mask %.2f percent of the array" % (100. * RFI_ratio))
        bad_ind = np.where(mask_array == False)
        for ipol in range(npol):
            noise_clean = np.zeros(len(bad_ind[2])).astype('uint8')
            for i in range(int(np.ceil(len(bad_ind[2]) / 8192))):
                if ((i + 1) * 8192 < len(bad_ind[2])):
                    noise_clean[i * 8192:(i + 1) * 8192] = np.round(gaussian_noise(shape=[int(8192)], std=std_val, median=median_val,
                                                                                   noise_min=0, noise_max=255)).astype('uint8')
                else:
                    # np.round(std_val*np.random.randn(len(bad_ind[2])-i*8192)+ median_val).astype('uint8')
                    noise_clean[i * 8192:] = np.round(gaussian_noise(shape=[int(len(bad_ind[2]) - i * 8192)], std=std_val,
                                                                     median=median_val, noise_min=0, noise_max=255)).astype('uint8')
            old_data[bad_ind[0], bad_ind[1], ipol, bad_ind[2], 0] = noise_clean
        del noise_clean

    print('--------clean-start-------')
    if (npol > 1):
        toclean_data = old_data.astype('uint16')
        toclean_data = ((toclean_data[:, :, 0, :, 0] + old_data[:, :, 1, :, 0]) / 2).astype('uint8')
    else:
        toclean_data = np.copy(old_data)
        toclean_data = toclean_data[:, :, 0, :, 0]

    print('--------mask calculation-------')

    std_array = np.std(toclean_data, axis=1)
    std_array = (std_array - np.median(std_array)) / (mad(std_array, axis='all') * 1.48)
    mean_array = np.mean(toclean_data, axis=1)
    mean_array = (mean_array - np.median(mean_array)) / (mad(mean_array, axis='all') * 1.48)

    ind_mean = np.where(mean_array > args.mean_threshold)
    ind_std = np.where(std_array > args.mean_threshold)
    # print(np.shape(ind_mean))
    # print(np.shape(ind_std))
    mask_array[ind_mean[0], :, ind_mean[1]] = False
    mask_array[ind_std[0], :, ind_std[1]] = False

    std_spectrum = np.zeros(nchan).astype('float32')
    time_serie = np.zeros(line_lenght * nline).astype('float32')

    for ichan in range(nchan):  # because np.std with axis=X force to use float64 when use a masked_array (extremely memorivore)
        toclean_data_tmp = ma.masked_array(toclean_data[:, :, ichan], mask=~mask_array[:, :, ichan])
        std_spectrum[int(ichan)] = np.std(toclean_data_tmp, dtype=np.float32)  # (line_lenght, nchan)
    if not (args.no_dm0_clean):
        for iline in range(nline):  # because np.mean with axis=X force to use float64 when use a masked_array (extremely memorivore)
            toclean_data_tmp = ma.masked_array(toclean_data[iline, :, :], mask=~mask_array[iline, :, :])
            time_serie[iline * line_lenght:(iline + 1) * line_lenght] = np.mean(toclean_data_tmp, axis=1)
    # for iline in range(nline): # because np.mean with axis=X force to use float64 when use a masked_array (extremely memorivore)
    #    for line_ind in range(line_lenght):
    #        time_serie[iline*line_lenght +line_ind] = np.mean(toclean_data[iline, line_ind, :][mask_array[iline, line_ind, :]], dtype=np.float32)
    #        if (np.isnan(time_serie[iline*line_lenght +line_ind])):
    #            toclean_data_tmp = np.copy(toclean_data[iline, line_ind, :])
    #            time_serie[iline*line_lenght +line_ind] = np.mean(toclean_data_tmp, dtype=np.float32)
    std_spectrum = np.abs(std_spectrum - np.median(std_spectrum))
    if not (args.no_dm0_clean):
        time_serie = np.abs(time_serie - np.median(time_serie))
    std_freq = 1.48 * mad(std_spectrum, axis='all')
    if not (args.no_dm0_clean):
        std_time = 1.48 * mad(time_serie, axis='all')
    # print(std_time*float(args.mean_threshold))
    # print(std_freq*float(args.mean_threshold))
    # plt.plot(time_serie - np.median(time_serie))
    # plt.plot(std_spectrum-np.median(std_spectrum))
    # plt.show()   no_dm0_clean

    if not (args.no_dm0_clean):
        ind_bad_time = np.where((time_serie - np.median(time_serie)) > std_time * float(args.mean_threshold))
    ind_bad_chan = np.where((std_spectrum - np.median(std_spectrum)) > std_freq * float(args.mean_threshold))

    mask_array = np.reshape(mask_array, (nline * line_lenght, nchan))
    if not (args.no_dm0_clean):
        mask_array[ind_bad_time, :] = False
    mask_array[:, ind_bad_chan] = False

    # import matplotlib
    # import matplotlib.pyplot as plt
    # plt.imshow(np.rot90(mask_array.astype('uint8')), aspect='auto')
    # plt.show()

    mask_array = np.reshape(mask_array, (nline, line_lenght, nchan))

    if(dm_zero):
        for ipol in range(npol):
            for iline in range(nline):
                median = np.median(old_data[iline, :, ipol, :, 0], axis=1)  # (nline, line_lenght, npol, nchan, 1)
                for ichan in range(nchan):
                    old_data[iline, :, ipol, ichan, 0] = old_data[iline, :, ipol, ichan, 0] + 85 - median
    return old_data, mask_array

# CHECKING INPUT PARAMETERS


if os.path.isfile(args.fileName):        # Checking file existence
    print('\nExtraction of data from {:s}.\n'.format(args.fileName))
else:
    print('\n{:s} is not a file.\n'.format(args.fileName))
    sys.exit()


if args.newFileName:                # Define the name of the new FITS file
    print('Scaled Integer arrays writed in {:s}.\n'.format(args.newFileName))
else:
    print('None new FITS file name defined. Default name used : new_{:s}.\n'.format(args.fileName))


def flat_bunch_row(minrow, maxrow):
    print("processe row %d to %d" % (minrow, maxrow))
    global DATA
    old_data = extract_data_array(args.fileName, minrow=minrow, maxrow=maxrow)
    ds = int(2**(round(np.log(args.ds) / np.log(2))))
    # calculate constantes
    nline, line_lenght, npol, nchan = np.shape(old_data[:, :, :, :, 0])

    if(args.pscrunch and npol > 1):
        print('---------pscrunch---------')
        old_data = np.sum(old_data[:, :, 0:1, :, :], axis=2)
        old_data = np.resize(old_data, (nline, line_lenght, 1, nchan, 1))
        npol = 1

    if (ds > 1):
        print('---------downsample---------')
        old_data = np.resize(old_data, (int(nline), int(line_lenght / ds), int(ds), int(npol), int(nchan), 1))
        old_data = np.sum(old_data, axis=2)
        old_data = np.resize(old_data, (int(nline), int(line_lenght / ds), int(npol), int(nchan), 1))
        line_lenght = line_lenght / ds

    if (int(args.subblock) > 1):
        print('---------reduce bloc size---------')
        old_data = np.reshape(old_data, (int(nline * int(args.subblock)), int(line_lenght / int(args.subblock)), int(npol), int(nchan), 1))
        line_lenght = line_lenght / int(args.subblock)
        nline = nline * int(args.subblock)

    if (args.subblock < 1):
        print('---------increas bloc size---------')
        if (nline % (1 / args.subblock) > 0):
            print('this should never append!!!!!!!')
            exit(0)
        old_data = np.reshape(old_data, (int(nline * args.subblock), int(line_lenght / args.subblock), npol, nchan, 1))
        line_lenght = int(float(line_lenght) / float(args.subblock))
        nline = int(float(nline) * float(args.subblock))
    print(nline, line_lenght)

    # calcul des std et median
    OFFSET = np.zeros([nline, npol, nchan])
    SCAL = np.ones([nline, npol, nchan])
    # nline, line_lenght, npol, nchan
    if (args.flat_per_block):
        print('---------flat_per_block with median and mad-----')
        for ipol in range(npol):
            for iline in range(nline):
                for ichan in range(nchan):
                    old_data[iline, :, ipol, ichan, :] = old_data[iline, :, ipol, ichan, :] - np.mean(old_data[iline, :, ipol, ichan, :])
                    mad = np.median(np.abs(old_data[iline, :, ipol, ichan, :]))
                    if (mad == 0):
                        mad = np.std(old_data[iline, :, ipol, ichan, :]) / 1.48
                    if (mad != 0):
                        old_data[iline, :, ipol, ichan, :] /= (1.48 * mad)
                    else:
                        old_data[iline, :, ipol, ichan, 0] = np.random.normal(loc=0.0, scale=1, size=line_lenght)
        old_data *= 255. / (float(args.threshold))
        old_data += 255 * (1. / 3)
        old_data[old_data > 255] = 255
    else:
        print('------recalculate scale and offset-----')
        for ipol in range(npol):
            for ichan in range(nchan):
                for iline in range(nline):
                    mean = np.mean(old_data[iline, :, ipol, ichan, :])
                    mad = np.median(np.abs(old_data[iline, :, ipol, ichan, :]))
                    if (mad == 0):
                        mad = np.std(old_data[iline, :, ipol, ichan, :]) / 1.48
                    # if (ipol < 2):
                    old_data[iline, :, ipol, ichan, :] = old_data[iline, :, ipol, ichan, :] - mean
                    old_data[iline, :, ipol, ichan, :] /= (1.48 * mad)

                    OFFSET[iline, ipol, ichan] += mean
                    OFFSET[iline, ipol, ichan] /= 255. / (float(args.threshold)) / (1.48 * mad)
                    OFFSET[iline, ipol, ichan] -= (255 * (1. / 3))
                    SCAL[iline, ipol, ichan] /= 255. / (float(args.threshold)) / (1.48 * mad)
        old_data *= 255. / (float(args.threshold))
        old_data += 255 * (1. / 3)
        old_data[old_data > 255] = 255

    std = 1.48 * np.median(np.abs(old_data - np.median(old_data)))
    #if not (args.flat_per_block):
    #    median = np.median(old_data)
    #    old_data -= median
    #    old_data *= 255. / (float(args.threshold)) / std
    #    old_data += 255 * (1. / 3)
    #    old_data[old_data > 255] = 255
    #    OFFSET += median
    #    OFFSET /= 255. / (float(args.threshold)) / std
    #    OFFSET -= (255 * (1. / 3))
    #    SCAL /= 255. / (float(args.threshold)) / std
    #    std = 1.48 * np.median(np.abs(old_data - np.median(old_data)))





    std_val = 255. / (float(args.threshold))
    median_val = 255 * (1. / 3)
    bad_ind = np.where((old_data < 0))
    if (args.clean_with_zero):
        for ipol in range(npol):
            noise_saturation = np.zeros(len(bad_ind[2]))
            for i in range(int(np.ceil(len(bad_ind[2]) / 8192))):
                if ((i + 1) * 8192 < len(bad_ind[2])):
                    noise_saturation[i * 8192:(i + 1) * 8192] = np.zeros(int(8192))  # np.round(std_val*np.random.randn(8192)+ median_val).astype('uint8')
                else:
                    # np.round(std_val*np.random.randn(len(bad_ind[2])-i*8192)+ median_val).astype('uint8')
                    noise_saturation[i * 8192:] = np.zeros(int(len(bad_ind[2]) - i * 8192))
            old_data[bad_ind[0], bad_ind[1], ipol, bad_ind[3], 0] = noise_saturation
    elif (args.clean):
        for ipol in range(npol):
            noise_saturation = np.zeros(len(bad_ind[2]))
            for i in range(int(np.ceil(len(bad_ind[2]) / 8192))):
                if ((i + 1) * 8192 < len(bad_ind[2])):
                    noise_saturation[i * 8192:(i + 1) * 8192] = gaussian_noise(shape=[int(8192)], std=std_val, median=median_val,
                                                                               noise_min=0, noise_max=255)  # np.round(std_val*np.random.randn(8192)+ median_val).astype('uint8')
                else:
                    # np.round(std_val*np.random.randn(len(bad_ind[2])-i*8192)+ median_val).astype('uint8')
                    noise_saturation[i * 8192:] = gaussian_noise(shape=[int(len(bad_ind[2]) - i * 8192)],
                                                                 std=std_val, median=median_val, noise_min=0, noise_max=255)
            old_data[bad_ind[0], bad_ind[1], ipol, bad_ind[3], 0] = noise_saturation

    old_data[old_data < 0] = 0
    old_data = np.round(old_data).astype('uint8')
    print(np.shape(old_data), np.shape(SCAL), np.shape(OFFSET))
    return old_data, SCAL, OFFSET


def gaussian_noise(shape, std=1, median=0, noise_min='None', noise_max='None'):
    data = std * np.random.randn(*shape) + median
    cont = 0
    while ((noise_min != 'None') or (noise_max != 'None')) or (cont < 1000):
        cont += 1
        if(noise_min != 'None'):
            min_ind = (data < noise_min)
            size = np.sum(min_ind)
            if (size > 0):
                noise = std * np.random.randn(*[size]) + median
                data[min_ind] = noise
            else:
                noise_min = 'None'
        if(noise_max != 'None'):
            max_ind = (data >= noise_max)
            size = np.sum(max_ind)
            if (size > 0):
                noise = std * np.random.randn(*[size]) + median
                data[max_ind] = noise
            else:
                noise_max = 'None'
    return data


#test = gaussian_noise(shape=[int(10000000), int(1)], std=42.5, median=85.0 - 1.28, noise_min=0, noise_max=255)
# print(np.median(test))
# print(np.std(test))
#print(mad(test, axis='all'))
# print(np.shape(test))
#import matplotlib
#import matplotlib.pyplot as plt
# plt.hist(test)
# plt.show()
# DATA EXTRACTION OF THE PREVIOUS FITS
headObs = fi.getheader(args.fileName, 0, do_not_scale_image_data=True, scale_back=True)        # Extraction of the observation header
head = fi.getheader(args.fileName, 1, do_not_scale_image_data=True, scale_back=True)        # Extraction of the data header
data = fi.getdata(args.fileName, do_not_scale_image_data=True, scale_back=True)            # Extraction of the data arrays
print(fi.info(args.fileName))


TSUBINT = data.field(0).astype('float32')
OFFS_SUB = data.field(1).astype('float32')
LST_SUB = data.field(2).astype('float32')
NROW = np.shape(data)[0]


# print(np.shape(TSUBINT))
# print(np.shape(OFFS_SUB))
# print(np.shape(LST_SUB))
#
# print(TSUBINT[1913:1920])
# print(OFFS_SUB[1913:1920])
# print(LST_SUB[1913:1920])
#print(data.field( 1 ) )
#print(len(data.field( 1 ) ))

# RESIZING ARRAYS

colList = []                # Field list for the new fits file


for i in range(14):
    oldArray = data.field(i)                   # Copy of the old amplitude data array
    oldCol = data.columns[i].copy()            # Copy of the old corresponding header
    print(i, oldCol.name, oldCol.format, oldCol.unit, oldCol.dim, np.shape(data.field(i)))
    #print(data.field( i )[0:9])
    if(oldCol.name == 'OFFS_SUB') and (int(args.subblock) > 1):
        old_OFFS_SUB = data.field(i)
        tsubint = old_OFFS_SUB[1] - old_OFFS_SUB[0]
        off = old_OFFS_SUB[0] - tsubint / 2.
        nint = np.size(old_OFFS_SUB)
        oldArray = np.arange(tsubint / float(args.subblock) / 2., nint * tsubint, tsubint / float(args.subblock)) - off
    elif(oldCol.name == 'OFFS_SUB') and (int(args.subblock) < 1):
        old_OFFS_SUB = data.field(i)
        tsubint = old_OFFS_SUB[1] - old_OFFS_SUB[0]
        off = old_OFFS_SUB[0] - tsubint / 2.
        nint = np.size(old_OFFS_SUB)
        oldArray = np.arange(tsubint / float(args.subblock) / 2., nint * tsubint, tsubint / float(args.subblock)) - off
    elif(int(args.subblock) > 1):
        oldArray = np.repeat(oldArray, int(args.subblock), axis=0)
    elif(int(args.subblock) < 1):
        shape = np.shape(oldArray)
        inv_subblock = int(1. / float(args.subblock))
        if(np.size(shape) == 1):
            shape_new = int(float(shape[0]) * float(args.subblock))
            oldArray = np.reshape(oldArray[:shape_new * inv_subblock], (shape_new, inv_subblock))
            oldArray = np.mean(oldArray, axis=1)
        if(np.size(shape) == 2):
            shape_new = int(float(shape[0]) * float(args.subblock))
            oldArray = np.reshape(oldArray[:shape_new * inv_subblock, :], (shape_new, int(1. / float(args.subblock)), shape[1]))
            oldArray = np.mean(oldArray, axis=1)
        #old_data = np.reshape(old_data,(int(nline*args.subblock), int(line_lenght/args.subblock), npol, nchan, 1))
        #oldArray = np.repeat(oldArray, int(args.subblock), axis=0)
    newCol = fi.Column(name=oldCol.name,         # Creation of the new field
                       format=oldCol.format,
                       unit=oldCol.unit,
                       dim=oldCol.dim,
                       array=oldArray)
    print(oldCol.name, np.shape(oldArray))
    # print(oldArray)
    colList.append(newCol)                     # Adding to the new field list

oldCol_offset = data.columns[14].copy()               # Copy of the old corresponding header
oldCol_scale = data.columns[15].copy()               # Copy of the old corresponding header
oldCol_data = data.columns[16].copy()               # Copy of the old corresponding header

print(head)
print(head['NBITS'])
head['NBITS'] = 8
npol = int(head['NPOL'])

if(args.pscrunch and npol > 1):
    if(args.intscales):
        head['TFORM15'] = str(int(float(head['TFORM15'][0:-1]) / npol)) + 'B'
        head['TFORM16'] = str(int(float(head['TFORM16'][0:-1]) / npol)) + 'B'
    else:
        head['TFORM15'] = str(int(float(head['TFORM15'][0:-1]) / npol)) + 'E'
        head['TFORM16'] = str(int(float(head['TFORM16'][0:-1]) / npol)) + 'E'
    head['TFORM17'] = str(int(float(head['TFORM17'][0:-1]) / float(npol) / float(ds) / float(args.subblock))) + 'B'

    head['NPOL'] = 1
    head['POL_TYPE'] = 'AA+BB'
else:
    if(args.intscales):
        head['TFORM15'] = str(int(float(head['TFORM15'][0:-1]))) + 'B'
        head['TFORM16'] = str(int(float(head['TFORM16'][0:-1]))) + 'B'
    else:
        head['TFORM15'] = str(int(float(head['TFORM15'][0:-1]))) + 'E'
        head['TFORM16'] = str(int(float(head['TFORM16'][0:-1]))) + 'E'
    head['TFORM17'] = str(int(float(head['TFORM17'][0:-1]) / float(ds) / float(args.subblock))) + 'B'

newFormat_offset = fi.column._ColumnFormat(head['TFORM15'])      # Definition of the new data array format
newFormat_scale = fi.column._ColumnFormat(head['TFORM16'])      # Definition of the new data array format
newFormat_data = fi.column._ColumnFormat(head['TFORM17'])      # Definition of the new data array format

print(head['TFORM15'])
print(head['TFORM16'])
print(head['TFORM17'])
nline = int(head['NAXIS2'])
line_lenght = int(head['NSBLK'])
npol = int(head['NPOL'])
nchan = int(head['NCHAN'])
global DATA


DATA = np.memmap(os.path.dirname(args.newFileName) + 'tmp_memmapped_' + os.path.basename(args.newFileName).split('.')[0], dtype=np.float32,
                 mode='w+', shape=(nline, line_lenght, npol, nchan, 1))
# DATA = np.memmap(os.path.dirname(args.newFileName)+'tmp_memmapped_'+os.path.basename(args.newFileName).split('.')[0], dtype='uint8',
#          mode='w+', shape=(nline, line_lenght, npol, nchan, 1))
print(np.shape(DATA))
DATA = data.field(16).astype('float32')
print(np.shape(DATA))
print(nline, line_lenght, npol, nchan, 1)
# exit(0)

if (ds > 1):
    head['NSBLK'] = int(head['NSBLK']) / ds
    head['TBIN'] = float(head['TBIN']) * ds
if (int(args.subblock) > 1):
    head['NSBLK'] = int(head['NSBLK']) / int(args.subblock)
    head['NAXIS2'] = head['NAXIS2'] * int(args.subblock)
if (args.subblock < 1):
    head['NSBLK'] = int(float(head['NSBLK']) / float(args.subblock))
    head['NAXIS2'] = int(float(head['NAXIS2']) * float(args.subblock))

ncore = float(args.ncore)
step = 16
step = int(ncore * np.floor(float(step) / float(ncore)))

if(float(args.subblock) < 1):
    nlost_block = nline % (1 / float(args.subblock))
    if(nlost_block > 0):
        nline = int(nline - nlost_block)

while (nline / ncore < step):
    step /= 2
row_vec = np.arange(0, nline, step)
row_vec = np.append(row_vec, nline)

pool = Pool(processes=int(ncore))

print(row_vec)
#flat_bunch_row(row_vec[0], row_vec[1])
# exit(0)
multiple_results = [pool.apply_async(flat_bunch_row,
                                     (row_vec[n], row_vec[n + 1]))
                    for n in range(len(row_vec) - 1)]


if (ds > 1):
    line_lenght = line_lenght / ds
if(args.pscrunch):
    npol = 1
if (float(args.subblock) < 1):
    line_lenght = int(float(line_lenght) / float(args.subblock))
    nline = int(float(nline) * float(args.subblock))
    head['TDIM17'] = '(1,' + str(int(nchan)) + ',' + str(int(npol)) + ',' + str(int(line_lenght)) + ')'
    row_vec = row_vec * float(args.subblock)
    row_vec = row_vec.astype('uint64')
if (int(args.subblock) > 1):
    line_lenght = line_lenght / int(args.subblock)
    nline = nline * int(args.subblock)
    head['TDIM17'] = '(1,' + str(int(nchan)) + ',' + str(int(npol)) + ',' + str(int(line_lenght)) + ')'
    row_vec = row_vec * int(args.subblock)


old_data = np.zeros([int(nline), int(line_lenght), int(npol), int(nchan), 1]).astype('uint8')
SCAL = np.zeros([int(nline), int(npol), int(nchan)])
OFFSET = np.ones([int(nline), int(npol), int(nchan)])


for n in range(int(len(row_vec) - 1)):
    #print('ROW_VEC: ',row_vec[int(n)],row_vec[int(n+1)])
    #A, B, C = multiple_results[int(n)].get()
    #print('SHAPE: ' , np.shape(A), n)
    old_data[int(row_vec[n]):int(row_vec[n + 1]), :, :, :, :], SCAL[int(row_vec[n]):int(row_vec[n + 1]), :,
                                                                    :], OFFSET[int(row_vec[n]):int(row_vec[n + 1]), :, :] = multiple_results[int(n)].get()  # get(timeout=1)


if (args.clean):
    mask_array = np.ones([int(nline), int(line_lenght), int(nchan)]).astype('bool')
    for iloop in range(4):
        old_data, mask_array = clean_data(old_data, mask_array, dm_zero=args.dm_zero)

    if (args.clean_with_zero):
        print('--------generating zeros and replace RFI-------')
        std_val = 255. / (float(args.threshold))
        median_val = 255 * (1. / 3)
        RFI_ratio = float(np.sum(~mask_array)) / float(np.size(mask_array))
        print("mask %.2f percent of the array" % (100. * RFI_ratio))
        bad_ind = np.where(mask_array == False)
        for ipol in range(npol):
            zeros_clean = np.zeros(len(bad_ind[2])).astype('uint8')
            for i in range(int(np.ceil(len(bad_ind[2]) / 8192))):
                if ((i + 1) * 8192 < len(bad_ind[2])):
                    # np.round(std_val*np.random.randn(8192)+ median_val).astype('uint8')
                    zeros_clean[i * 8192:(i + 1) * 8192] = np.round(np.zeros(int(8192))).astype('uint8')
                else:
                    # np.round(std_val*np.random.randn(len(bad_ind[2])-i*8192)+ median_val).astype('uint8')
                    zeros_clean[i * 8192:] = np.round(np.zeros(int(len(bad_ind[2]) - i * 8192))).astype('uint8')
            old_data[bad_ind[0], bad_ind[1], ipol, bad_ind[2], 0] = zeros_clean
        del zeros_clean
    else:
        print('--------generating normal white noise to replace RFI-------')
        std_val = 255. / (float(args.threshold))
        median_val = 255 * (1. / 3)
        RFI_ratio = float(np.sum(~mask_array)) / float(np.size(mask_array))
        print("mask %.2f percent of the array" % (100. * RFI_ratio))
        bad_ind = np.where(mask_array == False)
        for ipol in range(npol):
            noise_clean = np.zeros(len(bad_ind[2])).astype('uint8')
            for i in range(int(np.ceil(len(bad_ind[2]) / 8192))):
                if ((i + 1) * 8192 < len(bad_ind[2])):
                    noise_clean[i * 8192:(i + 1) * 8192] = np.round(gaussian_noise(shape=[int(8192)], std=std_val, median=median_val,
                                                                                   noise_min=0, noise_max=255)).astype('uint8')  # np.round(std_val*np.random.randn(8192)+ median_val).astype('uint8')
                else:
                    # np.round(std_val*np.random.randn(len(bad_ind[2])-i*8192)+ median_val).astype('uint8')
                    noise_clean[i * 8192:] = np.round(gaussian_noise(shape=[int(len(bad_ind[2]) - i * 8192)], std=std_val,
                                                                     median=median_val, noise_min=0, noise_max=255)).astype('uint8')
            old_data[bad_ind[0], bad_ind[1], ipol, bad_ind[2], 0] = noise_clean
        del noise_clean

std_val = 255. / (float(args.threshold))
median_val = 255. * (1. / 3.)

if (args.rmbase):
    for ipol in range(npol):
        for iline in range(nline):
            for ichan in range(nchan):
                old_data_TMP = old_data[iline, :, ipol, ichan, :].astype('float32')
                median = np.median(old_data_TMP)
                OFFSET[iline, ipol, ichan] = -median
    #OFFSET = -median_val*np.ones(np.shape(OFFSET))


os.remove(os.path.dirname(args.newFileName) + 'tmp_memmapped_' + os.path.basename(args.newFileName).split('.')[0])

# replace OFFSET and SCAL   '(1,'+str(nchan)+','+str(npol)+')'
newCol = fi.Column(name=oldCol_offset.name, format=newFormat_offset, unit=oldCol_offset.unit, dim='(1,' +
                   str(int(nchan)) + ',' + str(int(npol)) + ')', array=OFFSET)    # Creation of the new field
colList.append(newCol)
newCol = fi.Column(name=oldCol_scale.name, format=newFormat_scale, unit=oldCol_scale.unit, dim='(1,' +
                   str(int(nchan)) + ',' + str(int(npol)) + ')', array=SCAL)    # Creation of the new field
colList.append(newCol)

newCol = fi.Column(name=oldCol_data.name, format=newFormat_data, unit=oldCol_data.unit, dim='(1,' + str(int(nchan)) + ',' +
                   str(int(npol)) + ',' + str(int(line_lenght)) + ')', array=old_data.astype('uint8'))    # Creation of the new field
colList.append(newCol)                        # Adding to the new field list


# DEFINITION OF THE NEW FITS

print('---------save data to ' + args.newFileName + ' ---------')
colDefs = fi.ColDefs(colList)                    # Creation of the new fields object
tbhdu = fi.BinTableHDU.from_columns(colDefs, header=head)    # Creation of the new data table object

prihdu = fi.PrimaryHDU(header=headObs)            # Creation of the new observation header (exactly the same that the old fits file)
hdulist = fi.HDUList([prihdu, tbhdu])            # Creation of the new HDU object

hdulist.writeto(args.newFileName)  # output_verify='exception' )                # Writing the new HDU object on the new fits file
hdulist.close()

print('test')
