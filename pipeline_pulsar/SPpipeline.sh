# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                       SPpipeline

#       Single pulse pipeline for NenuFAR observations.
#               Version :       5f5
#               Date :          20/07/2020
#               Author :        Mark Brionne

#	       Comments :	Using dedispersion script with multiprocessing and scattering taking account.
#				Adding the minimal DM step, maximal number of DMs by CPU, and CPU number variables.
#				Adding the single pulses research using multiprocessing.
#				Removing the searching threshold variable (automatic threshold calculated directly inside the script).
#				Send prepsubband and single pulses research results in a log file.
#				Increasing PNG density to 300 dpi.
#				Removing flip in the PNG conversion of the single pulse research plot.
#				Add a minimal threshold for the single pulses plot.
#				Add an importing parameters function.
#				Add offset option for the jump zapping.
#				Add an option for the threshold for the single pulses research.


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INPUT PARAMETERS

# Computing parameters

nDiv=4								# Rescaling block time size factor
ds=1                                # time integration factor
nbPeriod=3                                                      # Number of period to use to the integration time of rfifind
rfiTimeSig=3.0							# Frequency threshold to reject blocks for the rfifind running
rfiFreqSig=20.0							# Frequency threshold to reject blocks for the rfifind running
rfiTimeFrac=0.3							# Fraction of bad intervals to reject a channel
rfiFreqFrac=0.5							# Fraction of bad channels to reject an interval
widthInt=1.5							# Width of the jumps to zap for the rfifind running
widthChan=0.5							# Width of the channels to zap for the rfifind running
offsetFits=120.0						# Time offset between the observation and the FITS for other KP than ES03
intervDM=2.0							# DM range to search the DM
dDMmin=0.001							# Minimum DM step tu use for the dedispersion
nbDMmax=2048							# Maximum number of DMs to try
nbCPUs=8							# Number of CPUs to use
spThres=7.5							# Minimal threshold to search the single pulses


# Importing non-default parameters
OPTS=$( getopt -o h,f:,d: -l div:,ds:,nper:,timesig:,freqsig:,timefrac:,freqfrac:,widthint:,widthchan:,interv:,ddm:,ndm:,ncpus:,thres: -- "$@" )

eval set -- "$OPTS"

usage ()
{
	printf "\nSPpipeline : script to search single pulses after a RFI research.\n"
	printf "\nMandatory options :\n\n"
	printf "\t-f : FITS file path to use ;\n"
	printf "\t-d : directory where works.\n"
	printf "\nOther options :\n\n"
	printf "\t--div : factor to divide the block size (default = 16) ;\n"
    printf "\t--ds : time integration factor (default = 1) ;\n"
	printf "\t--nper : number of periods to compute the statistics for the RFI research (default = 3) ;\n"
	printf "\t--timesig : threshold for the RFI research in median and standart deviation (default = 3.0 sigmas) ;\n"
	printf "\t--freqsig : threshold for the RFI research in maximum power (default = 20.0 sigmas) ;\n"
	printf "\t--timefrac : fraction of bad intervals to reject a channel (default = 0.3) ;\n"
	printf "\t--freqfrac : fraction of bad channels to reject an interval (default = 0.5) ;\n"
	printf "\t--widthint : time width of the jump to set weigths to 0 for the rfifind (default = 1.5 s) ;\n"
	printf "\t--widthchan : frequency width of the channel to set weigths to 0 for the rfifind (default = 0.5 MHz) ;\n"
	printf "\t--offsetfits : time offset between the observation and the FITS for other KP than ES03 (default = 120 s) ;\n"
	printf "\t--interv : DM interval around the used coherent dedispersion value (default = 3.0 pc.cm-3) ;\n"
	printf "\t--ddm : minimal DM step to use for the single pulse research (default = 0.001 pc.cm-3) ;\n"
	printf "\t--ndm : maximal numer of DMs to use at each step for the single pulse research (default = 500) ;\n"
	printf "\t--ncpus : number of CPUs to use for the single pulse research (default = 10) ;\n"
	printf "\t--thres : minimal threshold to search the single pulses (default = 7.5 sigmas);\n"
	printf "\t-h : display this help message.\n\n"
}

while true ; do

	case "$1" in

		-h) usage ; exit 0 ;;
		-f) fitsPath=$2 ; shift 2 ;;				# Path of the FITS file to work
		-d) dirPath=$2 ; shift 2 ;;				# Path of the working directory
		--div) nDiv=$2 ; shift 2 ;;
        --ds) ds=$2 ; shift 2 ;;
		--nper) nbPeriod=$2 ; shift 2 ;;
		--timesig) rfiTimeSig=$2 ; shift 2 ;;
		--freqsig) rfiFreqSig=$2 ; shift 2 ;;
		--timefrac) rfiTimeFrac=$2 ; shift 2 ;;
		--freqfrac) rfiFreqFrac=$2 ; shift 2 ;;
		--widthint) widthInt=$2 ; shift 2 ;;
		--widthchan) widthChan=$2 ; shift 2 ;;
		--offsetfits) offsetFits=$2 ; shift ;;
		--interv) intervDM=$2 ; shift 2 ;;
		--ddm) dDMmin=$2 ; shift 2 ;;
		--ndm) nbDMmax=$2 ; shift 2 ;;
		--ncpus) nbCPUs=$2 ; shift 2 ;;
		--thres) spThres=$2 ; shift 2 ;;
		--) shift ; break ;;

	esac

done


# Names and path parameters
scriptPath='/home/mbrionne/Scripts/SinglePulsePipeline'		# Path of the differents non-Presto scripts used
formatPath='/home/lbondonneau/scripts/pav/psrfits_search'	# Path of the changing format script


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# nMAIN

if [ -d $dirPath ] ;
	then
	cd $dirPath						# Positionning on the working directory.
else
	printf "\n\tNo working directory !\n"
	exit
fi
if [ -e $fitsPath ] ;
	then
	ln -s $fitsPath	.					# Creation of a symbolic link
else
	printf "\n\tNo file !\n"
	exit
fi

fitsName=$( ls *1.fits )					# Name of the FITS file in the working directory
outfileName=$( ls *1.fits | cut -d . -f 1 )			# Identifier of the observation
psr=$( echo $fitsName | cut -d _ -f 1 )				# PSR name used to named the different output files
				# PSR name used to named the different output files
type=$( echo $psr | cut -c1-3 )
if [ $type == 'FRB' ]
then
    nDiv=0.5 # increas block time size by factor 2 (to 20 sec)
    echo 'increas block time size by 1/'$nDiv
fi

#printf "\n----------------------------------------------------------------------------\n" | tee -a $errorLog
#printf "\n\tRescaling of the data block size.\n\n" | tee -a $errorLog
# Rescaling the FITS file to improve the RFI searching with shorter blocks.
#	The FITS file contains nDiv times more blocks where each block has a size nDiv shorter.
#ini_usedFitsName=$psr"_rescaled_32bits.fits"
errorLog=`echo "_error"$outfileName".log"`
#if [ ! -e *_rescaled_32bits.fits ]
#    then
#    python $scriptPath/RescaleTime.py -f $fitsName -o $ini_usedFitsName -n $nDiv 2>> $errorLog
#    #rm $fitsName
#fi


printf "\n----------------------------------------------------------------------------\n" | tee -a $errorLog
printf "\n\tChanging of the raw data format : 32 -> 8 bits.\n\n" | tee -a $errorLog
# Changing of the data format in the FITS file.
#	FITS file generated by LUPPI are data in 32 bits to have a better precision.
#	PRESTO needs to work on FITS file with data in 8 bits.
usedFitsName=$outfileName"_rescaled_8bits.fits"
min_freq='9999'
fits_info=`python $formatPath/fits_info.py $fitsName`
echo "$fits_info"
min_freq=`echo "$fits_info" | grep 'min frequency' | cut -d ':' -f2 | awk '{print$1}'`
DM=`echo "$fits_info" | grep 'coherent dedispersion' | cut -d ':' -f2 | awk '{print$1}'`
chan_bw=`echo "$fits_info" | grep 'chanel bandwidth' | cut -d ':' -f2 | awk '{print$1}'`
tsamp=`echo "$fits_info" | grep 'time sample' | cut -d ':' -f2 | awk '{print$1}'`
block_length=`echo "$fits_info" | grep 'block length' | cut -d ':' -f2 | awk '{print$1}'`
echo $min_freq
echo $chan_bw
echo $DM
echo $tsamp
echo $block_length
echo $intervDM
alow_dt=`python -c "print(4150.*((($min_freq)**-2) - (($min_freq+$chan_bw)**-2)))"`
DM_max=`python -c "print($block_length*$tsamp/$alow_dt)"`
echo $DM_max
echo $DM
echo $intervDM
max_nDiv=`python -c "max_nDiv=1; DM_max=$DM_max
while (DM_max > $DM):
    max_nDiv *=2
    DM_max /= 2
print(max_nDiv/2)"`
if [ $nDiv -gt $max_nDiv ]
then
    echo "Warning: the rescaling block time size factor implied a largeur dispersion in the last channel than the block size, the maximal value is "$max_nDiv
    nDiv=$max_nDiv
fi


if [ ! -e *_rescaled_8bits.fits ]
    then
    printf "python $formatPath/flat_bandpass_clean2.py -flat_per_block -dm_zero -clean -j $nbCPUs -ds $ds -subblock $nDiv -f $fitsName -o $usedFitsName -pscrunch\n" | tee -a $errorLog
    python $formatPath/flat_bandpass_clean2.py -flat_per_block -dm_zero -clean -j $nbCPUs -ds $ds -subblock $nDiv -f $fitsName -o $usedFitsName -pscrunch 2>> $errorLog
#rm $fitsName
fi



printf "\n----------------------------------------------------------------------------\n" | tee -a $errorLog
printf "\n\tRFI searching and mask creation.\n\n" | tee -a $errorLog
# Searching RFI by statistical methods on each block inside the FITS file.
#	Statistics used for the RFI searching are calculated on the nearest possible time of rfiTime.
#	The used time is the time size of a block multiply by a number of blocks.
period=`psrcat $psr -c P0 -o short -nohead -nonumber | cut -d ":" -f 1 | awk '{print $1}'`
if [ $period == 'Unknown' ]
then
    period='1.0'
fi

if [ ! -e *_rfifind.mask ]
    then
    printf "python $formatPath/CalcRfifindParam_light.py -p $period -n $nbPeriod -tsig $rfiTimeSig -fsig $rfiFreqSig -ifrac $rfiTimeFrac -cfrac $rfiFreqFrac -wi $widthInt -wc $widthChan -offs $offsetFits $usedFitsName\n" | tee -a $errorLog
    python $formatPath/CalcRfifindParam_light.py -p $period -n $nbPeriod -tsig $rfiTimeSig -fsig $rfiFreqSig -ifrac $rfiTimeFrac -cfrac $rfiFreqFrac -wi $widthInt -wc $widthChan -offs $offsetFits $usedFitsName 1> "_rfifind_"$outfileName".log" 2>> $errorLog
    good_intervals=`grep 'Number of  good  intervals' _rfifind_*.log | cut -d '(' -f2 | cut -d '%' -f1 | awk '{print$1}'`
    good_intervals=`python -c 'if('$good_intervals'< 50):
    print("True")
else:
    print("False")'`
    if [ $good_intervals == 'True' ]
    then
        rfiTimeFrac=0.99
        rfiFreqFrac=0.99
        printf "python $formatPath/CalcRfifindParam_light.py -p $period -n $nbPeriod -tsig $rfiTimeSig -fsig $rfiFreqSig -ifrac $rfiTimeFrac -cfrac $rfiFreqFrac -wi $widthInt -wc $widthChan -offs $offsetFits $usedFitsName\n" | tee -a $errorLog
        python $formatPath/CalcRfifindParam_light.py -p $period -n $nbPeriod -tsig $rfiTimeSig -fsig $rfiFreqSig -ifrac $rfiTimeFrac -cfrac $rfiFreqFrac -wi $widthInt -wc $widthChan -offs $offsetFits $usedFitsName 1> "_rfifind_"$outfileName".log" 2>> $errorLog
    fi
    pstoimg -density 300 -flip r270 $outfileName"_rfifind.ps"

fi

if [ ! -d "single_pulse" ]
then
    printf "\n----------------------------------------------------------------------------\n" | tee -a $errorLog
    printf "\n\tCreation of the dedispersion plan and corresponding dedispersed data.\n\n" | tee -a $errorLog
    #printf "python $formatPath/LFprepsubbandMP.py -i $intervDM -s $dDMmin -n $nbDMmax -c $nbCPUs -k $outfileName"_rfifind.mask" -o $outfileName $usedFitsName"
    
    printf "python $formatPath/LFprepsubbandMP.py -i $intervDM -s $dDMmin -n $nbDMmax -c $nbCPUs -t $spThres -k $outfileName"_rfifind.mask" -o $outfileName $usedFitsName\n" | tee -a $errorLog
    python $formatPath/LFprepsubbandMP.py -i $intervDM -s $dDMmin -n $nbDMmax -c $nbCPUs -t $spThres -k $outfileName"_rfifind.mask" -o $outfileName $usedFitsName 1> "_prepsubband_"$outfileName".log" 2>> $errorLog
    #exit
    
    
    printf "\n----------------------------------------------------------------------------\n" | tee -a $errorLog
    printf "\n\tSearching of the single pulses.\n\n" | tee -a $errorLog
    
    cd single_pulse
    printf "python $formatPath/single_pulse_search_mp.py -t $spThres *.singlepulse\n" | tee -a $errorLog
    python $formatPath/single_pulse_search_mp.py -t $spThres *.singlepulse 2>> ../$errorLog
    for f in $( ls -v *.singlepulse ) ;
    do
        wc -l $f >> $outfileName"_NbPulses.txt"
    done
    #rm *.singlepulse *.inf
    cd ../
    cd single_pulse_mark
    printf "python $formatPath/single_pulse_search/single_pulse_search_NenuFAR.py -t $spThres *.singlepulse\n" | tee -a $errorLog
    python $formatPath/single_pulse_search/single_pulse_search_NenuFAR.py -t $spThres *.singlepulse 2>> ../$errorLog
    for f in $( ls -v *.singlepulse ) ;
    do
    	wc -l $f >> $outfileName"_NbPulses.txt"
    done
    rm *.singlepulse *.inf
    cd ../
fi

printf "\n----------------------------------------------------------------------------\n" | tee -a $errorLog
printf "\n\tExtraction of the best single-pulse.\n\n" | tee -a $errorLog
printf "python $formatPath/plot_single_pulse.py -f $usedFitsName ./single_pulse/*.singlepulse -u ./ -nbest 5 -snr 4\n" | tee -a $errorLog
python $formatPath/plot_single_pulse.py -f $usedFitsName ./single_pulse/*.singlepulse -u ./ -nbest 5 -snr 4
printf "python $formatPath/plot_single_pulse.py -f $usedFitsName ./single_pulse/*.singlepulse -u ./single_pulse/ -nbest 10 -snr 4\n" | tee -a $errorLog
python $formatPath/plot_single_pulse.py -f $usedFitsName ./single_pulse/*.singlepulse -u ./single_pulse/ -nbest 10 -snr 4

printf "\n----------------------------------------------------------------------------\n" | tee -a $errorLog
printf "\nRemoving useless files.\n\n" | tee -a $errorLog
# Clearing the working directory to just keep files which will be in the .tar archive.
#	Removing time series (.dat) files, info (.inf) files.
#	Just keep the .singlepulse files, the 8 bits FITS file, and the different mask files.
#	Convert postscript files into PNG files.
#mv -v $outfileName"_NbPulses.txt" ../
#mv -v "_single_pulse_search_"$outfileName".log" ../
#rm $usedFitsName
mv -v single_pulse/search_ini_singlepulse.ps $outfileName"_ini_search.ps"
mv -v single_pulse_mark/search_singlepulse.ps $outfileName"_search.ps"
rm *.dat *.inf

if [ ! -e *_search.png ]
then
    pstoimg -density 300 $outfileName"_ini_search.ps"
    pstoimg -density 300 $outfileName"_search.ps"
fi

printf "\n----------------------------------------------------------------------------\n" | tee -a $errorLog


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
