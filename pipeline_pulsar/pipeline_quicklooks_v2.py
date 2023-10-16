import sys
#sys.path = ['/home/lbondonneau/lib/python/astropy/','/home/lbondonneau/scripts/Undysputed/PIPELINE/pipeline_quicklooks', '/usr/lib/python35.zip', '/usr/lib/python3.5', '/usr/lib/python3.5/plat-x86_64-linux-gnu', '/usr/lib/python3.5/lib-dynload', '/usr/local/lib/python3.5/dist-packages', '/usr/lib/python3/dist-packages']
sys.path.append('/home/lbondonneau/scripts/Undysputed/PIPELINE/pipeline_quicklooks/')

import os
import re
import glob
import socket
import argparse
from threading import Thread, RLock
import multiprocessing
import time
import subprocess
from subprocess import check_output
import numpy as np
from astropy.time import Time
from datetime import datetime
from datetime import timedelta
import MYDATABASE
import ATNF_16
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

"""
This code provide a quicklook 
pipeline v2.0.1
01/02/20: maintenant realise une recherche de single pulses
          produit un quicklook foldé pour les données singles pulses en plus du quichlook sur 5 min single pulses
          fonctionne maintenant uniquement sur les lanes de 37.5 MHz (la bande 75 MHz etait mal aligné)
09/02/20: recherche maintenan les fichiers à plus ou moins une seconde
          ajout d'une condition au cas ou le fichier 8bits n'est pas crée par SPpipeline.sh
30/03/20: used the new metadata_out of the quicklook pipeline
03/04/20: gestion de la memoire disponible avec les fonctions waiting_for_memory et used_memory
05/04/20: de-parallelize the single_pulse_thread for safety
07/08/20: join parset_file and the path in Warning mail
04/12/20: prepare_single_pulses_archive now pscrunch if fits file is > 100 GiB
04/12/20: do not bscrunch if nbin < 32
22/12/21: try/catch on valueError from the parset decoder
07/01/22: change mail for louis to nancay.fr, add MSP, RATT & RRAT, NRT 

"""


parser = argparse.ArgumentParser(description="This code will.")

parser.add_argument('-v', dest='verbose', action='store_true',
                    help="Verbose mode")
parser.add_argument('-debug', dest='debug', action='store_true',
                    help="Debug mode")

args = parser.parse_args()
global DEBUG
DEBUG = args.debug
PORT = [1242, 5586, 1242, 5586]
parset_user_dir = '/home/lbondonneau/OBS'
parset_user_dir_done = '/home/lbondonneau/parset'
parset_user_dir_crash = '/home/lbondonneau/parset_crash'
global tmp_dir
tmp_dir = '/home/lbondonneau/data/TMP/'


def TIME_TO_YYYYMMDD(TIME):
    Y = TIME.split('-')[0].strip()
    Y = Y[2] + Y[3]
    M = TIME.split('-')[1].strip()
    D = TIME.split('-')[2].split('T')[0].strip()
    H = TIME.split('T')[1].split(':')[0].strip()
    Mi = TIME.split(':')[1].strip()
    #TIME_TO_YYYYMMDDS = TIME.split(':')[2].split('Z')[0].strip()
    string = '20%s%s%sT%s%s' % (Y, M, D, H, Mi)
    return string


def TIME_TO_YYYY_MM_DD(TIME):
    Y = TIME.split('-')[0].strip()
    Y = Y[2] + Y[3]
    M = TIME.split('-')[1].strip()
    D = TIME.split('-')[2].split('T')[0].strip()
    H = TIME.split('T')[1].split(':')[0].strip()
    Mi = TIME.split(':')[1].strip()
    S = TIME.split(':')[2].split('Z')[0].strip()
    string = '20%s-%s-%sT%s:%s' % (Y, M, D, H, Mi)
    return string


def TIME_TO_MJDS(TIME):
    Y = TIME.split('-')[0].strip()
    M = TIME.split('-')[1].strip()
    D = TIME.split('-')[2].split('T')[0].strip()
    H = TIME.split('T')[1].split(':')[0].strip()
    Mi = TIME.split(':')[1].strip()
    S = TIME.split(':')[2].split('Z')[0].strip()
    SMJHD = int(H) * 3600 + int(Mi) * 60 + int(S)
    SMJHD = SMJHD % 86400
    return SMJHD


def fichiers_search(files):
    for ifile in range(len(files)):
        file_date = datetime.strptime(os.path.basename(files[ifile]).split('_')[1].split('*')[0], "D%Y%m%dT%H%M")
        for difmin in range(-2, 3):
            file_date_diff = file_date + timedelta(minutes=int(difmin))
            file_diff = os.path.dirname(files[ifile]) + '/' + os.path.basename(files[ifile]).split('_')[0] + \
                file_date_diff.strftime("_D%Y%m%dT%H%M*") + 'BEAM' + files[ifile].split('BEAM')[1]
            if not (glob.glob(file_diff) == []):
                files[ifile] = file_diff
    return files


def fichiers_exist(files):
    stdout = True
    for file in files:
        try:
            file_date = datetime.strptime(os.path.basename(file).split('_')[1].split('*')[0], "D%Y%m%dT%H%M")
        except ValueError:
            file_date = datetime.strptime(os.path.basename(file).split('_')[2].split('*')[0], "D%Y%m%dT%H%M")
        stdout_perfile = False
        for difmin in range(-2, 3):
            file_date_diff = file_date + timedelta(minutes=int(difmin))
            file_diff = os.path.dirname(file) + '/' + os.path.basename(file).split('_')[0] + \
                file_date_diff.strftime("_D%Y%m%dT%H%M*") + 'BEAM' + os.path.basename(file).split('BEAM')[1]
            if not (glob.glob(file_diff) == []):
                stdout_perfile = True
        if not (stdout_perfile):
            stdout = False
            break
    return stdout


def who_exist(files):
    file_exist = []
    for file in files:
        if not (glob.glob(file) == []):
            file_exist.append(True)
        else:
            file_exist.append(False)
    return file_exist


def decode_parset_user(pathto):
    print(pathto)
    Parsetfile = open(pathto, "r")
    nlane = 0
    AlltargetList = []
    AllstartTime = []
    AllparametersList = []
    AllmodeList = []
    AllRAdList = []
    AllDECdList = []
    AlllowchanList = []
    AllhighchanList = []
    for line in Parsetfile:
        if not re.search("AnaBeam", line):
            # print(line)
            if re.search("Observation.title", line):
                TITLE = line.split('=')[1].strip().strip('"')
            if re.search("Observation.stopTime", line):
                stopTime = line.split('=')[1].strip()
                #stopTime2 = TIME_TO_YYYYMMDD(stopTime)
                #stopTime = TIME_TO_YMDHMS(stopTime)
            # if re.search("Observation.startTime", line):
            #    STARTmjds = TIME_TO_YYYYMMDD(line.split('=')[1].strip())
            #    STARTmjds2 = TIME_TO_YYYY_MM_DD(line.split('=')[1].strip())
            if re.search("Observation.nrBeams", line):
                nBEAM = int(line.split('=')[1].strip())
            if re.search("Output.hd_lane", line):
                nlane += 1
    # print(stopTime)
    # print(nBEAM)
    # print(nlane)
    laneperbeam = np.zeros(nBEAM)
    beam_lane = np.zeros([nBEAM, nlane])
    lanenumber = np.nan * np.zeros([nBEAM, nlane])
    beamnumber = np.nan * np.zeros([nBEAM, nlane])
    nlane = 0

    for BEAM in range(nBEAM):
        AlltargetList = AlltargetList + ['NONE']
        AllparametersList = AllparametersList + ['NONE']
        AllstartTime = AllstartTime + ['NONE']
        AllRAdList = AllRAdList + ['NONE']
        AllDECdList = AllDECdList + ['NONE']
        AlllowchanList = AlllowchanList + ['NONE']
        AllhighchanList = AllhighchanList + ['NONE']
        AllmodeList = AllmodeList + ['NONE']
        Parsetfile = open(pathto, "r")
        for line in Parsetfile:
            if (re.search('Beam\[' + str(BEAM) + '\]', line)) and not (re.search("AnaBeam", line)):
                if re.search("startTime=", line):
                    AllstartTime[BEAM] = [line.split('=')[1].strip()]
                # if re.search("title=", line):
                #   MODE=line.split('"')[1].split('_')[1].rstrip()
                if (re.search("target=", line)):
                    AlltargetList[BEAM] = [line.split('"')[1].split('_')[0].strip()]
                if (re.search("toDo=TBD", line)):
                    AllmodeList[BEAM] = ['TBD']
                if (re.search("parameters=", line)):
                    if not (line[len(line.strip()) - 1] == '='):  # if parameters is not empti
                        # print(line.strip('"'))
                        AllmodeList[BEAM] = [line.split('=')[1].split(':')[0].strip('"').strip()]
                        param_tmp = line.strip('"').split(':')
                        if (np.size(param_tmp) > 1):
                            AllparametersList[BEAM] = [line.strip('"').split(':')[1].strip().strip('"').lower()]
                        else:
                            AllparametersList[BEAM] = ['']
                        if (re.search("--SRC=", line)):
                            split_param = line.split(':')[1].split(' ')
                            for param in split_param:
                                if re.search("--SRC=", param):
                                    AlltargetList[BEAM] = [param.strip().split('=')[1].strip('"')]
                    else:
                        AllmodeList[BEAM] = ['TF']
                if (re.search("angle1=", line)):
                    AllRAdList[BEAM] = [line.split('=')[1].strip()]
                if (re.search("angle2=", line)):
                    AllDECdList[BEAM] = [line.split('=')[1].strip()]
                if (re.search("lane.=", line)):
                    nlane = int(line.split('=')[0].split('.lane')[1])
                    beam_lane[int(BEAM), int(nlane)] = 1
                    lanenumber[int(BEAM), int(nlane)] = nlane
                    beamnumber[int(BEAM), int(nlane)] = BEAM
                    laneperbeam[BEAM] += 1
                    AlllowchanList[BEAM] = [line.split('=')[1].split('[')[1].split('.')[0].strip()]
                if (re.search("lane.=", line)):
                    AllhighchanList[BEAM] = [line.split('=')[1].split('.')[-1].split(']')[0].strip()]
    MODE = []
    PARAM = []
    SRC = []
    RAd = []
    DECd = []
    file = []
    minchan = []
    maxchan = []
    first_tf = 0
    for lane in range(nlane + 1):
        BEAM = np.argmax(beam_lane[:, lane])
        #-------------------------------TF---MODE------------------------------------------
        if (AllmodeList[BEAM][0] == 'TF'):
            if (first_tf == 1):
                first_tf = 0
                print(os.path.basename(pathto) + '  ' + datetime.utcnow().isoformat() + ' :' +
                      '-----------------------TF----MODE--BEAM-' + str(BEAM) + '--LANE--' + str(lane) + '----------')
                file.append('')
            else:
                file.append('')
        #-------------------------------SINGLE---MODE----------------------------------------
        elif (AllmodeList[BEAM][0] == 'SINGLE'):
            print(os.path.basename(pathto) + '  ' + datetime.utcnow().isoformat() + ' :' +
                  '--------------SINGLE--PULSE--MODE--BEAM-' + str(BEAM) + '-----LANE--' + str(lane) + '-------')
            file.append('/databf/nenufar-pulsar/DATA/'
                        + AlltargetList[BEAM][0]
                        + '/SEARCH/'
                        + AlltargetList[BEAM][0]
                        + '_D' + TIME_TO_YYYYMMDD(AllstartTime[BEAM][0])
                        + '*BEAM' + str(BEAM) + '_*.fits')
        #-------------------------------WAVEFORM---MODE----------------------------------------
        elif (AllmodeList[BEAM][0] == 'WAVE'):
            print(os.path.basename(pathto) + '  ' + datetime.utcnow().isoformat() + ' :' +
                  '--------------WAVEFORM---MODE---BEAM-' + str(BEAM) + '-----LANE--' + str(lane) + '-------')
            file.append('/databf/nenufar-pulsar/DATA/'
                        + AlltargetList[BEAM][0]
                        + '/RAW/'
                        + AlltargetList[BEAM][0]
                        + '_D' + TIME_TO_YYYYMMDD(AllstartTime[BEAM][0])
                        + '*BEAM' + str(BEAM) + '.*.raw')
        #-------------------------------WAVEFORM--OLAF--MODE----------------------------------------
        elif (AllmodeList[BEAM][0] == 'WAVEOLAF'):
            print(os.path.basename(pathto) + '  ' + datetime.utcnow().isoformat() + ' :' +
                  '--------------WAVEFORM--OLAF--MODE---BEAM-' + str(BEAM) + '-----LANE--' + str(lane) + '-------')
            if (lane < 2):
                machine = 'undysputedbk1'
            else:
                machine = 'undysputedbk2'

            file.append('/databf/nenufar-pulsar/TMP/'
                        + AlltargetList[BEAM][0]
                        + '_' + str(PORT[BEAM]) + '.' + machine + '.'
                        + TIME_TO_YYYY_MM_DD(AllstartTime[BEAM][0]) + '*.zst')
        #-------------------------------FOLD---MODE----------------------------------------
        elif (AllmodeList[BEAM][0] == 'FOLD'):
            print(os.path.basename(pathto) + '  ' + datetime.utcnow().isoformat() + ' :' +
                  '-----------------------FOLD--MODE--BEAM-' + str(BEAM) + '-------------------')
            file.append('/databf/nenufar-pulsar/DATA/'
                        + AlltargetList[BEAM][0]
                        + '/PSR/'
                        + AlltargetList[BEAM][0]
                        + '_D' + TIME_TO_YYYYMMDD(AllstartTime[BEAM][0])
                        + '*BEAM' + str(BEAM) + '.fits')
        elif (AllmodeList[BEAM][0] == 'TBD'):
            print(os.path.basename(pathto) + '  ' + datetime.utcnow().isoformat() + ' :' +
                  '-----------------------TBD--MODE--BEAM-' + str(BEAM) + '-------------------')
            file.append('')
        else:
            print(os.path.basename(pathto) + '  ' + datetime.utcnow().isoformat() + ' :' +
                  '-----------------------FAKE--MODE--BEAM-' + str(BEAM) + '-------------------')
            file.append('')
        MODE.append(AllmodeList[BEAM][0])
        PARAM.append(AllparametersList[BEAM][0])
        SRC.append(AlltargetList[BEAM][0])
        RAd.append(AllRAdList[BEAM][0])
        DECd.append(AllDECdList[BEAM][0])
        minchan.append(AlllowchanList[BEAM][0])
        maxchan.append(AllhighchanList[BEAM][0])
    for index in range(len(MODE)):
        for indey in range(index + 1, len(MODE)):
            if (index >= len(MODE)) or (indey >= len(MODE)):
                continue
            if ((int(maxchan[index]) + 1 == int(minchan[indey])) or (int(maxchan[indey]) + 1 == int(minchan[index]))) and (MODE[index] == MODE[indey]) and (SRC[index] == SRC[indey]) and (RAd[index] == RAd[indey]):
                if (MODE[index] == 'FOLD') or (MODE[index] == 'SINGLE'):
                    file[index] = file[index] + ' ' + file[indey]
                    file.pop(indey)
                    minchan.pop(indey)
                    maxchan.pop(indey)
                    MODE.pop(indey)
                    SRC.pop(indey)
                    RAd.pop(indey)
                    DECd.pop(indey)
                    PARAM.pop(indey)
    for index in range(len(MODE)-1, -1, -1):
        if (MODE[index] == 'TBD'):
            file.pop(index)
            minchan.pop(index)
            maxchan.pop(index)
            MODE.pop(index)
            SRC.pop(index)
            RAd.pop(index)
            DECd.pop(index)
            PARAM.pop(index)

    MAIL = title_to_mail(TITLE)
    return MAIL, TITLE, MODE, PARAM, SRC, file, stopTime


def title_to_mail(title):
    print(datetime.utcnow().isoformat() + ' :' + title)
    KP_list = ['census',
               'census_highdm',
               'census_lowdec',
               'ratt',
               'rrat',
               'lotas',
               'monitoring',
               'vlbi',
               'solar_wind',
               'eclipsing_bin',
               'drifting_subpulses',
               'giant_pulses',
               'single_pulses',
               'globularclusters',
               'polarisation_studies',
               'blind_survey',
               'simult',
               'msp',
               'nrt',
               'test',
               'cal',
               'student',
               'nenufar-survey']

    Mail_liste = [['louis.bondonneau@obs-nancay.fr'],  # census
                  ['louis.bondonneau@obs-nancay.fr', 'yerin.serge@gmail.com'],  # census_highdm
                  ['louis.bondonneau@obs-nancay.fr'],  # census_lowdec
                  ['louis.bondonneau@obs-nancay.fr', 'mckennd2@tcd.ie', 'ihor.kravtsov@obspm.fr'],  # ratt
                  ['louis.bondonneau@obs-nancay.fr', 'mckennd2@tcd.ie', 'ihor.kravtsov@obspm.fr'],  # rrat
                  ['louis.bondonneau@obs-nancay.fr'],  # census_lotas
                  ['louis.bondonneau@obs-nancay.fr', 'mark.brionne@cnrs-orleans.fr'],  # monitoring
                  ['louis.bondonneau@obs-nancay.fr', 'wucknitz@mpifr-bonn.mpg.de'],  # vlbi
                  ['louis.bondonneau@obs-nancay.fr', '1984cat.ti@gmail.com', 'golam.shaifullah@gmail.com'],  # solar_wind
                  ['louis.bondonneau@obs-nancay.fr', 'ramain@mpifr-bonn.mpg.de', 'andrea.possenti@inaf.it'],  # eclipsing_bin
                  ['louis.bondonneau@obs-nancay.fr', 'vlad.kondratiev@gmail.com', 'hanna.bilous@gmail.com'],  # drifting_subpulses
                  ['louis.bondonneau@obs-nancay.fr', 'jamesmckee23@tgmail.com'],  # giant_pulses
                  ['louis.bondonneau@obs-nancay.fr', 'vlad.kondratiev@gmail.com', 'hanna.bilous@gmail.com', 'mark.brionne@cnrs-orleans.fr'],  # single_pulses
                  ['louis.bondonneau@obs-nancay.fr', 'andrea.possenti@inaf.it'],  # globularclusters
                  ['louis.bondonneau@obs-nancay.fr', 'jean-mathias.griessmeier@cnrs-orleans.fr'],  # polarisation_studies , 'anoutsos@mpifr-bonn.mpg.de'
                  ['louis.bondonneau@obs-nancay.fr', 'mark.brionne@cnrs-orleans.fr'],  # blind_survey
                  ['louis.bondonneau@obs-nancay.fr', '1984cat.ti@gmail.com'],  # simult with NRT
                  ['louis.bondonneau@obs-nancay.fr', '1984cat.ti@gmail.com'],  # msp
                  ['louis.bondonneau@obs-nancay.fr', '1984cat.ti@gmail.com'],  # nrt
                  ['louis.bondonneau@obs-nancay.fr', 'mark.brionne@cnrs-orleans.fr', 'jean-mathias.griessmeier@cnrs-orleans.fr'],  # test
                  ['louis.bondonneau@obs-nancay.fr', 'cedric.viou@obs-nancay.fr'],  # cal
                  ['louis.bondonneau@obs-nancay.fr', 'jean-mathias.griessmeier@cnrs-orleans.fr'],  # student
                  ['louis.bondonneau@obs-nancay.fr', 'mark.brionne@cnrs-orleans.fr', 'fabian.jankowski@cnrs-orleans.fr']]  # nenufar-survey
    mail = []
    Findtitle = False
    for KP_id in range(len(KP_list)):
        # if(title == KP_list[KP_id]):
        if(KP_list[KP_id] in title):
            Findtitle = True
            for new_mail in Mail_liste[KP_id]:
                mail.append(new_mail)
    if not Findtitle:
        mail = ['louis.bondonneau@obs-nancay.fr']
    if(DEBUG):
        mail = ['louis.bondonneau@obs-nancay.fr']
    print(mail)
    return mail


def waiting_for_file(file, stop_time, mail, parset_path='/home/lbondonneau/OBS/PARSET.parset'):
    stop_mjd = Time(stopTime, format='isot', scale='utc').mjd  # FIXIT stopTime != stop_time
    curent_mjd = Time(datetime.utcnow(), scale='utc').mjd
    STEPTIME = 30
    WAITTIME = 60 * 60
    if (DEBUG):
        WAITTIME = 30
    #print('stop MJD = '+str(stop_mjd), 'current MJD = '+str(curent_mjd))
    if (curent_mjd < stop_mjd):
        print(datetime.utcnow().isoformat() + ' :' + 'waiting for ' + str((stop_mjd - curent_mjd) * 86400) + ' sec')
        time.sleep((stop_mjd - curent_mjd) * 86400)
    WAIT = True
    file_liste = file.split(' ')
    for f in file_liste:
        print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + 'search for file %s' % (f))
    cont = 0
    first = True
    while((cont < WAITTIME) and WAIT):  # waiting during 1h with a test per 30 sec
        cont += STEPTIME
        if fichiers_exist(file_liste):
            file_liste = fichiers_search(file_liste)
            WAIT = False
            for f in file_liste:
                print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + 'file %s found' % (f))
                if (first):
                    real_file = glob.glob(f)[0]
                    first = False
                else:
                    real_file = real_file + ' ' + glob.glob(f)[0]
            return real_file
        elif (cont >= WAITTIME):
            WAIT = False
            sendMail(MAIL, ('ERROR observation ' + parset_path), 'At least one file is not found 1 h after the end of the observation \n' +
                     ' '.join(file_liste) + '\nparset: ' + parset_path, glob.glob(parset_path))
            print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + 'WARNING: File not found after 1h ', file_liste)

            file_liste = np.asarray(file_liste)
            ind_exist = np.asarray(who_exist(file_liste))
            existing_files = file_liste[ind_exist]
            # print(len(existing_files))
            if (len(existing_files) >= 1):
                for f in existing_files:
                    print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' : file %s found' % (f))
                    if (first):
                        real_file = glob.glob(f)[0]
                        first = False
                    else:
                        real_file = real_file + ' ' + glob.glob(f)[0]
            else:
                print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + 'ERROR: no file found exit')
                exit(0)  # STOP
            return real_file
        else:
            time.sleep(STEPTIME)


def mkdir(dirname):
    """ """
    commande = 'mkdir ' + dirname
    try:
        output = check_output(commande, shell=True)
    except:
        return


def rmdir(dirname):
    """ """
    commande = 'rm -r ' + dirname
    try:
        output = check_output(commande, shell=True)
    except:
        return


def rmfile(filename):
    """ """
    commande = 'rm ' + filename
    try:
        output = check_output(commande, shell=True)
    except:
        return


def mvfile(filename, dirname):
    """ """
    commande = 'mv ' + filename + ' ' + dirname
    try:
        output = check_output(commande, shell=True)
    except:
        return


def targz(directory, extension=''):
    """ """
    pwd = os.getcwd()
    if(directory[-1] == '/'):
        directory = directory[0:-1]
    path = './'

    if(os.path.dirname(directory) != ''):
        path = os.path.dirname(directory) + '/'
        os.chdir(os.path.dirname(directory))
        directory = os.path.basename(directory)

    commande = "tar zcvf " + directory + extension + ".tar " + directory
    print(datetime.utcnow().isoformat() + ' :' + commande)
    try:
        output = check_output(commande, shell=True)
    except:
        os.chdir(pwd)
        return ''
    os.chdir(pwd)
    return path + directory + extension + ".tar"


def used_memory():
    """ """
    commande = 'top -n 1 | head -5 | tail -2 | cut -d \':\' -f2 | awk \'{total1+=$6}{total2+=$2}END{print 100*total1/total2}\''  # the result of top -n 1 have to be in GiB
    try:
        output = check_output(commande, shell=True)
    except:
        return 0
    return output


def psredit_dstime(bitsfile):
    """ """
    commande = 'psredit -c sub:tsamp ' + bitsfile + ' | cut -d \'=\' -f2'  # the result of top -n 1 have to be in GiB
    print(commande)
    try:
        output = check_output(commande, shell=True)
        print(output)
    except:
        return 0
    return float(output) / 5.12e-6


def psredit_nbin(bitsfile):
    """ """
    commande = 'psredit -c nbin ' + bitsfile + ' | cut -d \'=\' -f2'  # the result of top -n 1 have to be in GiB
    print(commande)
    try:
        output = check_output(commande, shell=True)
    except:
        return 0
    return float(output)


def psredit_length(bitsfile):
    """ """
    commande = 'psredit -c sub:tsamp ' + bitsfile + ' | cut -d \'=\' -f2'  # the result of top -n 1 have to be in GiB
    try:
        tsamp = check_output(commande, shell=True)
    except:
        tsamp = 0.001
    commande = 'psredit -c sub:nsblk ' + bitsfile + ' | cut -d \'=\' -f2'  # the result of top -n 1 have to be in GiB
    try:
        nsblk = check_output(commande, shell=True)
    except:
        nsblk = 16384
    commande = 'psredit -c sub:nrows ' + bitsfile + ' | cut -d \'=\' -f2'  # the result of top -n 1 have to be in GiB
    try:
        nrows = check_output(commande, shell=True)
    except:
        nrows = 16384
    return float(tsamp) * float(nsblk) * float(nrows)


def waiting_for_memory(used_limit=10):
    percent_of_memory = float(used_memory())
    while(percent_of_memory > used_limit):
        time.sleep(30)
        percent_of_memory = float(used_memory())


def build_single_pulses_archive(bitsfile, tmp_dir_new=tmp_dir):
    """ """
    psr_name = bitsfile.split('_')[0]
    commande = 'psredit -c freq ' + tmp_dir_new + bitsfile + ' | cut -d = -f2'
    print(datetime.utcnow().isoformat() + ' :' + commande)
    file_name = bitsfile.split('.')[0]
    output = check_output(commande, shell=True)
    freq = float(output)
    P0 = ATNF_16.search(psr_name, 'P0')[1]
    if (P0 == []):
        P0 = [1.0]
    P0 = float(P0[0])
    length = psredit_length(tmp_dir_new + bitsfile)
    if (600.0 / P0 > 2000.0):
        STR = ' -L 0.3 '
    else:
        STR = ' -s '
    if (float(freq) < 50):
        if (600 < length):
            STR = STR + ' -T 600'
    else:
        if (300 < length):
            STR = STR + ' -T 300'
    commande = 'dspsr -scloffs -N ' + psr_name + ' -K -A ' + STR + ' -E /databf/nenufar-pulsar/ES03/ephem/' + \
        psr_name + '.par  -O ' + tmp_dir_new + file_name + '_singlepulses ' + tmp_dir_new + bitsfile
    print(commande)
    try:
        subprocess.check_output(commande, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
    return tmp_dir_new + file_name + '_singlepulses.ar'


def prepare_single_pulses_archive(fits, bitsfile, tmp_dir_new=tmp_dir, dstime=1):
    file_size = float(os.path.getsize(fits)) / 1024 / 1024 / 1024  # GiB
    if (dstime > 1):
        dstime = ' -ds ' + str(dstime)
        file_size = file_size / dstime
    else:
        dstime = ''
    if (file_size > 100):
        print('WARNING: File size is largeur than 100 GiB in consequence the polarisation will be scrunch')
        dstime = ' -pscrunch ' + dstime

    commande = 'python2.7 /home/lbondonneau/scripts/pav/psrfits_search/ChangeFormat_rescale.py ' + str(dstime) + ' -f ' + fits + ' -o ' + tmp_dir_new + bitsfile
    output = check_output(commande, shell=True)


def build_folded_archive(bitsfile, tmp_dir_new=tmp_dir):
    """ """
    psr_name = bitsfile.split('_')[0]
    commande = 'psredit -c freq ' + tmp_dir_new + bitsfile + ' | cut -d = -f2'
    print(datetime.utcnow().isoformat() + ' :' + commande)
    file_name = bitsfile.split('.')[0]

    output = check_output(commande, shell=True)
    freq = float(output)
    P0 = ATNF_16.search(psr_name, 'P0')[1]
    if (P0 == []):
        P0 = MYDATABASE.search(psr_name, 'P0')[1]
    if (P0 == []):
        P0 = [1.0]
    P0 = float(P0[0])
    integration = 10.73741824
    if (integration < P0):
        STR = ' -s '
    else:
        STR = ' -L ' + str(integration) + ' '

    commande = 'dspsr -scloffs -N ' + psr_name + ' -K -A ' + STR + ' -E /databf/nenufar-pulsar/ES03/ephem/' + \
        psr_name + '.par  -O ' + tmp_dir_new + file_name + '_folded ' + tmp_dir_new + bitsfile
    try:
        subprocess.check_output(commande, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
    return tmp_dir_new + file_name + '_folded.ar'


def run_command(cmd, log_file):
    # Ouvrir le fichier de log en mode 'append'.
    with open(log_file, 'a') as f:
        try:
            # Lancer la commande.
            result = subprocess.run(cmd, stdout=f, stderr=f, text=True)
            # text=True fait en sorte que stdout et stderr soient retournés en tant que chaînes de caractères.
            # Si vous utilisez Python <3.7, utilisez `universal_newlines=True` à la place de `text=True`.

        except Exception as e:
            # En cas d'exception (par exemple, la commande n'existe pas), écrire l'erreur dans le fichier de log.
            print(f"Error executing command {cmd}: {str(e)}\n")
            f.write(f"Error executing command {cmd}: {str(e)}\n")

        else:
            # Si vous voulez vérifier le code de sortie de la commande.
            if result.returncode != 0:
                print(f"Command {cmd} exited with code {result.returncode}\n")
                f.write(f"Command {cmd} exited with code {result.returncode}\n")


def folding_thread(src, file, mail, title, stopTime, parset_path='/home/lbondonneau/OBS/PARSET.parset'):
    """Code à exécuter pendant l'exécution du thread folding_thread."""
    print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + 'FOLD')
    file = waiting_for_file(file, stopTime, mail, parset_path)
    commande = ['python', '/cep/lofar/pulsar/NenuPlot_v2.py', '-fit_DM', '-defaraday', '-uploadpdf', '-initmetadata', '-nopdf', '-png',
                '-b', '4', '-t', '2', '-mailtitle', title, '-mail', '[' + mail[0]]
    commande4 = ['python', '/cep/lofar/pulsar/NenuPlot_DIR/NenuPlot_v4/NenuPlot.py',
                 '-u', '/home/lbondonneau/data/TMP/', '-fit_DM', '-fit_RM', '-defaraday', '-noPDF_out', '-noPNG_out',
                 '-b', '4', '-t', '2',
                 '-mailtitle', title, '-mail',
                 '[louis.bondonneau@obs-nancay.fr] ']
    for i in range(1, len(mail)):
        commande[-1] = commande[-1] + ',' + mail[i]
    commande[-1] = commande[-1] + ']'
    for singlefile in file.split(" "):
        commande.append(singlefile)
        commande4.append(singlefile)
    print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + ' '.join(commande))
    print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + ' '.join(commande4))
    if(DEBUG):
        exit(0)
    waiting_for_memory(used_limit=20)
    run_command(commande, "/home/lbondonneau/data/TMP/NenuPlot_v2.log")
    run_command(commande4, "/home/lbondonneau/data/TMP/NenuPlot_v4.log")


def single_pulse_thread(src, file, mail, title, stopTime, parset_path='/home/lbondonneau/OBS/PARSET.parset'):
    """Code à exécuter pendant l'exécution du thread single_pulse_thread."""
    print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + '  :' + 'SINGLE')
    file = waiting_for_file(file, stopTime, mail, parset_path)
    exit()


    out_file = ''
    for fits in file.split(' '):
        basename = os.path.basename(fits)
        psr_name = basename.split('_')[0]
        tmp_dir_new = tmp_dir + basename.split('.')[0] + '/'
        # if(DEBUG):
        #    print(os.path.basename(parset_path)+'  '+datetime.utcnow().isoformat()+' : mkdir '+tmp_dir_new)
        #    continue
        mkdir(tmp_dir_new)

        if (psredit_length(fits) < 4 * 3600.):
            # bitsfile = basename.split('.')[0]+'_8bits.fits'
            bitsfile = basename.split('.')[0] + '_rescaled_8bits.fits'

            # if( psredit_dstime(fits) < 32 ):
            #    if not fichiers_exist([tmp_dir_new+bitsfile]):
            #        waiting_for_memory(used_limit=10)
            #        prepare_single_pulses_archive(fits, bitsfile, tmp_dir_new=tmp_dir_new, dstime=2)
            # else:
            # commande = 'bash /home/mbrionne/Scripts/SinglePulsePipeline/SPpipeline.sh --thres 7.5 --freqsig 3 --timesig 3 --div 256 -f '+fits+' -d '+tmp_dir_new
            commande = 'bash /home/lbondonneau/scripts/pav/psrfits_search/SPpipeline.sh --thres 5.5 --freqsig 3 --timesig 3 --div 1 --ncpus 8 --timefrac 0.2 --freqfrac 0.4 -f ' + fits + ' -d ' + tmp_dir_new

            # commande = 'bash /home/lbondonneau/scripts/Undysputed/PIPELINE/pipeline_quicklooks/single_pulse_test/SPpipeline.sh '+fits+'  '+tmp_dir_new
            print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + commande)

            # waiting_for_memory(used_limit=10)
            # if not (DEBUG):
            #     output = check_output(commande, shell=True)

            if not fichiers_exist([tmp_dir_new + bitsfile]):
                waiting_for_memory(used_limit=10)
                prepare_single_pulses_archive(fits, bitsfile, tmp_dir_new=tmp_dir_new)

            waiting_for_memory(used_limit=10)
            single_pulses_ar = build_single_pulses_archive(bitsfile, tmp_dir_new=tmp_dir_new)

            waiting_for_memory(used_limit=10)
            folded_ar = build_folded_archive(bitsfile, tmp_dir_new=tmp_dir_new)

            if fichiers_exist([folded_ar]):
                nbin = psredit_nbin(folded_ar)
            else:
                nbin = 1

            if(nbin < 64):
                bscrunch = ''
            else:
                bscrunch = ' -b 2 '

            if fichiers_exist([folded_ar]) and (nbin >= 8):
                # commande = 'python ~/scripts/Undysputed/quicklook_NenuFAR3/NenuPlot.py -defaraday -fit_DM -flat_cleaner -iterative -png  -uploadpdf -metadata_out -u '+tmp_dir_new+' -initmetadata '+bscrunch+folded_ar
                commande = 'python /cep/lofar/pulsar/NenuPlot_v2.py -p -fit_DM -png  -uploadpdf -metadata_out -u ' + tmp_dir_new + ' -initmetadata ' + bscrunch + folded_ar
                waiting_for_memory(used_limit=20)
                output = check_output(commande, shell=True)
            if fichiers_exist([single_pulses_ar]) and (nbin >= 8):
                commande = 'python /cep/lofar/pulsar/NenuPlot_v2.py -p -png  -u ' + tmp_dir_new + '  -initmetadata ' + bscrunch + single_pulses_ar
                waiting_for_memory(used_limit=20)
                output = check_output(commande, shell=True)

            # if(DEBUG):
            #    print(commande)
            #    continue
            try:
                metadata = open(glob.glob(tmp_dir_new + '/*.metadata')[0]).read()
            except:
                metadata = 'no metadata found in tmp directory'
            metadata = metadata + '\n' + fits

            rmdir(tmp_dir_new + '/*.ar')
            rmdir(tmp_dir_new + '/*.fits')
            rmdir(tmp_dir_new + '/*.ps')
            rmdir(tmp_dir_new + '/' + bitsfile)
            tarfile = targz(tmp_dir_new, extension='_singlepulses')
            # rmdir(tmp_dir_new)
        else:  # (psredit_length(fits) > 4*3600.):
            metadata = 'WARNING: file is too long for quicklook > 4h'
            tarfile = None

        mailto = mail[0]
        for i in range(1, len(mail)):
            mailto = mailto + ',' + mail[i]
        mailto = [mailto]
        mail_title = "New observation " + psr_name + ' ' + title

        sendMail(mailto, mail_title, metadata, glob.glob(tmp_dir_new + '/*.png'))
        uptodatabf2(tarfile, '/DATA/' + psr_name + '/SEARCH/')
        uptodatabf2(tmp_dir_new + '/*_ini_search.png', '/quicklook/')

        print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' :' + '/DATA/' + psr_name + '/SEARCH/')
        if tarfile is not None:
            rmdir(tarfile)
        rmdir(tmp_dir_new)
        if(DEBUG):
            print(os.path.basename(parset_path) + '  ' + datetime.utcnow().isoformat() + ' : finish dubug run ' + tmp_dir_new)
            continue


def attach_file(msg, nom_fichier):
    piece = open(nom_fichier, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((piece).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "piece; filename= %s" % os.path.basename(nom_fichier))
    msg.attach(part)


def sendMail(to, subject, text, files=[]):
    msg = MIMEMultipart()
    msg['From'] = socket.gethostname() + '@obs-nancay.fr'
    msg['To'] = ','.join(to)
    msg['Subject'] = subject
    msg.attach(MIMEText(text))
    if (len(files) > 0):
        for ifile in range(len(files)):
            attach_file(msg, files[ifile])
    mailserver = smtplib.SMTP('localhost')
    # mailserver.set_debuglevel(1)
    mailserver.sendmail(msg['From'], msg['To'].split(','), msg.as_string())
    mailserver.quit()


def uptodatabf2(path_from, path_to):
    cmd = """rsync -av -e \"ssh \"  %s nfrplsobs@databfnfrdt:/data/nenufar-pulsar/%s""" % (path_from, path_to)
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output, errors = p.communicate()
    except subprocess.CalledProcessError as e:
        print("subprocess CalledProcessError: while " + cmd)

    print(errors, output)


id_thread = 0
all_thread = []
cont = 0
while(1):  # (cont < 2):
    # if(os.listdir(parset_user_dir+'/')):
    if not (glob.glob(parset_user_dir + '/*.parset') == []):
        for the_file in os.listdir(parset_user_dir + '/'):
            file_path = os.path.join(parset_user_dir + '/', the_file)
            try:
                MAIL, TITLE, MODE, PARAM, SRC, file, stopTime = decode_parset_user(file_path)
            except ValueError:
                print("ValueError! decoding " + the_file + " parset mv to " + parset_user_dir_crash)
                mvfile(file_path, parset_user_dir_crash)
                break
            print(PARAM)
            print(file)
            if not (DEBUG):
                time.sleep(5)
                mvfile(file_path, parset_user_dir_done)
                file_path = os.path.join(parset_user_dir_done + '/', the_file)
            # if not (DEBUG):os.unlink(file_path)
            # sprint(MODE, SRC, file)
            for id_thread in range(len(MODE)):
                if(MODE[id_thread] == 'FOLD') and not (re.search("--notransfer", PARAM[id_thread])):
                    thread = multiprocessing.Process(target=folding_thread, args=(SRC[id_thread], file[id_thread], MAIL, TITLE, stopTime, file_path))
                    all_thread.append(thread)
                    all_thread[-1].start()
                elif(MODE[id_thread] == 'SINGLE') and not (re.search("--notransfer", PARAM[id_thread])):
                    for thread in all_thread:
                        thread.join()
                    single_thread = multiprocessing.Process(target=single_pulse_thread, args=(
                        SRC[id_thread], file[id_thread], MAIL, TITLE, stopTime, file_path))
                    single_thread.start()
                    single_thread.join()
                else:
                    print(os.path.basename(file_path) + '  ' + datetime.utcnow().isoformat() + ' :' + 'OTHER')
                    # if not (DEBUG):os.unlink(file_path)
        for thread in all_thread:
            thread.join()
        if(DEBUG):
            exit(0)
        time.sleep(20)
    else:
        time.sleep(20)
    cont += 1
