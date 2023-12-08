FROM ubuntu:22.04

RUN apt update
RUN apt -y install build-essential
RUN apt -y install wget rsync
RUN apt -y install locate git curl

ENV HOME=/home/root

RUN apt-get install -y csh swig python2-dev
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt install -y x11-apps gedit nano vi

# python3 install
RUN apt -y install python3-pip && \
    pip3 install numpy && \
    pip3 install matplotlib astropy astroplan scipy tinydb pyfits

# python2.7 install
RUN apt -y install python2.7 -y && \
    curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py && \
    python2.7 get-pip.py && \
    pip2 install numpy && \
    pip2 install matplotlib astropy astroplan scipy tinydb pyfits && \
    apt -y install python-tk

# https://gitlab.obspm.fr/nrt_dir/channelizer_daemon
RUN pip2 install psutil IPython && \
    cd /home/root && \
    git clone --recurse-submodules https://oauth2:glpat-zRHT-N7AsGT2yMxvs_Fy@gitlab.obspm.fr/nrt_dir/channelizer_daemon.git && \
    cd /home/root/channelizer_daemon

# fork de https://github.com/Smithsonian/adc_tests.git
RUN cd /home/root/ && \
    git clone https://github.com/CedricDViou/adc_tests.git && \
    cd /home/root/adc_tests && \
    git checkout master && \
    git checkout 86e4cfb && \
    python2 setup.py install

RUN cd /home/root/channelizer_daemon/adc5g_devel && \
    pip2 install -r requirements.txt && \
    pip2 install corr==0.6.7 && \
    pip2 install katcp==0.6.2

RUN pip2 install pyserial && \
    cd /home/root/channelizer_daemon/ValonSynth && \
    python2 setup.py build && \
    python2 setup.py install
#     usermod -a -G dialout $USER

RUN cd /home/root && \
    git clone https://github.com/casper-astro/tftpy.git && \
    cd tftpy && \
    git checkout master && \
    git checkout b87bd52 && \
    python2 setup.py install

RUN cd /home/root && \
    git clone https://github.com/casper-astro/casperfpga && \
    cd casperfpga && \
    git checkout master && \
    git checkout 0fed055d1c62a93dff68afec32b0c9ada776b07d && \
    pip2 install -r requirements.txt || true && \
    python2 setup.py install

RUN cd /home/root && \
    git clone https://github.com/CedricDViou/NRT_2G_channelizer.git


#
# docker build --build-arg GITHUB_TOKEN=glpat-G8RtsM9_yVkZvZwzwm7w -t my_first_image /Users/louis/Docker/Docker_receipts/Ubuntu_py2
# lancez le serveur X11 avec :
# open -a XQuartz
# autorisez les connexions locales avec :
# xhost +localhost
# lancer le conteneur Docker avec :
# docker run -it --env="DISPLAY=host.docker.internal:0" psrchive-docker /bin/bash
# ce qui peut prendre du temps la première fois à cause du téléchargement de l'image.
# 
# Once you have setup the environment, check that all the unit tests pass:
# 
#    * cd /home/root/channelizer_daemon/adc5g_devel/test
#    * python2 AllTests.py