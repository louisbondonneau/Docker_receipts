FROM ubuntu:20.04

RUN apt update
RUN apt -y install build-essential
RUN apt -y install wget rsync
RUN apt -y install locate git curl
RUN apt -y install autotools-dev autoconf 
RUN apt -y install libtool make g++ gfortran
RUN apt -y install libfftw3-dev pgplot5

ENV HOME=/home/root

RUN cd /home && \
    mkdir $HOME || true && \
    cd $HOME

RUN apt -y install python3-pip
RUN pip3 install numpy==1.21 matplotlib astropy astroplan scipy  --user
RUN pip3 install tinydb lmfit pyfits --user

ENV HOME=/home/root
ENV PSRHOME=/home/root/pulsar
RUN mkdir $PSRHOME

RUN echo 'export PSRHOME=$HOME/pulsar' >> ~/.bashrc
RUN echo 'export PATH=${PATH}:$PSRHOME/bin' >> ~/.bashrc
RUN echo 'export PGPLOT_DIR=/usr/lib/pgplot5' >> ~/.bashrc
RUN echo 'export PGPLOT_FONT=$PGPLOT_DIR/grfont.dat' >> ~/.bashrc
# RUN echo 'export TEMPO2=$PSRHOME/src/tempo2/T2runtime' >> ~/.bashrc
RUN echo 'export TEMPO2=$PSRHOME/tempo2' >> ~/.bashrc
RUN echo 'export PATH=$TEMPO2/bin:$PATH' >> ~/.bashrc
RUN echo 'export PSRCAT_FILE=$PSRHOME/psrcat/psrcat.db' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PSRHOME/lib' >> ~/.bashrc
RUN echo 'export C_INCLUDE_PATH=$C_INCLUDE_PATH:$PSRHOME/include' >> ~/.bashrc
RUN echo 'export PSRCHIVE_CONFIG=$HOME/.psrchive.cfg' >> ~/.bashrc
RUN echo '# export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:/$PYTHONPATH' >> ~/.bashrc
RUN echo '# export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:/$PYTHONPATH' >> ~/.bashrc
RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN cd $HOME/pulsar && \
    wget http://ftp.debian.org/debian/pool/main/c/cfitsio/libcfitsio9_3.490-3_amd64.deb && \
    wget http://ftp.br.debian.org/debian/pool/main/c/cfitsio/libcfitsio-dev_3.490-3_amd64.deb && \
    dpkg -i libcfitsio9_3.490-3_amd64.deb && \
    dpkg -i libcfitsio-dev_3.490-3_amd64.deb

RUN apt-get install -y csh swig
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y python-tk
RUN apt-get install -y pkg-config


RUN cd $PSRHOME && \
    git clone git://git.code.sf.net/p/psrchive/code psrchive

RUN cd $PSRHOME/psrchive && \
    ./bootstrap
RUN cd $PSRHOME/psrchive && \
    ./configure --prefix=$PSRHOME || true && \
    cd packages && make
RUN cd $PSRHOME/psrchive && \
    ./packages/fftw.csh
RUN cd $PSRHOME/psrchive && \
    ./packages/cfitsio.csh && \
    ./configure --prefix=$PSRHOME || true
RUN cd $PSRHOME/psrchive && \
    ./packages/tempo2.csh&& \
    ./configure --prefix=$PSRHOME || true
RUN cd $PSRHOME/psrchive && \
    ./packages/psrcat.csh && \
    ./configure --prefix=$PSRHOME --enable-shared --with-x --with-psrcat=psrcat
RUN cd $PSRHOME/psrchive && \
    ./configure --prefix=$PSRHOME --enable-shared --with-x && \
    make -j 6 && \
    make install

RUN echo 'export TEMPO=$PSRHOME/tempo' >> ~/.bashrc
RUN cd $PSRHOME && \
    git clone https://github.com/nanograv/tempo && \
    cd tempo && \
    ./prepare && \
    ./configure && \
    make && \
    make install

RUN echo '/home/root/pulsar/lib/python3.8/site-packages' > /usr/local/lib/python3.8/dist-packages/mymodule.pth

RUN apt install -y x11-apps gedit
RUN apt -y install ghostscript

#### configuration file for tempo2, tempo, psrchive
COPY tempo_config/observatories.dat /home/root/pulsar/tempo2/observatory/observatories.dat
COPY tempo_config/obsys.dat /home/root/pulsar/tempo/obsys.dat
COPY tempo_config/psrchive.cfg /home/root/.psrchive.cfg

# #### PRESTO
# # v2.2maint compilation crach under ubuntu 22.04
RUN apt -y install libpng-dev libx11-dev libglib2.0-dev mpich openmpi-bin

ENV PRESTO=$PSRHOME/presto
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH":$PSRHOME/lib:$PRESTO/lib
ENV PATH="$PATH":$PRESTO/bin
# ENV PYTHONPATH="$PYTHONPATH":$PRESTO/lib/python
RUN echo '/home/root/pulsar/presto/lib/python' >> /usr/local/lib/python3.8/dist-packages/mymodule.pth

RUN cd $PSRHOME && \
    git clone https://github.com/scottransom/presto presto
#     cd $PSRHOME/presto/ && \
#     echo 'export PRESTO=$PSRHOME/presto' >> ~/.bashrc && \
#     echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PSRHOME/lib:$PRESTO/lib' >> ~/.bashrc && \
#     echo 'export PATH=${PATH}:$PRESTO/bin' >> ~/.bashrc && \
#     echo 'export PYTHONPATH=${PYTHONPATH}:$PRESTO/lib/python' >> ~/.bashrc

RUN su root && \
    . /home/root/.bashrc && \
    cd $PRESTO/src && \
    make makewisdom && \
    make clean && \
    make prep && \
    make

# ERROR: pyslalib 1.0.6 has requirement numpy<1.22, but you'll have numpy 1.24.2 which is # incompatible.
RUN cd $PRESTO && \
    pip3 install ./ --user 

# nenupy
RUN pip3 install nenupy --user && \
    cd $PSRHOME/src && \
    git clone https://github.com/telegraphic/PyGSM && \
    cd PyGSM && \
    wget -O gsm2016_components.h5 https://zenodo.org/record/3479985/files/gsm2016_components.h5?download=1 && \
    wget -O gsm_components.h5 https://zenodo.org/record/3479985/files/gsm_components.h5?download=1 && \
    python3 setup.py install --user

# AntPat
RUN pip3 install python-casacore --user && \
    cd $PSRHOME/src && \
    git clone https://github.com/2baOrNot2ba/AntPat.git && \
    cd $PSRHOME/src/AntPat/ && \
    python3 setup.py install --user

# dreamBeam
RUN cd $PSRHOME/src && \
    git clone https://github.com/2baOrNot2ba/dreamBeam.git && \
    cd $PSRHOME/src/dreamBeam && \
    python3 setup.py install --user


# PRESTO='/home/root/pulsar/presto_py2'
# 
# ENV PRESTO=$PSRHOME/presto_py2
# ENV LD_LIBRARY_PATH=$PSRHOME/lib:$PRESTO/lib
# ENV PATH="$PATH":$PRESTO/bin
# 
# 
# apt -y install python2-dev
# rm /usr/bin/python
# ln -s /usr/bin/python2.7 /usr/bin/python
# curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
# python2.7 get-pip.py
# pip2 install numpy matplotlib astropy astroplan scipy tinydb lmfit
# pip2 install pyfits
# cd $PSRHOME/psrchive/
# 
# ./configure --prefix=$PSRHOME --enable-shared --with-x
# make -j 6
# make install
# 
# echo '/home/root/pulsar/lib/python2.7/site-packages' > /usr/local/lib/python2.7/dist-packages/mymodule.pth
# 
# 
# 
# git clone -b v2.2maint https://github.com/scottransom/presto presto_py2
# cd $PSRHOME/presto_py2/
# cd $PSRHOME/presto_py2/src && \
# make makewisdom && \
# make clean && \
# make prep && \
# make
# cd $PSRHOME/presto_py2/python
# pip2 install ./
# echo '/home/root/pulsar/presto_py2/python' >> /usr/local/lib/python2.7/dist-packages/mymodule.pth
# python2 -c 'import presto'


# lancez le serveur X11 avec :
# open -a XQuartz
# autorisez les connexions locales avec :
# xhost +localhost
# lancer le conteneur Docker avec :
# docker run -it --env="DISPLAY=host.docker.internal:0" psrchive-docker /bin/bash
# ce qui peut prendre du temps la première fois à cause du téléchargement de l'image.

#  ubuntu 22.04
# RUN cd $HOME/pulsar && \
#     wget http://archive.ubuntu.com/ubuntu/pool/universe/c/cfitsio/libcfitsio10_4.2.0-3_amd64.deb && \
#     wget http://archive.ubuntu.com/ubuntu/pool/universe/c/cfitsio/libcfitsio-dev_4.2.0-3_amd64.deb && \
#     dpkg -i libcfitsio10_4.2.0-3_amd64.deb && \
#     dpkg -i libcfitsio-dev_4.2.0-3_amd64.deb
# 