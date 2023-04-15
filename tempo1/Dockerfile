FROM ubuntu:20.04

RUN apt update
RUN apt -y install build-essential
RUN apt -y install wget rsync csh
RUN apt -y install locate git curl
RUN apt -y install autotools-dev autoconf 
RUN apt -y install libtool make g++ gfortran
RUN apt -y install libfftw3-dev pgplot5
RUN mkdir /home/root
RUN mkdir /home/root/pulsar
ENV HOME=/home/root
ENV PSRHOME=/home/root/pulsar

RUN echo 'export PSRHOME=$HOME/pulsar' >> ~/.bashrc
RUN echo 'export PATH=${PATH}:$PSRHOME/bin' >> ~/.bashrc
RUN echo 'export PGPLOT_DIR=/usr/lib/pgplot5' >> ~/.bashrc
RUN echo 'export PGPLOT_FONT=$PGPLOT_DIR/grfont.dat' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PSRHOME/lib' >> ~/.bashrc
RUN echo 'export C_INCLUDE_PATH=$C_INCLUDE_PATH:$PSRHOME/include' >> ~/.bashrc
RUN echo 'export TEMPO=$PSRHOME/tempo' >> ~/.bashrc

RUN cd $HOME/pulsar && \
    wget http://ftp.debian.org/debian/pool/main/c/cfitsio/libcfitsio9_3.490-3_amd64.deb && \
    wget http://ftp.br.debian.org/debian/pool/main/c/cfitsio/libcfitsio-dev_3.490-3_amd64.deb && \
    dpkg -i libcfitsio9_3.490-3_amd64.deb && \
    dpkg -i libcfitsio-dev_3.490-3_amd64.deb


RUN cd $PSRHOME && \
    git clone https://github.com/nanograv/tempo && \
    cd tempo && \
    ./prepare && \
    ./configure && \
    make && \
    make install

#### configuration file for tempo2, tempo, psrchive
COPY tempo_config/obsys.dat /home/root/pulsar/tempo/obsys.dat


# lancez le serveur X11 avec :
# open -a XQuartz
# autorisez les connexions locales avec :
# xhost +localhost
# lancer le conteneur Docker avec :
# docker run -it --env="DISPLAY=host.docker.internal:0" psrchive-docker /bin/bash
# ce qui peut prendre du temps la première fois à cause du téléchargement de l'image.