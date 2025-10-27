# Dockerfile équivalent à la recette Singularity fournie
# Base
# FROM ubuntu:22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04


SHELL ["/bin/bash","-lc"]

# --- Tags / versions utilisées dans la recette ---
# ARG GIT_PSRCHIVE_TAG=d706adce0b3faeeb7caa4dbb3f365e59dd06b299
# ARG GIT_DSPSR_TAG=879158a8e98a1ffbe99aa6cd06ea57a30508df79
ARG GIT_PRESTO_TAG=v5.1.0

# ENV GIT_PSRCHIVE_TAG=${GIT_PSRCHIVE_TAG}
# ENV GIT_DSPSR_TAG=${GIT_DSPSR_TAG}
ENV GIT_PRESTO_TAG=${GIT_PRESTO_TAG}

# --- Fichiers de config ---
COPY config/ /config/

# --- Python 3 + dépendances système ---
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Paris

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get -y upgrade && \
    apt-get -y install \
    python3-pyqt5 python3-pyqt5.qtwebengine python3-tk python3-distutils \
    build-essential manpages-dev htop wget tmux rsync locate git curl \
    autotools-dev autoconf libtool make g++ gfortran \
    libfftw3-dev libfftw3-single3 libfftw3-mpi-dev libgsl-dev pgplot5 libxml2-dev libxml2-utils \
    python3-pip csh tcsh swig python-tk pkg-config x11-apps gedit ghostscript \
    libpng-dev libx11-dev libglib2.0-dev mpich openmpi-bin g++-10 gcc-10 \
    firefox imagemagick gnuplot s-nail bc libboost-all-dev f2c mailutils  \
    software-properties-common

# RUN pip3 install numpy==1.21 && \
RUN pip3 install numpy && \
    pip3 install matplotlib astropy astroplan scipy tinydb lmfit && \
    pip3 install psrqpy astroquery mailutils && \
    pip3 install pygedm plotly && \
    pip3 install --upgrade widgetsnbextension ipykernel

RUN apt-get -y install  python3.10-distutils python3.10-venv python3.10-dev 

# LATEX
RUN apt-get -y install texlive-latex-base texlive-extra-utils

# --- Variables d'environnement de la recette ---
ENV PSRHOME=/usr/local/pulsar
ENV PGPLOT_DIR=/usr/lib/pgplot5 \
    PGPLOT_FONT=/usr/lib/pgplot5/grfont.dat \
    PRESTO=${PSRHOME}/presto \
    TEMPO2=${PSRHOME}/tempo2 \
    TEMPO=${PSRHOME}/tempo \
    PSRCAT_FILE=${PSRHOME}/psrcat/psrcat.db \
    PSRCHIVE_CONFIG=${PSRHOME}/.psrchive.cfg
ENV NVCCFLAGS="--std=c++14 --expt-relaxed-constexpr -Xcompiler -fPIC"

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PSRHOME}/lib:${PSRHOME}/lib/x86_64-linux-gnu \
    C_INCLUDE_PATH=${C_INCLUDE_PATH}:${PSRHOME}/include:${PSRHOME}/tempo2/include \
    PYTHONPATH=${PYTHONPATH}:${PSRHOME}/presto/lib/python \
    PATH=${PATH}:${PSRHOME}/presto/bin:${PSRHOME}/bin:${PSRHOME}/tempo2/bin \
    PKG_CONFIG_PATH=${PSRHOME}/lib/pkgconfig:${PSRHOME}/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH} \
    CPPFLAGS="-I${PSRHOME}/lib -I$PSRHOME/lib/x86_64-linux-gnu -I$PSRHOME/include" \
    LDFLAGS="-L${PSRHOME}/lib -L$PSRHOME/lib/x86_64-linux-gnu -L$PSRHOME/include"



RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    mkdir -p $PSRHOME/src && \
    # Remplacer le bashrc par celui fourni
    cat /config/bashrc > /etc/bash.bashrc
    # CFITSIO (versions spécifiques Debian comme dans la recette)
    # cd $PSRHOME && \
    # wget http://ftp.debian.org/debian/pool/main/c/cfitsio/libcfitsio9_3.490-3_amd64.deb && \
    # wget http://ftp.br.debian.org/debian/pool/main/c/cfitsio/libcfitsio-dev_3.490-3_amd64.deb && \
    # dpkg -i libcfitsio9_3.490-3_amd64.deb && \
    # dpkg -i libcfitsio-dev_3.490-3_amd64.deb

# 
# # --- Outils GPU (sans driver) ---
# RUN apt-get -y install pciutils mesa-utils xvfb && \
#     echo 'export LIBGL_ALWAYS_SOFTWARE=1' >> /etc/bash.bashrc
# ENV LIBGL_ALWAYS_SOFTWARE=1
# 
# # --- Python 2.7 (comme dans la recette) ---
RUN apt-get -y install python2-dev python2.7 && \
    curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py && \
    python2.7 get-pip.py && \
    pip2 install --user numpy && \
    pip2 install --user matplotlib astropy astroplan scipy tinydb lmfit && \
    pip2 install --user pyfits


# Et on les ajoute aussi au bashrc comme dans la recette
# RUN echo 'export PSRHOME=$PSRHOME' >> /etc/bash.bashrc && \
#     echo 'export PGPLOT_DIR=/usr/lib/pgplot5' >> /etc/bash.bashrc && \
#     echo 'export PGPLOT_FONT="$PGPLOT_DIR"/grfont.dat' >> /etc/bash.bashrc && \
#     echo 'export PRESTO="$PSRHOME"/presto' >> /etc/bash.bashrc && \
#     echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":"$PSRHOME"/lib:"$PRESTO"/lib' >> /etc/bash.bashrc && \
#     echo 'export C_INCLUDE_PATH="$C_INCLUDE_PATH":"$PSRHOME"/include:"$PSRHOME"/tempo2/include' >> /etc/bash.bashrc && \
#     echo 'export PATH="$PATH":$PRESTO/bin' >> /etc/bash.bashrc && \
#     echo 'export TEMPO2="$PSRHOME"/tempo2' >> /etc/bash.bashrc && \
#     echo 'export TEMPO="$PSRHOME"/tempo' >> /etc/bash.bashrc && \
#     echo 'export PSRCAT_FILE="$PSRHOME"/psrcat/psrcat.db' >> /etc/bash.bashrc && \
#     echo 'export PSRCHIVE_CONFIG="$PSRHOME"/.psrchive.cfg' >> /etc/bash.bashrc && \
#     echo 'export PATH=${PATH}:"$PSRHOME"/bin:"$TEMPO2"/bin' >> /etc/bash.bashrc && \
#     echo 'export PYTHONPATH="$PYTHONPATH":$PRESTO/lib/python' >> /etc/bash.bashrc
# 
# --- TEMPO2 ---
RUN cd $PSRHOME && \
    git clone https://bitbucket.org/psrsoft/tempo2.git && \
    cd $TEMPO2 && \
    ./bootstrap && \
    ./configure --prefix=$PSRHOME && \
    make && make install && \
    make plugins && make plugins-install && \
    cp -r ${TEMPO2}/T2runtime/* ${TEMPO2}/

ENV CPPFLAGS="${CPPFLAGS} -I${PSRHOME}/tempo2"


# Install PSRCAT
RUN cd $PSRHOME \
    && wget http://www.atnf.csiro.au/research/pulsar/psrcat/downloads/psrcat_pkg.tar.gz \
    && tar --no-same-owner -xzvf psrcat_pkg.tar.gz \
    && cd psrcat_tar \
    && sh makeit \
    && mkdir -p $PSRHOME/bin || true \
    && mv psrcat $PSRHOME/bin \
    && mkdir -p $PSRHOME/psrcat \
    && mv psrcat.db $PSRHOME/psrcat/

# --- PSRCHIVE ---
RUN cd $PSRHOME && \
    git clone http://git.code.sf.net/p/psrchive/code psrchive && \
    cd $PSRHOME/psrchive && \
    git checkout master && \
    # git checkout $GIT_PSRCHIVE_TAG && \
    ./bootstrap && \
    ./configure --prefix=$PSRHOME || true && \
    cd packages && make && \
    cd $PSRHOME/psrchive && \
    ./packages/fftw.csh

# unlock http request issues from http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-3.49.tar.gz -> cfitsio build 
# RUN echo "use_ipv6 = off" > ~/.wgetrc # unlock 
# RUN printf 'inet4_only = on\n' > /etc/wgetrc

RUN cd $PSRHOME/psrchive && \
    cfitsio="cfitsio-3.49" && \
    place="https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c" && \
    curl -v -L -o ${PSRHOME}/src/${cfitsio}.tar.gz "${place}/${cfitsio}.tar.gz" && \
    ./packages/cfitsio.csh && \
    ./configure --prefix="$PSRHOME" --enable-shared --with-x --with-psrcat="$PSRHOME/bin/psrcat" && \
    make -j"$(nproc)" && \
    make install && \
    echo "$PSRHOME/lib/python3.10/site-packages" > /usr/local/lib/python3.10/dist-packages/mymodule.pth

# --- TEMPO 1 ---
RUN cd $PSRHOME && \
    rm -rf tempo || true && \
    git clone https://github.com/nanograv/tempo && \
    cd tempo && \
    ./prepare && ./configure && make -j8 && make install

# --- Outils graphiques divers ---
RUN apt-get update && apt-get install -y x11-apps gedit gimp evince nano vim emacs subversion ghostscript
#
# --- PRESTO ---
RUN apt-get install -y cmake libcfitsio-bin libcfitsio-dev libpng-dev latex2html libx11-dev libglib2.0-dev mpich openmpi-bin && \
    cd $PSRHOME && \
    git clone https://github.com/scottransom/presto presto && \
    cd $PSRHOME/presto/ && \
    git checkout master && \
    git checkout $GIT_PRESTO_TAG

RUN cd $PRESTO && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install meson meson-python ninja && \
    meson setup build --prefix=$PSRHOME && \
    meson compile -C build && \
    meson install -C build && \
    cd $PRESTO/python && \
    LDFLAGS="-L$PSRHOME/lib/x86_64-linux-gnu" \
    CPPFLAGS="-I$PSRHOME/include" \
    pip3 install . -v

# --- SIGPROC ---
# RUN cd $PSRHOME && \
#     git clone https://github.com/SixByNine/sigproc && \
#     cd sigproc && \
#     ./bootstrap && \
#     ./configure --prefix=$PSRHOME && \
#     make && make install

# --- RFIClean ---
# RUN cd $PSRHOME && \
#     git clone https://github.com/ymaan4/RFIClean.git && \
#     cd RFIClean && \
#     sed -i "s#MYBIN=.*#MYBIN=$PSRHOME/bin/#" Makefile && \
#     make && make install
# 
# --- GPTOOL ---
RUN cd $PSRHOME && \
    git clone https://github.com/chowdhuryaditya/gptool.git && \
    cd gptool && \
    sed -i "s#PGPLOT_INCL=.*#PGPLOT_INCL=-I$PGPLOT_DIR/include#" makefile && \
    sed -i "s#PGPLOT_LIB=.*#PGPLOT_LIB=-L$PGPLOT_DIR/lib#" makefile && \
    make -j8 && \
    mv gptool $PSRHOME/bin/ && \
    cp plotgptoolsummary.py $PSRHOME/bin/
# 
# --- DSPSR ---
RUN cd $PSRHOME && \
    git clone git://git.code.sf.net/p/dspsr/code dspsr && \
    cd $PSRHOME/dspsr && \
    git checkout master && \
    # git checkout $GIT_DSPSR_TAG && \
    ./bootstrap && \
    echo  'apsr asp bcpm bpsr cpsr cpsr2 gmrt lbadr lbadr64 mark4 mark5 maxim mwa pdev pmdaq puma2 sigproc psrfits caspsr dummy fits spda1k spigot guppi lump' > $PSRHOME/dspsr/backends.list && \
    ./configure --prefix=$PSRHOME --with-x && \
    # ./configure --prefix=$PSRHOME --enable-shared --with-x && \
    make -j8 && make install
# 
# # --- PSRSALSA ---
# RUN set -e; \
#     gsl_version=`gsl-config --version`; \
#     gsl_version=`python -c "import numpy as np; print('%d0%d' % (np.floor($gsl_version), 10*($gsl_version % 1)))"`; \
#     cd "$PSRHOME" && \
#     git clone https://github.com/weltevrede/psrsalsa.git && \
#     cd psrsalsa && \
#     sed -i "s/GSLFLAGS *= *-DGSL_VERSION_NUMBER=.*/GSLFLAGS = -DGSL_VERSION_NUMBER=${gsl_version}/" Makefile && \
#     make && \
#     echo 'export PATH="$PATH":$PSRHOME/psrsalsa/bin' >> /etc/bash.bashrc
# 
# --- NENUPY ---
RUN cd $PSRHOME/src && \
    git clone https://github.com/AlanLoh/nenupy.git && \
    cd $PSRHOME/src/nenupy/ && \
    pip3 install -r requirements.txt && \
    pip3 install .

# --- PyGSM ---
RUN cd $PSRHOME/src && \
    git clone https://github.com/telegraphic/PyGSM && \
    cd PyGSM && \
    curl -v -L -o gsm2016_components.h5 "https://zenodo.org/record/3479985/files/gsm2016_components.h5?download=1" && \
    curl -v -L -o gsm_components.h5 "https://zenodo.org/record/3479985/files/gsm_components.h5?download=1" && \
    python3 setup.py install

# --- CASACORE_DATA ---
RUN bash /config/get_casacore_data.sh

# --- ANTPAT ---
RUN pip3 install python-casacore && \
    cd $PSRHOME/src && \
    git clone https://github.com/2baOrNot2ba/AntPat.git && \
    cd $PSRHOME/src/AntPat/ && \
    python3 setup.py install

# --- DREAMBEAM ---
RUN cd $PSRHOME/src && \
    git clone https://github.com/2baOrNot2ba/dreamBeam.git && \
    cd $PSRHOME/src/dreamBeam && \
    python3 setup.py install

## --- PYGLOW ---
#RUN cd $PSRHOME/src && \
#    pip3 install future && \
#    git clone https://github.com/timduly4/pyglow.git pyglow && \
#    cd $PSRHOME/src/pyglow && \
#    make -C src/pyglow/models source && \
#    python3 setup.py install

# # --- SOFA / PlotX / TransientX (commentés comme dans la recette) ---
# # RUN cd $PSRHOME/src && \
# #     wget https://www.iausofa.org/2020_0721_C/sofa_c-20200721.tar.gz --no-check-certificate && \
# #     tar -zxvf sofa_c-20200721.tar.gz && \
# #     cd $PSRHOME/src/sofa/20200721/c/src && make && make test && \
# #     cd $PSRHOME/src && \
# #     git clone https://github.com/ypmen/PlotX.git PlotX && \
# #     git clone https://github.com/ypmen/TransientX.git TransientX && \
# #     cd $PSRHOME/src/PlotX && ./bootstrap && ./configure --prefix=$PSRHOME && make && make install && make clean && \
# #     cd $PSRHOME/src/TransientX && \
# #     ./bootstrap && \
# #     ./configure --prefix=$PSRHOME CXXFLAGS="-std=c++11 -O3" LDFLAGS="-L$PSRHOME/src/sofa/20200721/c/src -L$PSRHOME/lib" CPPFLAGS="-I$PSRHOME/src/sofa/20200721/c/src -I$PSRHOME/include" && \
# #     make && make install && make clean
# 
# --- Déplacement final des fichiers de config ---
RUN mv /config/observatories.dat $PSRHOME/tempo2/observatory/observatories.dat && \
    mv /config/obsys.dat $PSRHOME/tempo/obsys.dat && \
    mv /config/psrchive.cfg $PSRHOME/.psrchive.cfg && \
    mv /config/clk_dir $PSRHOME/ && \
    mv /config/plk_setup_fr606_embrace.dat $PSRHOME/

# (Optionnel) Dossier de travail
WORKDIR $PSRHOME
