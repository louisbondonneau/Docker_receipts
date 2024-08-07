FROM ubuntu:20.04

# Define variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris
ENV PSRHOME=/usr/local/pulsar
ENV PGPLOT_DIR=/usr/lib/pgplot5
ENV PGPLOT_FONT=$PGPLOT_DIR/grfont.dat
ENV PRESTO=$PSRHOME/presto
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PSRHOME/lib:$PRESTO/lib
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$PSRHOME/include:$PSRHOME/tempo2/include
ENV PATH=$PATH:$PRESTO/bin
ENV TEMPO2=$PSRHOME/tempo2
ENV TEMPO=$PSRHOME/tempo
ENV PSRCAT_FILE=$PSRHOME/psrcat/psrcat.db
ENV PSRCHIVE_CONFIG=$PSRHOME/.psrchive.cfg
ENV PATH=$PATH:$PSRHOME/bin:$TEMPO2/bin
ENV PYTHONPATH=$PYTHONPATH:$PRESTO/lib/python
ENV LIBGL_ALWAYS_SOFTWARE=1

# Copy configuration files
COPY config/bashrc /etc/bash.bashrc
COPY config/observatories.dat /usr/local/pulsar/tempo2/observatory/observatories.dat
COPY config/obsys.dat /usr/local/pulsar/tempo/obsys.dat
COPY config/psrchive.cfg /usr/local/pulsar/.psrchive.cfg
COPY config/clk_dir /usr/local/pulsar/clk_dir
COPY config/plk_setup_fr606_embrace.dat /usr/local/pulsar/plk_setup_fr606_embrace.dat
COPY config/get_casacore_data.sh /config/get_casacore_data.sh

# Update and install dependencies
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt update && apt -y upgrade \
    && apt -y install python3-pyqt5 python3-pyqt5.qtwebengine \
    python3-tk build-essential manpages-dev htop wget tmux rsync locate git curl autotools-dev autoconf libtool make g++ gfortran libfftw3-dev libfftw3-single3 libfftw3-mpi-dev libgsl-dev pgplot5 python3-pip csh tcsh swig python-tk pkg-config x11-apps gedit ghostscript libpng-dev libx11-dev libglib2.0-dev mpich openmpi-bin nvidia-cuda-toolkit firefox imagemagick gnuplot s-nail bc libboost-all-dev f2c \
    && pip3 install numpy==1.21 matplotlib astropy astroplan scipy tinydb lmfit psrqpy astroquery mailutils pygedm plotly widgetsnbextension --upgrade ipykernel \
    && apt -y install software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt -y install python3.11 python3.11-distutils python3.11-venv python3.8-venv python3.11-dev \
    && ln -s /usr/bin/python3.8 /usr/bin/python \
    && mkdir -p $PSRHOME/src \
    && wget http://ftp.debian.org/debian/pool/main/c/cfitsio/libcfitsio9_3.490-3_amd64.deb \
    && wget http://ftp.br.debian.org/debian/pool/main/c/cfitsio/libcfitsio-dev_3.490-3_amd64.deb \
    && dpkg -i libcfitsio9_3.490-3_amd64.deb \
    && dpkg -i libcfitsio-dev_3.490-3_amd64.deb \
    && apt-get install -y csh swig \
    && apt-get update \
    && apt-get install -y python-tk pkg-config texlive-latex-base texlive-extra-utils \
    && apt-get -y install pciutils mesa-utils xvfb \
    && echo 'export LIBGL_ALWAYS_SOFTWARE=1' >> /etc/bash.bashrc \
    && apt -y install python2-dev python2.7 \
    && curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py \
    && python2.7 get-pip.py \
    && pip2 install numpy matplotlib astropy astroplan scipy tinydb lmfit pyfits --user

# # Install SurrealDB
# RUN curl -sSf https://install.surrealdb.com | sh \
#     && wget https://github.com/StarlaneStudios/Surrealist/releases/download/v1.11.4/surrealist-v1.11.4-linux.deb \
#     && dpkg -i surrealist-v1.11.4-linux.deb \
#     && pip3 install surrealdb

# Install TEMPO2
RUN cd $PSRHOME/src \
    && git clone https://bitbucket.org/psrsoft/tempo2.git \
    && cd tempo2 \
    && ./bootstrap \
    && ./configure --prefix=$PSRHOME \
    && make \
    && make install \
    && make plugins \
    && make plugins-install \
    && cp -r T2runtime/* $TEMPO2/

# Install PSRCHIVE
ARG GIT_PSRCHIVE_TAG="d706adce0b3faeeb7caa4dbb3f365e59dd06b299"
RUN cd $PSRHOME \
    && git clone http://git.code.sf.net/p/psrchive/code psrchive \
    && cd psrchive \
    && git checkout master \
    && git checkout $GIT_PSRCHIVE_TAG \
    && ./bootstrap \
    && ./configure --prefix=$PSRHOME || true \
    && cd packages && make \
    && cd $PSRHOME/psrchive \
    && ./packages/fftw.csh \
    && ./packages/cfitsio.csh \
    && ./configure --prefix=$PSRHOME || true

# Install PSRCAT
RUN cd $PSRHOME/src \
    && wget http://www.atnf.csiro.au/research/pulsar/psrcat/downloads/psrcat_pkg.tar.gz \
    && tar --no-same-owner -xzvf psrcat_pkg.tar.gz \
    && cd psrcat_tar \
    && csh makeit \
    && mkdir -p /usr/local/pulsar/bin || true \
    && mv psrcat /usr/local/pulsar/bin \
    && mkdir -p /usr/local/pulsar/psrcat \
    && mv psrcat.db /usr/local/pulsar/psrcat \
    && cd $PSRHOME/psrchive \
    && ./configure --prefix=$PSRHOME --enable-shared --with-x --with-psrcat=psrcat \
    && make -j 6 \
    && make install \
    && echo '/usr/local/pulsar/lib/python3.8/site-packages' > /usr/local/lib/python3.8/dist-packages/mymodule.pth

# Install TEMPO
RUN cd $PSRHOME \
    && rm -r tempo || true \
    && git clone https://github.com/nanograv/tempo \
    && cd tempo \
    && ./prepare \
    && ./configure \
    && make \
    && make install

# Install PRESTO
ARG GIT_PRESTO_TAG="v4.0"
RUN apt-get install -y libpng-dev libx11-dev libglib2.0-dev mpich openmpi-bin \
    && cd $PSRHOME \
    && git clone https://github.com/scottransom/presto presto \
    && cd presto/ \
    && git checkout master \
    && git checkout $GIT_PRESTO_TAG \
    && . /etc/bash.bashrc \
    && cd $PRESTO/src \
    && make makewisdom \
    && make clean \
    && make prep \
    && make \
    && cd $PRESTO \
    && pip3 install ./

# Install SIGPROC
RUN . /etc/bash.bashrc \
    && cd $PSRHOME \
    && git clone https://github.com/SixByNine/sigproc \
    && cd sigproc \
    && ./bootstrap \
    && ./configure --prefix=$PSRHOME \
    && make \
    && make install

# Install RFICLEAN
RUN . /etc/bash.bashrc \
    && cd $PSRHOME \
    && git clone https://github.com/ymaan4/RFIClean.git \
    && cd RFIClean \
    && sed -i "s#MYBIN=.*#MYBIN=/usr/local/pulsar/bin/#" Makefile \
    && make \
    && make install

# Install GPTOOL
RUN . /etc/bash.bashrc \
    && cd $PSRHOME \
    && git clone https://github.com/chowdhuryaditya/gptool.git \
    && cd gptool \
    && sed -i "s#PGPLOT_INCL=.*#PGPLOT_INCL=-I$(PGPLOT_DIR)/include#" makefile \
    && sed -i "s#PGPLOT_LIB=.*#PGPLOT_LIB=-L$(PGPLOT_DIR)/lib#" makefile \
    && make \
    && mv gptool /usr/local/pulsar/bin/ \
    && cp plotgptoolsummary.py /usr/local/pulsar/bin/

# Install DSPSR
ARG GIT_DSPSR_TAG="879158a8e98a1ffbe99aa6cd06ea57a30508df79"
RUN . /etc/bash.bashrc \
    && cd $PSRHOME \
    && git clone git://git.code.sf.net/p/dspsr/code dspsr \
    && cd dspsr \
    && git checkout master \
    && git checkout $GIT_DSPSR_TAG \
    && ./bootstrap \
    && echo 'apsr asp bcpm bpsr cpsr cpsr2 gmrt lbadr lbadr64 mark4 mark5 maxim mwa pdev pmdaq puma2 sigproc psrfits caspsr dummy fits spda1k spigot guppi lump' > $PSRHOME/dspsr/backends.list \
    && ./configure --prefix=$PSRHOME --enable-shared --with-x \
    && make \
    && make install

# Install PSRSALSA
RUN . /etc/bash.bashrc \
    && gsl_version=`gsl-config --version` \
    && gsl_version=`python -c "import numpy as np; print('%d0%d' % (np.floor($gsl_version), 10*($gsl_version % 1)))"` \
    && cd $PSRHOME \
    && git clone https://github.com/weltevrede/psrsalsa.git \
    && cd psrsalsa \
    && sed -i "s/GSLFLAGS *= *-DGSL_VERSION_NUMBER=.*/GSLFLAGS = -DGSL_VERSION_NUMBER=$gsl_version/" Makefile \
    && make \
    && echo 'export PATH="$PATH":$PSRHOME/psrsalsa/bin' >> /etc/bash.bashrc

# Install Nenupy
RUN cd $PSRHOME/src \
    && git clone https://github.com/AlanLoh/nenupy.git \
    && cd nenupy/ \
    && pip3 install -r requirements.txt \
    && pip3 install .

# Install PyGSM
RUN cd $PSRHOME/src \
    && git clone https://github.com/telegraphic/PyGSM \
    && cd PyGSM \
    && wget -O gsm2016_components.h5 https://zenodo.org/record/3479985/files/gsm2016_components.h5?download=1 \
    && wget -O gsm_components.h5 https://zenodo.org/record/3479985/files/gsm_components.h5?download=1 \
    && python3 setup.py install

# Install casacore data
RUN bash /config/get_casacore_data.sh

# Install AntPat
RUN pip3 install python-casacore \
    && cd $PSRHOME/src \
    && git clone https://github.com/2baOrNot2ba/AntPat.git \
    && cd AntPat/ \
    && python3 setup.py install

# Install DreamBeam
RUN cd $PSRHOME/src \
    && git clone https://github.com/2baOrNot2ba/dreamBeam.git \
    && cd dreamBeam \
    && python3 setup.py install

# Install PyGLOW
RUN cd $PSRHOME/src \
    && git clone https://github.com/timduly4/pyglow.git pyglow \
    && cd pyglow \
    && make -C src/pyglow/models source \
    && python3 setup.py install

# Install additional utilities
RUN apt-get install -y x11-apps gedit gimp evince nano vim emacs subversion ghostscript

CMD ["/bin/bash"]
