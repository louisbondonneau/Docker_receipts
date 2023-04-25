Singularity Containers
======================

Singularity + Docker receipts can be found on github https://github.com/louisbondonneau/Docker_receipts

Depending on the installation singularity executable can be named "singularity" or "apptainer".


Run a contaner
--------------
With psrchive, presto2.2 and dspsr bind under python2.7 + tempo2 + tempo1 + psrsalsa + python3(nenupy + AntPat + dreamBeam + psrqpy)
> singularity run -B $HOME:$HOME  -B /databf:/databf -B /data:/data -B /cep:/cep -B ~/.Xauthority:/home/root/.Xauthority /cep/lofar/pulsar/Singularity/pschive_py2.sif

With psrchive, presto4 and dspsr bind under python3.8 + tempo2 + tempo1 + psrsalsa + nenupy + AntPat + dreamBeam + psrqpy
> singularity run -B $HOME:$HOME  -B /databf:/databf -B /data:/data -B /cep:/cep -B ~/.Xauthority:/home/root/.Xauthority /cep/lofar/pulsar/Singularity/pschive_py3.sif


Build a container from nothing
------------------------------

### install Go and Singularity

> apt-get update
> 
> apt-get install -y build-essential libssl-dev uuid-dev libgpgme11-dev squashfs-tools libseccomp-dev wget pkg-config git cryptsetup libglib2.0-dev
> 
> GO_VERSION=1.20.2 OS=linux ARCH=amd64
> 
> wget -O /tmp/go${GO_VERSION}.${OS}-${ARCH}.tar.gz https://dl.google.com/go/go${GO_VERSION}.${OS}-${ARCH}.tar.gz
> 
> tar -C /usr/local -xzf /tmp/go${GO_VERSION}.${OS}-${ARCH}.tar.gz
> 
> echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
> 
> . ~/.bashrc
> 
> git clone --recurse-submodules https://github.com/sylabs/singularity.git singularity
> 
> cd singularity
> 
> ./mconfig
> 
> cd builddir
> 
> make
> 
> make install


### build python2 psrchive container

> git clone https://github.com/louisbondonneau/Docker_receipts
> 
> cd Docker_receipts/psrchive_py2
> 
> singularity build /cep/lofar/pulsar/Singularity/pschive_py2.sif Singularity


### build python3 psrchive container

> git clone https://github.com/louisbondonneau/Docker_receipts
> 
> cd Docker_receipts/psrchive_py3
> 
> singularity build /cep/lofar/pulsar/Singularity/pschive_py3.sif Singularity


### try an installation in the container

> singularity run --writable-tmpfs

known issues
------------
  1. relativ path to personnal home do not work ($HOME=/home/root)
  2. psrdata, hdf5... and other things in Vlad installed used by LOFAR are not installed at this time

TODO
----
ajouter:
  cuda driver
  jupyter

tempo1
> bash /cep/lofar/pulsar/ephem_scripts/par_conv_to_tempo1.sh /databf/nenufar-pulsar/ES03/ephem/B1919+21.par

tempo2
> cd ~/pulsar/tempo2/example_data
> tempo2 -gr plk -f example1.par example1.tim  -nofit
> psrchive_info  # Tempo2::Predictor support enabled

psrchive
> python -c 'import psrchive'
> python /cep/lofar/pulsar/NenPlot...

psrcat
> psrcat -E B1919+21

psredit
> psredit -c dm ....

presto
> python -c 'import presto'
> python /home/root/pulsar/presto/tests/test_presto_python.py

psrsalsa
> 

dreamBeam
> calibration of a NenuFAR archive

dspsr
> python -c 'import dspsr'
> by runnning it on a single-pulses file