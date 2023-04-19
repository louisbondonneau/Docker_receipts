Singularity Containers
======================

Singularity + Docker receipts can be found on github https://github.com/louisbondonneau/Docker_receipts

Depending on the installation singularity executable can be named "singularity" or "apptainer".


Run a contaner
--------------
With psrchive and presto2.2 working under python2.7 + tempo2 + tempo1 + dspsr
> singularity shell -B $HOME:$HOME  -B /databf:/databf -B /data:/data -B /cep:/cep /cep/lofar/pulsar/Singularity/pschive_py2.sif

With psrchive and presto working under python3.8 + tempo2 + tempo1 + dspsr + nenupy + AntPat + dreamBeam
> singularity shell -B $HOME:$HOME  -B /databf:/databf -B /data:/data -B /cep:/cep /cep/lofar/pulsar/Singularity/pschive_py3.sif

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


TODO
----

test script using
tempo1
> by generating a polyco file
tempo2
> tempo2 -gr transform ....
psrchive
> python /cep/lofar/pulsar/NenPlot...
psredit
> psredit -c dm ....
presto
> python $PRESTO/tests/???.py
dreamBeam
> calibration of a NenuFAR archive