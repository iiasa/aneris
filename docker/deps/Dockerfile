FROM ubuntu:16.04

# get add-apt-repository
RUN apt-get update 
RUN apt-get install -y --force-yes software-properties-common python-software-properties

RUN apt-get install -y --force-yes python-pip

RUN pip install \
    argparse \
    cython \
    numpy \
    pytest \
    pandas \
    xlrd \
    yaml

RUN apt-get install -y --force-yes python3-pip 

RUN pip3 install \
    argparse \
    cython \
    numpy \
    pytest \
    pandas \
    xlrd \
    yaml
    