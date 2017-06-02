FROM ubuntu:16.04

# get add-apt-repository
RUN apt-get update 
RUN apt-get install -y --force-yes wget 
RUN apt-get install -y --force-yes software-properties-common python-software-properties

RUN apt-get install -y --force-yes python-pip
RUN pip install argparse
RUN pip install cython
RUN pip install numpy
RUN pip install pandas
RUN pip install pytest
RUN pip install pyyaml
RUN pip install xlrd
RUN pip install xlsxwriter

RUN apt-get install -y --force-yes python3-pip 
RUN pip3 install argparse
RUN pip3 install cython
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install pytest
RUN pip3 install pyyaml
RUN pip3 install xlrd
RUN pip3 install xlsxwriter

