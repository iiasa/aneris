FROM gidden/aneris-base

COPY . /aneris
ADD http://data.ene.iiasa.ac.at/gidden/aneris_regression_data.tar.gz /aneris/tests
RUN tar xvf /aneris/tests/aneris_regression_data.tar.gz -C /aneris/tests/ 
WORKDIR /aneris
