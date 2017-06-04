FROM gidden/aneris-base

COPY . /aneris

RUN wget -q -c --no-check-certificate http://data.ene.iiasa.ac.at/gidden/aneris_regression_data.tar.gz 
RUN tar xvf aneris_regression_data.tar.gz -C /aneris/tests/ 

WORKDIR /aneris
RUN python2 setup.py install
RUN python3 setup.py install

# this should probably be removed, but I'm lazy and tired
ENV COVERALLS_REPO_TOKEN uwv4564ZW7b6XGOEX4luJ9yummfqeUle1
