tar cvzf data.tar.gz history.csv rc.yaml test-gcam test-msg
scp data.tar.gz $1:/opt/data.ene.iiasa.ac.at/docs/continuous_integration/aneris
rm data.tar.gz
