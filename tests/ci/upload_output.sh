tar cvzf output.tar.gz gcam_harmonized.xlsx msg_harmonized.xlsx
scp output.tar.gz $1:/opt/data.ene.iiasa.ac.at/docs/continuous_integration/aneris
rm output.tar.gz
