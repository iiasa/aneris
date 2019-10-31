set -x
set -e

python download_data.py

tar xvf data.tar.gz
tar xvf output.tar.gz

pytest -v -s .
