set -x
set -e

python download_data.py

pytest -v -s .
