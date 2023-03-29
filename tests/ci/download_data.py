import os
import tarfile

import requests


username = os.environ["ANERIS_CI_USER"]
password = os.environ["ANERIS_CI_PW"]

url = "https://data.ene.iiasa.ac.at/continuous_integration/aneris/"


def download(filename):
    r = requests.get(url + filename, auth=(username, password))

    if r.status_code == 200:
        print(f"Downloading {filename} from {url}")
        with open(filename, "wb") as out:
            for bits in r.iter_content():
                out.write(bits)
        assert os.path.exists(filename)
        print(f"Untarring {filename}")
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()
        os.remove(filename)
    else:
        raise OSError(f"Failed download with user/pass: {username}/{password}")


download("data.tar.gz")
download("output.tar.gz")
