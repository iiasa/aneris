import os
import requests
import tarfile

print('FOO')
username = os.environ['ANERIS_CI_USER']
print('BAR', username)
password = os.environ['ANERIS_CI_PW']
print('BAZ')

url = 'https://data.ene.iiasa.ac.at/continuous_integration/aneris/'


def download(filename):
    r = requests.get(url + filename, auth=(username, password))

    if r.status_code == 200:
        print('Downloading {} from {}'.format(filename, url))
        with open(filename, 'wb') as out:
            for bits in r.iter_content():
                out.write(bits)
        print('Untarring {}'.format(filename))
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()
        os.remove(filename)
    else:
        raise IOError(
            'Failed download with user/pass: {}/{}'.format(username, password))


download('data.tar.gz')
download('output.tar.gz')
