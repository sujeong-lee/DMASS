import requests
import os

job_id = '9d1501718a80438681bd9ae66d8913e8'
job_type = 'query'
username = 'warner785'
base_url = 'https://des.ncsa.illinois.edu/files-desaccess/'

def download_job_files(url, outdir):
    if os.path.exists(outdir) == False:
        os.makedirs(outdir, exist_ok=True)
    r = requests.get('{}/json'.format(url))
    for item in r.json():
        if item['type'] == 'directory':
            suburl = '{}/{}'.format(url, item['name'])
            subdir = '{}/{}'.format(outdir, item['name'])
            download_job_files(suburl, subdir)
        elif item['type'] == 'file':
            data = requests.get('{}/{}'.format(url, item['name']))
            with open('{}/{}'.format(outdir, item['name']), "wb") as file:
                  file.write(data.content)

    return r.json()

job_url = '{}/{}/{}/{}'.format(base_url, username, job_type, job_id)
download_dir = './{}'.format(job_id)
download_job_files(job_url, download_dir)
print('Files for job "{}" downloaded to "{}"'.format(job_id, download_dir))
