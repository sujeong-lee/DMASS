import requests
import os
​
def download_job_files(url, outdir):
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
​
      return r.json()
​
####
​
job_id = 'feae9705305d4430993687930f1cc3ad'
job_type = 'query'
username = 'sjlee'
#base_url = 'https://des.ncsa.illinois.edu/desaccess/api'
base_url = 'https://deslabs.ncsa.illinois.edu/files-desaccess/'
download_dir = './{}'.format(job_id)
job_url = '{}/{}/{}/{}'.format(base_url, username, job_type, job_id)
download_job_files(job_url, download_dir)
print('Files for job "{}" downloaded to "{}"'.format(job_id, download_dir))
