import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('GITHUB_TOKEN')
username = 'ngntrgduc'
url = f"https://api.github.com/users/{username}/repos"
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": "Bearer " + TOKEN,
    "X-GitHub-Api-Version": "2022-11-28",
}
params = {
    "sort": "pushed",
    "per_page": 100,
    "page": 1,
}

data = requests.get(url, headers=headers, params=params).json()

# Backup data
# with open('result.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, indent=2)
# data = json.load(open('result.json', 'r', encoding='utf-8'))

with open('README.md', 'w', encoding='utf-8') as f:
    f.write('### My projects\n')
    f.write('| **Repository** | **Description** |\n')
    f.write('| -------------- | --------------- |\n')

    for i in range(len(data)):
        repo_name   = data[i]['name']
        repo_url    = data[i]['html_url']
        description = data[i]['description']
        fork        = data[i]['fork']
        # stars       = data[i]['stargazers_count']
        # language    = data[i]['language']

        fork = '(*fork*)' if fork else ''
        if not description:
            description = ''

        f.write(f'| **[{repo_name}]({repo_url})** {fork} | {description} |\n')
