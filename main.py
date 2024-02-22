import json
import requests
from dotenv import dotenv_values

class User:
    def __init__(self, username: str) -> None:
        self.username = username
        self.url = f'https://api.github.com/users/{self.username}/repos'
        TOKEN = dotenv_values('.env')['GITHUB_TOKEN']
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + TOKEN,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.params = {
            "sort": "pushed",
            "per_page": 100,
            "page": 1,
        }
    
    def get_data(self, all: bool = False) -> None:
        """Get data from GitHub API"""
        
        if not all:
            self.data = requests.get(self.url, headers=self.headers, 
                                     params=self.params).json()
        else:
            self.data = []
            while True:
                data = requests.get(self.url, headers=self.headers, 
                                    params=self.params).json()
                if not data:
                    break
                self.params['page'] += 1
                self.data += data
    
    def backup(self, file_name: str = 'result.json') -> None:
        """Write a backup to file"""

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)
    
    def get_ignore(self, file_name: str = 'ignore.txt'):
        """Get ignore repo from file"""

        with open(file_name, 'r', encoding='utf-8') as f:
            ignore_repo = f.read().splitlines()

        return ignore_repo
    
    def get_repositories(self, file_name: str = 'README.md', 
                         ignore: bool = True, all: bool = False) -> None:
        """Get repositories of user"""

        self.get_data(all)
        
        if ignore:
            ignore_repo = self.get_ignore()

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('| **Repository** | **Description** |\n')
            f.write('| -------------- | --------------- |\n')

            for i in range(len(self.data)):
                name        = self.data[i]['name']
                url         = self.data[i]['html_url']
                description = self.data[i]['description']
                fork        = self.data[i]['fork']
                # stars       = self.data[i]['stargazers_count']
                # language    = self.data[i]['language']

                fork = '(*fork*)' if fork else ''
                if not description:
                    description = ''
                
                if name in ignore_repo:
                    continue

                f.write(f'| **[{name}]({url})** {fork} | {description} |\n')

    def get_starred(self, file_name: str = 'STARRED.md', 
                    all: bool = False) -> None:
        """Get starred repositories of user"""

        self.url = f'https://api.github.com/users/{self.username}/starred'
        self.get_data(all)

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('### My starred repositories\n')
            f.write('| **Repository** | **Description** |\n')
            f.write('| -------------- | --------------- |\n')

            for i in range(len(self.data)):
                name        = self.data[i]['full_name']
                url         = self.data[i]['html_url']
                # language    = self.data[i]['language']
                description = self.data[i]['description']
                stars       = self.data[i]['stargazers_count']

                if not description:
                    description = ''

                f.write(f'| **[{name}]({url})** \| ⭐ *{stars}* | {description}\n')
    
    def get_gists(self, file_name: str = 'GISTS.md', 
                  all: bool = False) -> None:
        """Get all gists of user"""

        self.url = f'https://api.github.com/gists'
        self.get_data(all)

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('### My gists\n')
            f.write('| **Gist name** | **Description** |\n')
            f.write('| ------------- | --------------- |\n')

            for i in range(len(self.data)):
                name        = list(self.data[i]['files'])[0]
                url         = self.data[i]['html_url']
                description = self.data[i]['description']

                if not description:
                    description = ''

                f.write(f'| **[{name}]({url})** | {description} |\n')

def create_folder(name):
    from pathlib import Path
    Path(name).mkdir()

def main(name, folder=''):
    if folder:
        assert folder[-1] == '/', 'Folder name must include / at the end'
        create_folder(folder)

    user = User(name)
    user.get_repositories(f'{folder}README.md')
    user.get_starred(f'{folder}STARRED.md')
    user.get_gists(f'{folder}GISTS.md')

if __name__ == '__main__':
    from time import perf_counter
    name = 'ngntrgduc'
    tic = perf_counter()
    main(name)
    # main(name, f'{name}')
    print(f'Took {perf_counter() - tic:.2f}s to crawl for user {name}')