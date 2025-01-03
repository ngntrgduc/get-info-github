import json
import requests
from time import perf_counter

from pathlib import Path
from dotenv import dotenv_values
import fire


class User:
    def __init__(self, username: str) -> None:
        self.username = username
        self.url = ''
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
        self.fetch_all = False
    
    def reset_params(self) -> None:
        self.params["page"] = 1

    def set_fetch_all(self): 
        """Enable crawling all the results"""
        self.fetch_all = True

    def get_data(self) -> None:
        """Get data from GitHub API"""
        
        self.reset_params()

        if not self.fetch_all:
            self.data = requests.get(self.url, headers=self.headers, params=self.params).json()
        else:
            self.data = []
            while True:
                data = requests.get(self.url, headers=self.headers, params=self.params).json()
                if not data:
                    break
                self.params['page'] += 1
                self.data += data

    def backup(self, file_name: str = 'result.json') -> None:
        """Write a backup to file"""

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

    def get_ignore(self, file_name: str = 'ignore.txt') -> list[str]:
        """Get ignore repo from file"""
        if not Path(file_name).exists():
            return []

        with open(file_name, 'r', encoding='utf-8') as f:
            ignore_repo = f.read().splitlines()

        return ignore_repo

    def get_repositories(self, file_name: str = 'README.md') -> None:
        """Get repositories of user"""

        print('Getting repositories...')
        self.url = f'https://api.github.com/users/{self.username}/repos'
        self.get_data()
        print(f' -> Number of repositories: {len(self.data)}')

        ignore_repos = self.get_ignore()

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

                if name in ignore_repos:
                    continue

                f.write(f'| **[{name}]({url})** {fork} | {description} |\n')

    def get_starred(self, file_name: str = 'STARRED.md') -> None:
        """Get starred repositories of user"""

        print('Getting starred...')
        self.url = f'https://api.github.com/users/{self.username}/starred'
        self.get_data()
        print(f' -> Number of starred: {len(self.data)}')

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
                stars       = format_stars(stars)

                if not description:
                    description = ''

                f.write(f'| **[{name}]({url})** \| ⭐ *{stars}* | {description}\n')

    def get_gists(self, file_name: str = 'GISTS.md') -> None:
        """Get all gists of user"""

        print('Getting gists...')
        self.url = f'https://api.github.com/users/{self.username}/gists'
        self.get_data()
        print(f' -> Number of gists: {len(self.data)}')

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('### My gists\n')
            f.write('| **Gist** | **Description** |\n')
            f.write('| ------------- | --------------- |\n')

            for i in range(len(self.data)):
                name        = list(self.data[i]['files'])[0]
                url         = self.data[i]['html_url']
                description = self.data[i]['description']

                if not description:
                    description = ''

                f.write(f'| **[{name}]({url})** | {description} |\n')

def create_folder(name: str) -> None:
    """Create folder if not existed"""
    folder = Path(name)
    if not folder.exists():
        folder.mkdir()

def format_stars(number: int) -> int | str:
    """Format number of stars"""
    if number > 1000:
        return f'{number/1000:.1f}K'
    return number

def main(
        name: str,  
        all: bool = False, 
        folder: bool = False
    ) -> None:

    directory = ''
    if folder:
        # Ensure the data folder exists
        if not Path('data/').exists():
            create_folder(f'data/')

        directory = f'data/{name}/'
        create_folder(directory)

    tic = perf_counter()

    user = User(name)
    if all:
        user.set_fetch_all()

    user.get_repositories(f'{directory}README.md')
    user.get_starred(f'{directory}STARRED.md')
    user.get_gists(f'{directory}GISTS.md')

    print(f'Took {perf_counter() - tic:.2f}s to crawl data for user: {name}')


if __name__ == '__main__':
    fire.Fire(main)
