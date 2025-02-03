import json
import requests
from time import perf_counter, sleep

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
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def reset_params(self) -> None:
        self.params["page"] = 1

    def set_fetch_all(self): 
        """Enable crawling all the results"""
        self.fetch_all = True

    def get_data(self) -> None:
        """Get data from GitHub API"""
        
        self.reset_params()

        if not self.fetch_all:
            self.data = self.session.get(self.url, params=self.params).json()
        else:
            self.data = []
            while True:
                data = self.session.get(self.url, params=self.params).json()
                if not data:
                    break
                self.params['page'] += 1
                self.data.extend(data)
                # sleep(0.2)

    def backup(self, file_name: str = 'result.json') -> None:
        """Write a backup to file"""

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

    def get_repositories(self, file_name: str = 'README.md') -> None:
        """Get repositories of user"""

        print('Getting repositories...')
        self.url = f'https://api.github.com/users/{self.username}/repos'
        self.get_data()
        print(f' -> Number of repositories: {len(self.data)}')

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('| **Repository** | **Description** |\n')
            f.write('| -------------- | --------------- |\n')

            for data in self.data:
                name        = data['name']
                url         = data['html_url']
                description = data['description']
                fork        = data['fork']
                # stars       = data['stargazers_count']
                # language    = data['language']

                fork = '(*fork*)' if fork else ''
                if not description:
                    description = ''

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

            for data in self.data:
                name        = data['full_name']
                url         = data['html_url']
                # language    = data['language']
                description = data['description']
                stars       = data['stargazers_count']
                stars       = format_stars(stars)

                if not description:
                    description = ''

                f.write(rf'| **[{name}]({url})** \| ⭐ *{stars}* | {description}')
                f.write('\n')

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

            for data in self.data:
                name        = list(data['files'])[0]
                url         = data['html_url']
                description = data['description']

                if not description:
                    description = ''

                f.write(f'| **[{name}]({url})** | {description} |\n')

    def __enter__(self):
        print(f'Crawling for user: {self.username}')
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.session.close()

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

    with User(name) as user:
        if all:
            user.set_fetch_all()

        user.get_repositories(f'{directory}README.md')
        user.get_starred(f'{directory}STARRED.md')
        user.get_gists(f'{directory}GISTS.md')

    print(f'Took {perf_counter() - tic:.2f}s to crawl')


if __name__ == '__main__':
    fire.Fire(main)
