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
        self.headers = { "Authorization": "Bearer " + TOKEN }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.repos = []
        self.gists = []
        self.starred = []

    def __enter__(self):
        print(f'Crawling for user: {self.username}')
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.session.close()

    def get_data(self) -> None:
        """Get data of user using GitHub GraphQL API"""

        def format_cursor(cursor: str | None) -> str:
            """Format cursor for GraphQL query"""
            return "null" if cursor is None else f'"{cursor}"'

        def fetch_pagination(data: dict, key: str, storage: list) -> tuple:
            """Handle pagination"""
            section = data.get(key, {})
            storage.extend(section.get('nodes', []))
            page_info = section.get('pageInfo', {})
            return page_info.get('endCursor'), page_info.get('hasNextPage', False)

        def generate_query(
                repos_cursor, stars_cursor, gists_cursor, 
                repos_active, stars_active, gists_active
            ) -> str:
            """Dynamically generates the GraphQL query based on active pagination states"""
            first_limit = 100
            repos_query = f"""
                repositories(
                    first: {first_limit},
                    orderBy: {{
                        field: PUSHED_AT,
                        direction: DESC
                    }},
                    privacy: PUBLIC,
                    ownerAffiliations: OWNER,
                    after: {format_cursor(repos_cursor)},
                ) {{
                    nodes {{
                        name
                        description
                        url
                        isFork
                    }}
                    pageInfo {{
                        endCursor
                        hasNextPage
                    }}
                }}""" if repos_active else ''

            starred_query = f"""
                starredRepositories(
                    first: {first_limit},
                    orderBy: {{
                        field: STARRED_AT,
                        direction: DESC
                    }},
                    after: {format_cursor(stars_cursor)},
                ) {{
                    nodes {{
                        nameWithOwner
                        description
                        url
                        stargazerCount 
                    }}
                    pageInfo {{
                        endCursor
                        hasNextPage
                    }}
                }}""" if stars_active else ''

            gists_query = f"""
                gists(
                    first: {first_limit},
                    orderBy: {{
                        field: UPDATED_AT,
                        direction: DESC
                    }},
                    after: {format_cursor(gists_cursor)},
                ) {{
                    nodes {{
                        files (limit: 1) {{
                            name
                        }}
                        description
                        url
                    }} 
                    pageInfo {{
                        endCursor
                        hasNextPage
                    }}
                }}""" if gists_active else ''
        
            return f"""{{ 
                user(login: "{self.username}") {{ 
                    {repos_query}
                    {starred_query}
                    {gists_query}
                }}
            }}"""

        repos_cursor = gists_cursor = stars_cursor = None
        repos_active = stars_active = gists_active = True

        print('Getting data...')

        while repos_active or stars_active or gists_active:
            graphql_query = generate_query(
                repos_cursor, stars_cursor, gists_cursor, 
                repos_active, stars_active, gists_active
            )
            response = self.session.post(
                'https://api.github.com/graphql', 
                json={'query': graphql_query},
                headers=self.headers
            )
            data = response.json()['data']['user']

            if repos_active:
                repos_cursor, repos_active = fetch_pagination(data, 'repositories', self.repos)

            if stars_active:
                stars_cursor, stars_active = fetch_pagination(data, 'starredRepositories', self.starred)

            if gists_active:
                gists_cursor, gists_active = fetch_pagination(data, 'gists', self.gists)
        
        print(f' - Number of repositories: {len(self.repos)}')
        print(f' - Number of starred: {len(self.starred)}')
        print(f' - Number of gists: {len(self.gists)}')
        
    def write_repositories(self, file_name: str = 'README.md') -> None:
        """Write repositories to file"""
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"### {self.username}'s repositories\n")
            f.write('| **Repository** | **Description** |\n')
            f.write('| -------------- | --------------- |\n')

            for repo in self.repos:
                name = repo['name']
                url = repo['url']
                fork = '(*fork*)' if repo['isFork'] else ''
                description = repo['description'] or ''

                f.write(f'| **[{name}]({url})** {fork} | {description} |\n')

    def write_starred(self, file_name: str = 'STARRED.md') -> None:
        """Write starred repositories to file"""

        def format_stars(number: int) -> int | str:
            """Format number of stars"""
            return f'{number/1000:.1f}K' if number > 1000 else number

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"### {self.username}'s starred repositories\n")
            f.write('| **Repository** | **Description** |\n')
            f.write('| -------------- | --------------- |\n')

            for repo in self.starred:
                name = repo['nameWithOwner']
                url = repo['url']
                stars = format_stars(repo['stargazerCount'])
                description = repo['description'] or ''

                f.write(rf'| **[{name}]({url})** \| ⭐ *{stars}* | {description}')
                f.write('\n')

    def write_gists(self, file_name: str = 'GISTS.md') -> None:
        """Write gists to file"""

        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(f"### {self.username}'s gists\n")
            f.write('|    **Gist*    | **Description** |\n')
            f.write('| ------------- | --------------- |\n')

            for gist in self.gists:
                name = gist['files'][0]['name']
                url = gist['url']
                description = gist['description'] or ''

                f.write(f'| **[{name}]({url})** | {description} |\n')


def create_folder(name: str) -> None:
    """Create folder if not existed"""
    folder = Path(name)
    if not folder.exists():
        folder.mkdir()

def main(
        name: str,
        folder: bool = False,
    ) -> None:

    directory = ''
    if folder:
        create_folder(f'data/')  # Ensure the data folder exists
        directory = f'data/{name}/'
        create_folder(directory)

    tic = perf_counter()

    with User(name) as user:
        user.get_data()
        user.write_repositories(f'{directory}README.md')
        user.write_starred(f'{directory}STARRED.md')
        user.write_gists(f'{directory}GISTS.md')

    print(f'Took {perf_counter() - tic:.2f}s to crawl')


if __name__ == '__main__':
    fire.Fire(main)
