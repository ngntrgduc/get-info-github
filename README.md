## Get-Info-GitHub

### Why?
I often find myself being lost when visiting someone's repositories/stars. If that user has more than 100 repositories/stars, then it will be very exhausting (by default, 30 repositories will be display each navigation). So I made this to crawl all of it.

### Features
- Get all user's public repositories, stars, gists using GitHub GraphQL API
- Get crawled result to a folder if needed, easier to manage (default will store results in `data/<github_username>/`)
- Selectively crawl repositories, stars, or gists using flag arguments (`-r`/`--repo`, `-s`/`--star`, `-g`/`--gist`)

### How to use?
- Install requirements:
    ```python
    pip install requests python-dotenv fire
    ```
- Create GitHub Token, with `repo` scope
- Create a `.env` file, and put the token in:
    ```
    GITHUB_TOKEN = <your_token_here>    
    ```
- For basic crawling (crawl all repositories/stars/gists):
    ```python
    python main.py <github_username>
    ```
- If you want the crawled results in a folder, pass `-f` or `--folder`:
    ```python
    python main.py <github_username> -f
    ```
- Use flags to crawl specific data:
    - Crawl only repositories:
        ```python
        python main.py <github_username> -r
        ```
    - Crawl only stars:
        ```python
        python main.py <github_username> -s
        ```
    - Crawl only gists:
        ```python
        python main.py <github_username> -g
        ```
    - Combine flags to customize the crawl:
        ```python
        python main.py <github_username> -r -s
        ```