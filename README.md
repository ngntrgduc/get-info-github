## Get-Info-GitHub

### Why?
I often find myself being lost when visiting someone's repositories/stars. If that user has more than 100 repositories/stars, then it will be very exhausting (by default, 30 repositories will be display each navigation). So I made this to crawl all of it.

### Features
- Get all repositories, starred, gists using GitHub GraphQL API
- Get crawled result to a folder if needed, easier to manage (default will store results in `data/<github_username>/`)
- Selectively crawl repositories, starred, or gists using flag arguments (`-r`/`--repo`, `-s`/`--star`, `-g`/`--gist`)

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
- For basic crawling (crawl all repositories/starred/gists):
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
    - Crawl only starred:
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
### Todo
- [x] More arguments setting: 
    - [x] --starred/-s for crawl only starred repo, also for repos (-r) and gists (-g)
    - [ ] ~~--sort for sorting crawled result based on number of stars~~ -> Redundant
    - [ ] ~~Maybe switch to Click CLI library for more flexibility? -> FP style~~ -> Keep it simple, less overhead
    - [ ] ~~Handle multiple usernames -> reuse session for all users~~ -> Redundant