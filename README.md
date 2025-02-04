## Get-Info-GitHub

### Why?
I often find myself being lost when visiting someone's repositories/stars. If that user has more than 100 repositories/stars, then it will be very exhausted. So I made this to crawl all of it.

### Features
- Get all repositories, starred, gists using GitHub GraphQL API
- Get crawled result to a folder if needed, easier to manage (default will store results in `data/<github_username>/`)

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

### Todo
- [ ] More arguments setting: 
    - [ ] --starred/-s for crawl only starred repo, also for repos (-r) and gists (-g)
    - [ ] ~~--sort for sorting crawled result based on number of stars~~ -> Redundant
    - [ ] Maybe switch to Click CLI library for more flexibility? -> FP style
    - [ ] Handle multiple usernames -> reuse session for all users