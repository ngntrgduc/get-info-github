Here are all repositories/stars/gists of my GitHub account, crawled using GitHub API. 


### Why?
I often find myself being lost when visiting someone's repositories/stars. If that user has more than 100 repositories/stars, then it will be very exhausted. So I've made this to crawl all of it.

I first made this to view other accounts, but then I realized that I could use this for myself.

### Features
- Get all repositories, starred, gists
- Crawled result to a directory if provide
- Ignore specific repositories in `ignore.txt`

### How to use?
- Install requirements:
    ```python
    pip install requests python-dotenv
    ```
- Create GitHub Token, with `repo` scope.
- Create a `.env` file, and put the token in:
    ```
    GITHUB_TOKEN = <your_token_here>    
    ```
- Run `main.py`.

If you want to ignore specific repositories: Create a `ignore.txt` file, and write the name of the repositories you want to ignore, separated by line.

For more information, visit: https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28#list-repositories-for-a-user
