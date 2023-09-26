Here's my crawled repositories list, using GitHub API. 

I find myself being lost when visiting someone's repositories. If that user has more than 100 repositories, then it will be very exhausted...

And here be dragons 🐉.

**PS**: I made this to view repositories of other accounts. But then I realized this can be used for personal.

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

For more information, visit: https://docs.github.com/en/rest/repos/repos?apiVersion=2022-11-28#list-repositories-for-a-user

### TODO
- [x] `Ignore` repo lists
- [ ] (*WIP*) Starred repo crawler