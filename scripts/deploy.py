import os

from git import Repo

current_branch = Repo('.').head.reference.name

if (current_branch == "main"):
    print('Current branch is main, please switch to prod, merge and push.')
else:
    os.system('git push heroku prod:main -f')
