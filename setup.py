import setuptools
from typing import List # type: ignore

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.1.0"

REPO_NAME = "nsa-ds-challenge"
AUTHOR_USER_NAME = "Ujjwal Solanki"
SRC_REPO = "ML-Project"
AUTHOR_EMAIL = "ujjwal.programmer@gmail.com"

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return requirements
    '''
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readline()
        requirements = [req.replace('\n','') for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="This is data science challenge given by NSA storage",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=setuptools.find_packages(),
    install_requires=get_requirements('requirements.txt')
)