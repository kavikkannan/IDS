from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_req(file_path:str)->List[str]:
    '''
    function to returning list of req
    '''
    reqirements=[]
    with  open(file_path) as file_obj:
        reqirements=file_obj.readlines()
        reqirements=[req.replace("\n","") for req in reqirements]

        if HYPHEN_E_DOT in reqirements:
            reqirements.remove(HYPHEN_E_DOT)

setup(
name = 'ids_project',
version = '0.0.1',
author = 'kavikkannan k',
author_email = 'kavikkannan.k@gmal.com',
packages = find_packages(),
install_requires = get_req('requirements.txt')
)