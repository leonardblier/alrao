""" Setup file alrao """
from setuptools import setup

main_version = '0'
subversion = '1'

version = main_version + '.' + subversion
setup(name='alrao',
      version=version,
      url='https://github.com/leonardblier/alrao',
      author='Leonard Blier, Pierre Wolinski, Yann Ollivier',
      author_email='leonardb@fb.com',
      license='CeCILL',
      packages=['alrao'],
      zip_safe=False)
