from setuptools import setup
from distutils.core import setup
import numpy as np

setup(name='shared_memory_wrapper',
      version='0.0.13',
      description='Shared Memory Wrapper',
      url='http://github.com/ivargr/shared_memory_wrapper',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["shared_memory_wrapper"],
      zip_safe=False,
      install_requires=['numpy', 'SharedArray'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ],
      include_dirs=[np.get_include()]

)

"""
rm -rf dist
python3 setup.py sdist
twine upload --skip-existing dist/*

"""