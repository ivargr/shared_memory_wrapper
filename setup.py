from setuptools import setup
from distutils.core import setup

setup(name='shared_memory_wrapper',
      version='0.0.7',
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
      ]

)

"""
rm -rf dist
python3 setup.py sdist
twine upload --skip-existing dist/*

"""