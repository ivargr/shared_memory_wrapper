from setuptools import setup
from distutils.core import setup
from setuptools.command.build_ext import build_ext as _build_ext

# to make installation work without already having numpy
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(name='shared_memory_wrapper',
      version='0.0.22',
      description='Shared Memory Wrapper',
      url='http://github.com/ivargr/shared_memory_wrapper',
      author='Ivar Grytten',
      author_email='',
      license='MIT',
      packages=["shared_memory_wrapper"],
      zip_safe=False,
      setup_requires=["numpy"],
      cmdclass={'build_ext':build_ext},
      install_requires=['numpy', 'SharedArray'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ],
      #include_dirs=[numpy.get_include()]

)

"""
rm -rf dist
python3 setup.py sdist
twine upload --skip-existing dist/*

"""