from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext



compiler_args = ["-O3", "-DCYTHON_WITHOUT_ASSERTIONS"]


ext_modules = [
      Extension("substrate", ["substrate.py"], compiler_args),
      Extension("genome", ["genome.py"], compiler_args),
      Extension("substrate", ["substrate.py"], compiler_args),
      Extension("network", ["network.py"], compiler_args),
      Extension("config", ["config.py"], compiler_args),
      Extension("genes", ["genes.py"], compiler_args),
      Extension("activations", ["activations.py"], compiler_args),
      Extension("species", ["species.py"], compiler_args),
      Extension("environment", ["environment.py"], compiler_args)
    ]

ext_modules = cythonize(ext_modules, compiler_directives={"language_level": "3"})
py_modules = ['example1', 'evaluate_evolutionary_search', 'hpc_initialisation', 'evolution',  'evolution_parallel']

setup(
      name = 'NAS',
      version='0.1',
      description='Evolutionary Search Worker',
      author='Joe Sarsfield',
      author_email='joe.sarsfield@gmail.com',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      py_modules = py_modules,
      package_data={'NAS': ["build/lib.linux-x86_64-3.7/*.so", *py_modules]}
)

"""
from setuptools import setup

setup(name='NAS',
      version='0.1',
      description='Evolutionary Search Worker',
      author='Joe Sarsfield',
      author_email='joe.sarsfield@gmail.com',
      py_modules=['genome', 'substrate', 'network', 'config', 'genes', 'activations', 'evolution_parallel', 'species', 'environment'],
      zip_safe=True)
"""