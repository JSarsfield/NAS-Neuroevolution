from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext



compiler_args = ["-O3", "-DCYTHON_WITHOUT_ASSERTIONS"]


ext_modules = [
      Extension("example1", ["example1.py"], compiler_args),
      Extension("evaluate_evolutionary_search", ["evaluate_evolutionary_search.py"], compiler_args),
      Extension("evolution", ["evolution.py"], compiler_args),
      Extension("hpc_initialisation", ["hpc_initialisation.py"], compiler_args),
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
ext_modules.append(Extension("evolution_parallel", ["evolution_parallel.py"], compiler_args))

setup(
      name = 'NAS',
      version='0.1',
      description='Evolutionary Search Worker',
      author='Joe Sarsfield',
      author_email='joe.sarsfield@gmail.com',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      package_data={'NAS': ["*.so", "evolution_parallel.py"]}
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