from setuptools import setup

setup(name='NE',
      version='0.1',
      description='Evolutionary Search Worker',
      author='Joe Sarsfield',
      author_email='joe.sarsfield@gmail.com',
      py_modules=['evolution', 'hpc_initialisation', 'genome', 'substrate', 'network', 'config', 'genes', 'activations', 'evolution_parallel', 'species', 'environment'],
      zip_safe=True)