from setuptools import setup

setup(name='NE-worker',
      version='0.1',
      description='Evolutionary Search Worker',
      author='Joe Sarsfield',
      author_email='joe.sarsfield@gmail.com',
      py_modules=['genome', 'substrate', 'network', 'config', 'genes', 'activations', 'evolution_parallel', 'species', 'environment'],
      zip_safe=True)
