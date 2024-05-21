import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='backtesting_mc',  # this should be unique
    version='0.0.1',
    author='mc',
    description='backtesing package',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3 ',
        'License :: OSI Approved :: MIT License '],
    python_requires='>=3.5'
)
