from setuptools import find_packages, setup

setup(
    name='zpgenerator',
    author="quandela",
    version='0.2.0',
    packages=find_packages(include=["zpgenerator", "zpgenerator.*"]),
    python_requires=">=3.9,<3.13",
    install_requires=[
        'qutip',
        'numpy',
        'scipy',
        'frozendict',
        'matplotlib'
    ],
    extras_require={
        'interactive': ['jupyter'],
    }
)
