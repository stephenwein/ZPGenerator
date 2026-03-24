from setuptools import find_packages, setup

setup(
    name='zpgenerator',
    author="quandela",
    version='0.3.0',
    packages=find_packages(include=["zpgenerator", "zpgenerator.*"]),
    python_requires=">=3.10,<3.13",
    install_requires=[
        'qutip>=5.2.3,<6',
        'numpy',
        'scipy',
        'frozendict',
        'matplotlib'
    ],
    extras_require={
        'interactive': ['jupyter'],
    }
)
