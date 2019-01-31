from setuptools import setup, find_packages
curr_version = '0.1.0'

setup(
    name='shapedata',
    version=curr_version,
    packages=find_packages(),
    url='https://github.com/justussschock/shapedata',
    license='MIT',
    author='Justus Schock',
    author_email='justus.schock@rwth-aachen.de',
    description='',
    install_requires=["delira@git+https://github.com/justusschock/delira.git"],
    python_requires="> 3.5",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ]
)
