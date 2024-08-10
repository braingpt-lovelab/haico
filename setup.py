from setuptools import setup, find_packages

setup(
    name='haico',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',  # Add your dependencies here
        'pandas',
        'torch',
        'pyro-ppl',
        # Add any other dependencies your project uses
    ],
    entry_points={
        'console_scripts': [],
    },
    python_requires='>=3.6',  # Specify the minimum Python version
    author='Felipe Yanez',
    author_email='felipe.yanez@mpinb.mpg.de',
    description='Human-AI complementarity',
    long_description=open('README.md').read(),  # Include your README file
    license='License Name',  # Choose an appropriate license
)
