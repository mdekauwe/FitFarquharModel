from setuptools import setup
setup(
    name='FitFarquharModel',
    version='1.0',
    author='Martin De Kauwe',
    author_email='mdekauwe@gmail.com',
    platforms = ['any'],
    description='Fit farquhar model parameters to A-Ci observations.',
    package_dir = {'ffm': 'src'},
    packages = ['ffm']
)