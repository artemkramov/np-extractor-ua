import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='noun-phrase-ua',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/artemkramov/np-extractor-ua',
    author='Artem Kramov',
    author_email='artemkramov@gmail.com',
    description='Noun phrase extractor for the Ukrainian language',
    long_description=long_description,
    install_requires=[
          'ufal.udpipe', 'dateparser'
    ],
    include_package_data=True
)
