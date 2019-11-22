import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='noun-phrase-ua',
    version='0.0.4',
    packages=setuptools.find_packages(),
    url='https://github.com/artemkramov/np-extractor-ua',
    author='Artem Kramov',
    author_email='artemkramov@gmail.com',
    description='Noun phrase extractor for the Ukrainian language',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
          'ufal.udpipe', 'dateparser'
    ],
    include_package_data=True
)
