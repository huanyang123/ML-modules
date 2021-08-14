import setuptools

with open('README.md', 'r') as file:
	long_description = file.read()


setuptools.setup(
	name = 'ml', #this must be unique
	include_package_data=True,
	version = '0.1',
	author = 'Huanwang (Henry) Yang',
	author_email = 'huanwang.yang@gmail.com',
	description = 'The ML untility modules',
	long_description = long_description,
	long_description_content_type = 'text/markdown' #,
	#packages=['modules', 'tests']
	#packages = setuptools.find_packages()
	
	'''
	long_description=read('README'),
    	classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    	]
	'''

  '''
	packages = setuptools.find_packages(),
	classifiers = [
	'Programming Language :: Python :: 3',
	'License :: OSI Aproved :: MIT License',
	"Operating System :: OS Independent"],
	python_requires = '>=3.5'
  '''
	)
