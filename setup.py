from setuptools import setup

NAME = 'zoomsweb'
VERSION = '0.1.0'

setup(
    name=NAME,
    version=VERSION,
    py_modules=['app', 'model'],
    install_requires=[
        'Flask>=2.0.0',
        'numpy<2.0.0',
        'onnxruntime>=1.18',
        'gunicorn',
    ],
    python_requires='>=3.7',
    author='Vladimir Starostin',
    author_email='v.starostin.m@gmail.com',
    description='ZooMS web application',
    url='https://github.com/mlcolab/zoomsweb',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'zoomsweb=app:main',
        ],
    },
)
