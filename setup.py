from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dnnviewer',
    version='0.1.0.dev12',
    author='A. Hue',
    url='https://github.com/tonio73/dnnviewer',
    license='MIT',
    description='Deep Neural Network inspection: view weights, gradients and activations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['DNN', 'CNN', 'Interpretability', 'Tensorflow', 'Keras'],
    packages=find_packages(),
    package_data={'dnnviewer': ["assets/*.css"]},
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=[
        'numpy',
        'tensorflow>=2.0',
        'tensorflow-datasets',
        'plotly>=4.0',
        'dash',
        'dash-core-components',
        'dash-html-components',
        'dash-bootstrap-components',
        'pillow'
    ],
    python_requires='>=3.5',
    entry_points={
        'console_scripts': [
            'dnnviewer = dnnviewer.main:main'
        ]
    },
)
