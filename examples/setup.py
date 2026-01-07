from setuptools import setup
import setuptools

from Cython.Build import cythonize
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    version="0.0.1",
    author="Philip McLauchlan",
    author_email="philipmclauchlan6@gmail.com",
    description="Robust M-estimation example algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    name='robust_m_estimation',
    ext_modules=cythonize("src/cython_files/*.pyx"),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires='>=3.10',
)
