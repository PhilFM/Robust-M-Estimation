import os
import numpy
from setuptools import setup, Extension


try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

extensions = [
    Extension("gnc_smoothie_philfm.cython_files.linear_regressor_welsch_fast", ["src/gnc_smoothie_philfm/cython_files/linear_regressor_welsch_fast.pyx"]),
    Extension("gnc_smoothie_philfm.cython_files.linear_regressor_pseudo_huber_fast", ["src/gnc_smoothie_philfm/cython_files/linear_regressor_pseudo_huber_fast.pyx"]),
    Extension("gnc_smoothie_philfm.cython_files.linear_regressor_gnc_irls_p_fast", ["src/gnc_smoothie_philfm/cython_files/linear_regressor_gnc_irls_p_fast.pyx"]),
    Extension("gnc_smoothie_philfm.cython_files.linear_regressor_weighted_fit", ["src/gnc_smoothie_philfm/cython_files/linear_regressor_weighted_fit.pyx"]),
]

CYTHONIZE = cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

setup(
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires='>=3.10',
)
