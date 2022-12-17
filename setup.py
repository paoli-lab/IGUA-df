import sys

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize
except ImportError as err:
    cythonize = err


class build_ext(_build_ext):
    """A `build_ext` that disables optimizations if compiled in debug mode.
    """

    # --- Build code ---

    def build_extension(self, ext):
        # add debug symbols if we are building in debug mode
        if self.debug:
            if self.compiler.compiler_type in {"unix", "cygwin", "mingw32"}:
                ext.extra_compile_args.append("-g")
            elif self.compiler.compiler_type == "msvc":
                ext.extra_compile_args.append("/Z7")
            if sys.implementation.name == "cpython":
                ext.define_macros.append(("CYTHON_TRACE_NOGIL", 1))
        else:
            ext.define_macros.append(("CYTHON_WITHOUT_ASSERTIONS", 1))
        # build the rest of the extension as normal
        _build_ext.build_extension(self, ext)

    def build_extensions(self):
        # check `cythonize` is available
        if isinstance(cythonize, ImportError):
            raise RuntimeError("Cython is required to run `build_ext` command") from cythonize

        # use debug directives with Cython if building in debug mode
        cython_args = {
            "include_path": ["include"],
            "compiler_directives": {
                "cdivision": True,
                "nonecheck": False,
            },
        }
        if self.force:
            cython_args["force"] = True
        if self.debug:
            cython_args["annotate"] = True
            cython_args["compiler_directives"]["cdivision_warnings"] = True
            cython_args["compiler_directives"]["warn.undeclared"] = True
            cython_args["compiler_directives"]["warn.unreachable"] = True
            cython_args["compiler_directives"]["warn.maybe_uninitialized"] = True
            cython_args["compiler_directives"]["warn.unused"] = True
            cython_args["compiler_directives"]["warn.unused_arg"] = True
            cython_args["compiler_directives"]["warn.unused_result"] = True
            cython_args["compiler_directives"]["warn.multiple_declarators"] = True
        else:
            cython_args["compiler_directives"]["boundscheck"] = False
            cython_args["compiler_directives"]["wraparound"] = False

        # cythonize the extensions and then build as normal
        self.extensions = cythonize(self.extensions, **cython_args)
        for extension in self.extensions:
            extension._needs_stub = False
        _build_ext.build_extensions(self)


setuptools.setup(
    ext_modules=[
        setuptools.Extension(
            "htgcf._manhattan",
            sources=["htgcf/_manhattan.pyx"],
        ),
    ],
    cmdclass={
        "build_ext": build_ext,
    }
)
