"""Build configuration for the Windows wheel that bundles the native backend."""

from __future__ import annotations

import sys

from setuptools import Distribution, setup


if sys.platform == "win32":
    from wheel.bdist_wheel import bdist_wheel

    class WindowsBinaryDistribution(Distribution):
        def has_ext_modules(self) -> bool:
            return True

    class WindowsBinaryWheel(bdist_wheel):
        def get_tag(self) -> tuple[str, str, str]:
            _python, _abi, platform = super().get_tag()
            return "py3", "none", platform

    setup(
        distclass=WindowsBinaryDistribution,
        cmdclass={"bdist_wheel": WindowsBinaryWheel},
    )
else:
    setup(exclude_package_data={"mimillm": ["_native/*.dll"]})
