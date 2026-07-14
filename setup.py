"""Build wheels containing the native CPU backend and runtime CUDA kernels."""

from __future__ import annotations

import sys
from pathlib import Path
from shutil import copy2

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class BuildPythonWithCudaKernels(build_py):
    """Include CUDA source compiled by NVRTC at runtime."""

    def run(self) -> None:
        super().run()
        source = ROOT / "cuda" / "kernels.cu"
        destination = Path(self.build_lib) / "mimillm" / "_native" / "mimillm_cuda_kernels.cu"
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)


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
        cmdclass={"bdist_wheel": WindowsBinaryWheel, "build_py": BuildPythonWithCudaKernels},
    )
else:
    setup(
        exclude_package_data={"mimillm": ["_native/*.dll"]},
        cmdclass={"build_py": BuildPythonWithCudaKernels},
    )
