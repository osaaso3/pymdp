import setuptools

setuptools.setup(
    name="pymdp",
    version="0.1.0",
    description=("An Python-based implementation of active inference for Markov Decision Processes"),
    license="Apache 2.0",
    url="https://github.com/infer-actively/pymdp",
    packages=[
        "pymdp", "pymdp.envs"
    ],
)
