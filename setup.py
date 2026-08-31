from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def yaml_files(directory: str) -> list[str]:
    return [str(path) for path in sorted((ROOT / directory).glob("*.yaml"))]


setup(
    name="lnl-toolbox",
    version="0.1.0",
    description="A reproducible research toolbox for learning with noisy labels",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy>=1.24", "pyyaml>=6.0"],
    extras_require={
        "train": [
            "torch>=2.2",
            "torchvision>=0.17",
            "randaugment==1.0.2",
            "pyyaml>=6.0",
            "scikit-learn>=1.7,<2",
        ],
        "dev": [
            "build>=1.2",
            "coverage[toml]>=7.6",
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "ruff>=0.9",
            "tomli; python_version < '3.11'",
        ],
    },
    entry_points={
        "console_scripts": [
            "lnl=lnl_toolbox.cli.main:main",
            "lnl-clean-train=lnl_toolbox.cli.clean_train:main",
            "lnl-inspect-data=lnl_toolbox.cli.inspect_data:main",
            "lnl-make-noise=lnl_toolbox.cli.make_noise:main",
            "lnl-train=lnl_toolbox.cli.train:main",
        ]
    },
    package_data={
        "lnl_toolbox": ["paper_catalog.json"],
        "lnl_toolbox.cli": ["data/recipe_catalog.json"],
        "lnl_toolbox.scratch": ["recipes/**/*.yaml", "web/*.html", "web/*.js", "web/*.css"],
    },
    data_files=[
        ("share/lnl-toolbox/configs/experiment", yaml_files("configs/experiment")),
        ("share/lnl-toolbox/configs/reproduction", yaml_files("configs/reproduction")),
    ],
)
