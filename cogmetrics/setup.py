"""
Setup script for CogMetrics consciousness measurement library
"""
from setuptools import setup, find_packages

setup(
    name="cogmetrics",
    version="0.1.0",
    description="Empirical measurement tools for consciousness emergence in AI systems",
    author="Consciousness Research Lab",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "prometheus-client>=0.12.0",
        "mlflow>=1.20.0",
        "wandb>=0.12.0",
    ],
    extras_require={
        "full": [
            "pyphi>=1.3.0",
            "jpype1>=1.3.0",
            "tensorflow>=2.6.0",
            "torch>=1.10.0",
            "transformers>=4.12.0",
        ],
        "jidt": ["jpype1>=1.3.0"],
        "iit": ["pyphi>=1.3.0"],
        "viz": ["plotly>=5.0.0", "dash>=2.0.0"],
    },
    python_requires=">=3.8",
)