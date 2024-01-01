from setuptools import setup, find_packages

setup(
    name="supply-chain-analytics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "prophet>=1.0",
        "networkx>=2.6.0",
        "pulp>=2.5.0",
        "plotly>=5.3.0",
        "dash>=2.0.0",
        "boto3>=1.18.0",
        "streamlit>=1.12.0",
        "xgboost>=1.5.0",
        "great-expectations>=0.15.0",
        "pydantic>=1.9.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "flake8>=3.9.2",
            "black>=22.1.0",
            "jupyter>=1.0.0"
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive supply chain analytics solution",
    keywords="supply chain, analytics, data science, optimization",
    url="https://github.com/yourusername/supply-chain-analytics",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Information Technology",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
) 