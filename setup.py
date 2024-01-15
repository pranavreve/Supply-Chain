from setuptools import find_packages, setup

setup(
    name="supply_chain_analytics",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "prophet>=1.0",
        "networkx>=2.6.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "pulp>=2.5.0",
        "plotly>=5.3.0",
        "dash>=2.0.0",
    ],
) 