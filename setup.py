from setuptools import setup, find_packages

setup(
    name="heart-disease-mlops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "joblib",
        "mlflow",
        "prometheus-fastapi-instrumentator",
    ],
)
