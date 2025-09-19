from setuptools import setup, find_packages

setup(
    name="ai-scientist-team",
    version="0.1.0",
    description="Multi-agent research system using Amazon Bedrock",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "boto3>=1.34.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.1.4",
        "numpy>=1.24.3",
        "requests>=2.31.0",
        "httpx>=0.25.2",
        "structlog>=23.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ]
    },
)