"""Cloud Platform Integration.

This module provides integration with major cloud platforms:
- AWS (Amazon Web Services)
- GCP (Google Cloud Platform)
- Azure (Microsoft Azure)

Features:
- Compute instance management
- Storage integration (S3, GCS, Blob Storage)
- Kubernetes cluster management (EKS, GKE, AKS)
- Serverless deployment (Lambda, Cloud Functions, Azure Functions)

Author: Nonequilibrium Physics Agents
"""

from .aws import (
    AWSConfig,
    AWSCompute,
    AWSS3Storage,
    AWSEKSCluster,
)

from .gcp import (
    GCPConfig,
    GCPCompute,
    GCSStorage,
    GKECluster,
)

from .azure import (
    AzureConfig,
    AzureCompute,
    AzureBlobStorage,
    AKSCluster,
)

__all__ = [
    # AWS
    'AWSConfig',
    'AWSCompute',
    'AWSS3Storage',
    'AWSEKSCluster',
    # GCP
    'GCPConfig',
    'GCPCompute',
    'GCSStorage',
    'GKECluster',
    # Azure
    'AzureConfig',
    'AzureCompute',
    'AzureBlobStorage',
    'AKSCluster',
]
