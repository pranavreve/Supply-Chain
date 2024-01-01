import argparse
import os
import sys
import logging
import boto3
import json
from aws_infrastructure import AWSDeployment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Deploy Supply Chain Analytics solution')
    parser.add_argument('--env', choices=['dev', 'test', 'prod'], default='dev',
                       help='Deployment environment')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region')
    parser.add_argument('--profile', default=None,
                       help='AWS profile name')
    return parser.parse_args()

def load_config(env):
    """Load environment-specific configuration"""
    config_path = os.path.join(os.path.dirname(__file__), f'config/{env}.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

def main():
    """Main deployment function"""
    args = parse_arguments()
    config = load_config(args.env)
    
    # Create AWS deployment handler
    aws = AWSDeployment(region_name=args.region, profile_name=args.profile)
    
    logger.info(f"Starting deployment to {args.env} environment in {args.region}")
    
    try:
        # Deploy CloudFormation stack
        stack_name = f"supply-chain-analytics-{args.env}"
        template_file = os.path.join(os.path.dirname(__file__), 'cloudformation/main.yaml')
        
        logger.info(f"Deploying CloudFormation stack: {stack_name}")
        outputs = aws.deploy_with_cloudformation(stack_name, template_file)
        
        # Output important resources
        logger.info("Deployment completed successfully")
        logger.info("Resource URLs:")
        if 'DashboardURL' in outputs:
            logger.info(f"Dashboard: {outputs['DashboardURL']}")
        if 'ApiEndpoint' in outputs:
            logger.info(f"API Endpoint: {outputs['ApiEndpoint']}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 