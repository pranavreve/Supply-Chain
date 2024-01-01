#!/usr/bin/env python
"""
AWS CDK infrastructure for Supply Chain Analytics platform deployment.
This module defines AWS resources required for the application.
"""

import os
from constructs import Construct
from aws_cdk import (
    App, Stack, Duration, RemovalPolicy,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecr as ecr,
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_dynamodb as dynamodb,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cloudwatch_actions,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subscriptions,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_ecr_assets as ecr_assets,
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as events_targets,
    aws_rds as rds,
    aws_secretsmanager as secretsmanager,
    CfnOutput
)


class SupplyChainAnalyticsStack(Stack):
    """Supply Chain Analytics Infrastructure Stack."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """Initialize the Supply Chain Analytics CDK Stack."""
        super().__init__(scope, construct_id, **kwargs)

        # Environment variables
        app_name = os.environ.get("APP_NAME", "supply-chain-analytics")
        environment = os.environ.get("ENVIRONMENT", "dev")
        admin_email = os.environ.get("ADMIN_EMAIL", "admin@example.com")
        vpc_cidr = os.environ.get("VPC_CIDR", "10.0.0.0/16")
        db_username = os.environ.get("DB_USERNAME", "admin")
        retention_days = int(os.environ.get("LOG_RETENTION_DAYS", "30"))
        
        # Create a VPC
        vpc = ec2.Vpc(
            self, f"{app_name}-vpc",
            vpc_name=f"{app_name}-{environment}-vpc",
            max_azs=2,
            cidr=vpc_cidr,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24
                )
            ],
            nat_gateways=1
        )

        # S3 Bucket for data storage
        data_bucket = s3.Bucket(
            self, f"{app_name}-data-bucket",
            bucket_name=f"{app_name}-{environment}-data",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="archive-old-data",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(90)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(180)
                        )
                    ],
                    expiration=Duration.days(365)
                )
            ]
        )

        # DynamoDB Table for metadata and caching
        metadata_table = dynamodb.Table(
            self, f"{app_name}-metadata-table",
            table_name=f"{app_name}-{environment}-metadata",
            partition_key=dynamodb.Attribute(
                name="pk",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sk",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery=True
        )

        # Create a Secret for database credentials
        db_secret = secretsmanager.Secret(
            self, f"{app_name}-db-credentials",
            secret_name=f"{app_name}/{environment}/db-credentials",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template=f'{{"username": "{db_username}"}}',
                generate_string_key="password",
                exclude_punctuation=True,
                password_length=16
            )
        )

        # PostgreSQL RDS Database
        db_security_group = ec2.SecurityGroup(
            self, f"{app_name}-db-sg",
            vpc=vpc,
            description=f"Security group for {app_name} database",
            security_group_name=f"{app_name}-{environment}-db-sg"
        )

        db_subnet_group = rds.SubnetGroup(
            self, f"{app_name}-db-subnet-group",
            description=f"Subnet group for {app_name} database",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
            )
        )

        database = rds.DatabaseInstance(
            self, f"{app_name}-database",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_14
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE3,
                ec2.InstanceSize.MEDIUM
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_ISOLATED
            ),
            subnet_group=db_subnet_group,
            security_groups=[db_security_group],
            credentials=rds.Credentials.from_secret(db_secret),
            database_name=f"{app_name.replace('-', '_')}_{environment}",
            instance_identifier=f"{app_name}-{environment}",
            backup_retention=Duration.days(7),
            deletion_protection=True,
            removal_policy=RemovalPolicy.SNAPSHOT,
            storage_encrypted=True,
            multi_az=True
        )

        # ECR Repository for Docker images
        repository = ecr.Repository(
            self, f"{app_name}-repository",
            repository_name=f"{app_name}-{environment}",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep only the 10 most recent images",
                    max_image_count=10,
                    rule_priority=1
                )
            ]
        )

        # ECS Cluster
        cluster = ecs.Cluster(
            self, f"{app_name}-cluster",
            vpc=vpc,
            cluster_name=f"{app_name}-{environment}-cluster",
            container_insights=True
        )

        # ECS Task Execution Role
        execution_role = iam.Role(
            self, f"{app_name}-execution-role",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            role_name=f"{app_name}-{environment}-execution-role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ]
        )

        # ECS Task Role
        task_role = iam.Role(
            self, f"{app_name}-task-role",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            role_name=f"{app_name}-{environment}-task-role"
        )

        # Grant S3 permissions to task role
        data_bucket.grant_read_write(task_role)
        
        # Grant DynamoDB permissions to task role
        metadata_table.grant_read_write_data(task_role)
        
        # Grant Secret access to task role
        db_secret.grant_read(task_role)
        
        # CloudWatch Log Group
        log_group = logs.LogGroup(
            self, f"{app_name}-log-group",
            log_group_name=f"/ecs/{app_name}-{environment}",
            removal_policy=RemovalPolicy.DESTROY,
            retention=logs.RetentionDays.from_days(retention_days)
        )

        # Task Definition
        task_definition = ecs.FargateTaskDefinition(
            self, f"{app_name}-task-def",
            family=f"{app_name}-{environment}",
            execution_role=execution_role,
            task_role=task_role,
            cpu=1024,
            memory_limit_mib=2048
        )

        container = task_definition.add_container(
            f"{app_name}-container",
            image=ecs.ContainerImage.from_ecr_repository(repository, "latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=f"{app_name}-{environment}",
                log_group=log_group
            ),
            environment={
                "ENVIRONMENT": environment,
                "S3_BUCKET_NAME": data_bucket.bucket_name,
                "DYNAMODB_TABLE_NAME": metadata_table.table_name,
                "DB_HOST": database.db_instance_endpoint_address,
                "DB_PORT": database.db_instance_endpoint_port,
                "DB_NAME": f"{app_name.replace('-', '_')}_{environment}",
                "SECRET_ARN": db_secret.secret_arn
            },
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8050/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(60)
            )
        )

        container.add_port_mappings(
            ecs.PortMapping(
                container_port=8050,
                host_port=8050,
                protocol=ecs.Protocol.TCP
            )
        )

        # Fargate Service
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, f"{app_name}-service",
            service_name=f"{app_name}-{environment}",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=2,
            min_healthy_percent=50,
            max_healthy_percent=200,
            public_load_balancer=True,
            assign_public_ip=False,
            listener_port=80,
            target_protocol=elbv2.ApplicationProtocol.HTTP,
            circuit_breaker=ecs.DeploymentCircuitBreaker(
                rollback=True
            ),
            health_check_grace_period=Duration.seconds(60)
        )

        # Auto Scaling
        scaling = service.service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=10
        )

        scaling.scale_on_cpu_utilization(
            f"{app_name}-cpu-scaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60)
        )

        # SNS Topic for alerts
        alerts_topic = sns.Topic(
            self, f"{app_name}-alerts-topic",
            topic_name=f"{app_name}-{environment}-alerts",
            display_name=f"{app_name.capitalize()} {environment.capitalize()} Alerts"
        )

        # Subscribe admin email to alerts
        alerts_topic.add_subscription(
            sns_subscriptions.EmailSubscription(admin_email)
        )

        # CloudWatch Alarms
        cpu_alarm = cloudwatch.Alarm(
            self, f"{app_name}-cpu-alarm",
            alarm_name=f"{app_name}-{environment}-high-cpu",
            metric=service.service.metric_cpu_utilization(),
            threshold=85,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING
        )

        memory_alarm = cloudwatch.Alarm(
            self, f"{app_name}-memory-alarm",
            alarm_name=f"{app_name}-{environment}-high-memory",
            metric=service.service.metric_memory_utilization(),
            threshold=85,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING
        )

        # Add alarm actions
        cpu_alarm.add_alarm_action(cloudwatch_actions.SnsAction(alerts_topic))
        memory_alarm.add_alarm_action(cloudwatch_actions.SnsAction(alerts_topic))

        # Scheduled data validation Lambda
        validation_lambda = lambda_.Function(
            self, f"{app_name}-validation-lambda",
            function_name=f"{app_name}-{environment}-data-validation",
            runtime=lambda_.Runtime.PYTHON_3_9,
            code=lambda_.Code.from_asset("src/lambda"),
            handler="data_validation.handler",
            timeout=Duration.minutes(5),
            memory_size=1024,
            environment={
                "S3_BUCKET_NAME": data_bucket.bucket_name,
                "DYNAMODB_TABLE_NAME": metadata_table.table_name,
                "SNS_TOPIC_ARN": alerts_topic.topic_arn,
                "ENVIRONMENT": environment
            }
        )

        # Grant permissions to Lambda
        data_bucket.grant_read(validation_lambda)
        metadata_table.grant_read_data(validation_lambda)
        alerts_topic.grant_publish(validation_lambda)

        # Schedule Lambda to run daily
        validation_schedule = events.Rule(
            self, f"{app_name}-validation-schedule",
            rule_name=f"{app_name}-{environment}-daily-validation",
            schedule=events.Schedule.cron(
                minute="0",
                hour="1",
                month="*",
                week_day="*",
                year="*"
            )
        )

        validation_schedule.add_target(
            events_targets.LambdaFunction(validation_lambda)
        )

        # Outputs
        CfnOutput(
            self, "LoadBalancerDNS",
            description="Load Balancer DNS",
            value=service.load_balancer.load_balancer_dns_name
        )
        
        CfnOutput(
            self, "DataBucketName",
            description="S3 Bucket for Data Storage",
            value=data_bucket.bucket_name
        )
        
        CfnOutput(
            self, "MetadataTableName",
            description="DynamoDB Metadata Table",
            value=metadata_table.table_name
        )
        
        CfnOutput(
            self, "ECRRepositoryURI",
            description="ECR Repository URI",
            value=repository.repository_uri
        )
        
        CfnOutput(
            self, "DatabaseEndpoint",
            description="RDS Database Endpoint",
            value=database.db_instance_endpoint_address
        )
        
        CfnOutput(
            self, "AlertsTopicARN",
            description="SNS Alerts Topic ARN",
            value=alerts_topic.topic_arn
        )


app = App()
SupplyChainAnalyticsStack(app, "SupplyChainAnalyticsStack")
app.synth() 