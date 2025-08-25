# Production-Ready Multi-Worker Deployment Guide
## TEKNOFEST 2025 - Eğitim Eylemci

## Overview

This guide provides comprehensive instructions for deploying the TEKNOFEST 2025 application in a production-ready multi-worker configuration with auto-scaling, high availability, and advanced monitoring.

## Architecture Components

### Core Services
- **HAProxy**: Load balancer with health checks and circuit breaking
- **API Workers**: Auto-scaling FastAPI/Gunicorn workers (2-20 replicas)
- **Celery Workers**: Task queue workers with specialized queues
- **PostgreSQL**: Primary-replica database setup with connection pooling
- **Redis**: Master-replica cache with Sentinel for HA
- **RabbitMQ**: Message broker for reliable task queuing

### Monitoring & Observability
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing
- **Elasticsearch/Kibana**: Log analysis
- **Alertmanager**: Alert routing

### Orchestration
- **Container Orchestrator**: Custom Python-based orchestrator for Docker
- **Kubernetes Support**: Full K8s deployment manifests with HPA/VPA
- **Health Checks**: Comprehensive liveness and readiness probes

## Deployment Options

### Option 1: Docker Compose (Recommended for Single-Host)

```bash
# Deploy with Docker Compose
docker-compose -f docker-compose.production-multiworker.yml up -d

# Scale services manually
docker-compose -f docker-compose.production-multiworker.yml scale api=8

# Monitor logs
docker-compose -f docker-compose.production-multiworker.yml logs -f
```

### Option 2: Docker Swarm (Multi-Host Cluster)

```bash
# Initialize Swarm
docker swarm init --advertise-addr <MANAGER-IP>

# Deploy stack
docker stack deploy -c docker-compose.production-multiworker.yml teknofest

# Scale services
docker service scale teknofest_api=10
```

### Option 3: Kubernetes (Enterprise-Grade)

```bash
# Create namespace and deploy
kubectl apply -f k8s/production-deployment.yaml

# Check deployment status
kubectl get pods -n teknofest-production

# Access dashboard
kubectl port-forward -n teknofest-production svc/grafana 3000:3000
```

## Configuration

### Environment Variables

Create `.env.production` file:

```env
# Application
APP_ENV=production
SECRET_KEY=your-secure-secret-key
DEBUG=false

# Database
DB_HOST=postgres-primary
DB_PORT=5432
DB_NAME=teknofest_db
DB_USER=teknofest
DB_PASSWORD=secure-password
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Redis
REDIS_HOST=redis-master
REDIS_PORT=6379
REDIS_PASSWORD=redis-password
REDIS_DB=0

# Celery
CELERY_BROKER_URL=redis://redis-master:6379/1
CELERY_RESULT_BACKEND=redis://redis-master:6379/2

# Scaling
MIN_API_WORKERS=2
MAX_API_WORKERS=20
MIN_CELERY_WORKERS=1
MAX_CELERY_WORKERS=10
SCALE_UP_THRESHOLD=80
SCALE_DOWN_THRESHOLD=30

# Monitoring
PROMETHEUS_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
GRAFANA_USER=admin
GRAFANA_PASSWORD=secure-password
```

### HAProxy Configuration

The HAProxy configuration (`deploy/haproxy/haproxy.cfg`) includes:
- Multiple backends with health checks
- Rate limiting (100 req/s per IP)
- Circuit breaker pattern
- WebSocket support
- SSL/TLS termination
- Session affinity

### Worker Configuration

#### API Workers
- **Min Replicas**: 2
- **Max Replicas**: 20
- **CPU Target**: 70%
- **Memory Target**: 80%
- **Scaling Policy**: Aggressive scale-up, gradual scale-down

#### Celery Workers
- **Default Queue**: 2-10 workers, general tasks
- **AI Queue**: 2-4 workers, GPU-enabled for ML tasks
- **Data Queue**: 2-10 workers, ETL operations

## Monitoring Setup

### Access Points

- **Grafana Dashboard**: http://localhost:3001 (admin/teknofest2025)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **Jaeger UI**: http://localhost:16686
- **Flower (Celery)**: http://localhost:5555
- **HAProxy Stats**: http://localhost:8404/stats

### Key Metrics to Monitor

1. **Application Metrics**
   - Request rate and latency
   - Error rate (4xx, 5xx)
   - Active connections
   - Queue lengths

2. **System Metrics**
   - CPU utilization
   - Memory usage
   - Disk I/O
   - Network traffic

3. **Database Metrics**
   - Connection pool usage
   - Query performance
   - Replication lag
   - Lock waits

4. **Custom Business Metrics**
   - User activity
   - Task completion rate
   - AI model inference time

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

```yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
- type: Resource
  resource:
    name: memory
    target:
      type: Utilization
      averageUtilization: 80
- type: Pods
  pods:
    metric:
      name: http_requests_per_second
    target:
      type: AverageValue
      averageValue: "100"
```

### Scaling Behavior

```yaml
behavior:
  scaleDown:
    stabilizationWindowSeconds: 300
    policies:
    - type: Percent
      value: 50
      periodSeconds: 60
  scaleUp:
    stabilizationWindowSeconds: 60
    policies:
    - type: Percent
      value: 100
      periodSeconds: 30
```

## Health Checks

### API Health Endpoints

- `/health/live` - Liveness probe (is the service running?)
- `/health/ready` - Readiness probe (can it accept traffic?)
- `/health/startup` - Startup probe (is initialization complete?)

### Database Health Check

```python
async def check_database_health():
    try:
        await db.execute("SELECT 1")
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Redis Health Check

```python
async def check_redis_health():
    try:
        await redis.ping()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Deployment Checklist

### Pre-Deployment

- [ ] Review and update environment variables
- [ ] Generate secure passwords and keys
- [ ] Configure SSL certificates
- [ ] Set up monitoring dashboards
- [ ] Configure backup strategy
- [ ] Review security settings
- [ ] Test disaster recovery procedures

### Deployment

- [ ] Deploy infrastructure services (DB, Redis, RabbitMQ)
- [ ] Run database migrations
- [ ] Deploy application services
- [ ] Configure load balancer
- [ ] Set up monitoring agents
- [ ] Configure auto-scaling policies
- [ ] Test health checks

### Post-Deployment

- [ ] Verify all services are healthy
- [ ] Check monitoring dashboards
- [ ] Run smoke tests
- [ ] Configure alerting rules
- [ ] Document deployment details
- [ ] Set up log rotation
- [ ] Schedule regular backups

## Maintenance Operations

### Rolling Updates

```bash
# Docker Compose
docker-compose -f docker-compose.production-multiworker.yml pull
docker-compose -f docker-compose.production-multiworker.yml up -d --no-deps --scale api=8 api

# Kubernetes
kubectl set image deployment/api api=teknofest/api:v2.0.0 -n teknofest-production
kubectl rollout status deployment/api -n teknofest-production
```

### Database Backup

```bash
# Automated backup (runs daily via Celery beat)
docker exec teknofest-postgres-primary pg_dump -U teknofest teknofest_db > backup_$(date +%Y%m%d).sql

# Manual backup
docker-compose -f docker-compose.production-multiworker.yml exec postgres-primary \
    pg_dump -U teknofest teknofest_db | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz
```

### Log Management

```bash
# View logs
docker-compose -f docker-compose.production-multiworker.yml logs -f api

# Export logs
docker logs teknofest-api-primary > api_logs_$(date +%Y%m%d).log

# Clean old logs
find /var/log/teknofest -name "*.log" -mtime +30 -delete
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks in application code
   - Review worker configurations
   - Adjust `MAX_REQUESTS` to restart workers periodically

2. **Database Connection Errors**
   - Check connection pool settings
   - Review PgBouncer configuration
   - Monitor connection counts

3. **Slow Response Times**
   - Check load balancer configuration
   - Review database query performance
   - Monitor Redis cache hit rates

4. **Worker Crashes**
   - Check worker logs for errors
   - Review resource limits
   - Monitor Celery queue lengths

### Debug Commands

```bash
# Check service status
docker-compose -f docker-compose.production-multiworker.yml ps

# Inspect container
docker inspect teknofest-api-primary

# Execute commands in container
docker exec -it teknofest-api-primary /bin/bash

# Check resource usage
docker stats

# View network connections
docker exec teknofest-api-primary netstat -tuln
```

## Security Considerations

### Network Security
- All services run in isolated networks
- External access only through load balancer
- Internal services use private IPs

### Application Security
- Non-root user in containers
- Read-only root filesystem where possible
- Security headers configured
- Rate limiting enabled

### Data Security
- Encrypted connections (TLS/SSL)
- Secure password storage
- Regular security updates
- Audit logging enabled

## Performance Optimization

### Database Optimization
- Connection pooling with PgBouncer
- Query optimization and indexing
- Read replicas for scaling reads
- Partitioning for large tables

### Caching Strategy
- Redis for session storage
- Application-level caching
- CDN for static assets
- Browser caching headers

### Application Optimization
- Async request handling
- Background task processing
- Response compression
- Lazy loading

## Disaster Recovery

### Backup Strategy
- Daily automated backups
- Off-site backup storage
- Point-in-time recovery
- Regular restore testing

### Failover Procedures
1. Database failover to replica
2. Redis Sentinel automatic failover
3. Load balancer health checks
4. Container auto-restart

### Recovery Time Objectives
- **RTO**: < 1 hour
- **RPO**: < 15 minutes

## Support and Resources

### Documentation
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [HAProxy Documentation](http://www.haproxy.org/#docs)
- [Celery Documentation](https://docs.celeryproject.org/)

### Monitoring Resources
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [ELK Stack Guide](https://www.elastic.co/guide/)

### Contact Information
- **DevOps Team**: devops@teknofest.org
- **On-Call**: +90-XXX-XXX-XXXX
- **Slack Channel**: #teknofest-ops

## Version History

- **v1.0.0** - Initial production deployment
- **v1.1.0** - Added Kubernetes support
- **v1.2.0** - Enhanced monitoring and auto-scaling
- **v1.3.0** - Added disaster recovery procedures

---

Last Updated: 2024
TEKNOFEST 2025 - Production Deployment Guide