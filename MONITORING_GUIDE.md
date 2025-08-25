# TEKNOFEST 2025 - Production Monitoring Guide

## Overview

Production-ready monitoring stack for TEKNOFEST 2025 Eğitim Teknolojileri platform using Grafana, Prometheus, Loki, and various exporters.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Grafana UI                            │
│                    (Dashboards & Alerts)                     │
└──────────┬────────────────┬────────────┬──────────┬─────────┘
           │                │            │          │
    ┌──────▼──────┐  ┌──────▼──────┐  ┌─▼──┐  ┌───▼───┐
    │ Prometheus  │  │    Loki     │  │Tempo│  │Jaeger │
    │  (Metrics)  │  │   (Logs)    │  │     │  │       │
    └──────┬──────┘  └──────┬──────┘  └──┬──┘  └───┬───┘
           │                │             │          │
    ┌──────▼────────────────▼─────────────▼──────────▼────────┐
    │                    Data Sources                          │
    ├──────────────────────────────────────────────────────────┤
    │ • Node Exporter      • PostgreSQL Exporter              │
    │ • Redis Exporter     • Nginx Exporter                   │
    │ • Blackbox Exporter  • cAdvisor                         │
    │ • Application Metrics • Promtail (Log Shipper)          │
    └──────────────────────────────────────────────────────────┘
```

## Components

### 1. Grafana
- **Port**: 3000
- **Default Credentials**: Set via environment variables
- **Features**:
  - Pre-configured dashboards
  - Alert management
  - Data source provisioning
  - Custom panels and plugins

### 2. Prometheus
- **Port**: 9090
- **Retention**: 30 days
- **Storage**: 10GB max
- **Scrape Interval**: 15s
- **Features**:
  - Service discovery
  - Alert rules
  - PromQL queries
  - Federation support

### 3. Loki
- **Port**: 3100
- **Retention**: 31 days
- **Features**:
  - Log aggregation
  - LogQL queries
  - Label-based indexing
  - Compression

### 4. AlertManager
- **Port**: 9093
- **Features**:
  - Alert routing
  - Grouping
  - Silencing
  - Inhibition
  - Multiple receivers (Email, Slack, PagerDuty)

### 5. Jaeger
- **Port**: 16686 (UI)
- **Features**:
  - Distributed tracing
  - Service dependency analysis
  - Performance optimization
  - Root cause analysis

## Dashboards

### Main Dashboard
- System uptime
- Request rate
- Error metrics
- CPU/Memory usage
- Response time percentiles
- HTTP status codes
- Database connections
- Redis operations

### API Performance Dashboard
- Endpoint-specific metrics
- Latency heatmaps
- Error rates by endpoint
- Top 10 endpoints
- Request distribution

### Database Dashboard (To be created)
- Connection pool metrics
- Query performance
- Table statistics
- Replication lag
- Lock monitoring

### Infrastructure Dashboard (To be created)
- Container metrics
- Network I/O
- Disk usage
- System load
- Service health

## Alert Rules

### Critical Alerts
- Service down
- Database unreachable
- High error rate (>10%)
- Possible DDoS attack
- Security breaches

### Warning Alerts
- High CPU usage (>80%)
- High memory usage (>90%)
- Slow response times
- Database connection pool exhaustion
- Cache misses

### Info Alerts
- Low user engagement
- Scheduled maintenance reminders
- Certificate expiration warnings

## Setup Instructions

### 1. Environment Variables

Create `.env` file with monitoring credentials:

```bash
# Grafana
GRAFANA_USER=admin
GRAFANA_PASSWORD=secure_password
GRAFANA_URL=http://localhost:3000

# AlertManager
SMTP_USER=alerts@teknofest2025.com
SMTP_PASSWORD=smtp_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
PAGERDUTY_SERVICE_KEY=your_key

# Prometheus
PROMETHEUS_PORT=9090

# Database Monitoring
DB_USER=teknofest
DB_PASSWORD=db_password
DB_NAME=teknofest
```

### 2. Start Monitoring Stack

```bash
# Start main services
docker-compose -f docker-compose.production.yml up -d

# Start additional monitoring services
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093
- **Jaeger**: http://localhost:16686

### 4. Configure Alerts

1. Update email settings in `monitoring/alertmanager/alertmanager.yml`
2. Configure Slack webhooks
3. Set up PagerDuty integration
4. Test alert routing

## Monitoring Best Practices

### 1. Metric Naming Convention
```
<namespace>_<subsystem>_<name>_<unit>

Examples:
- http_requests_total
- db_connections_active
- cache_hits_total
- api_response_time_seconds
```

### 2. Label Usage
- Keep cardinality low
- Use consistent label names
- Avoid high-cardinality labels (user_id, request_id)
- Use label values that are bounded

### 3. Dashboard Design
- Group related metrics
- Use appropriate visualization types
- Set meaningful thresholds
- Include documentation panels
- Use variables for flexibility

### 4. Alert Configuration
- Set appropriate thresholds
- Use evaluation periods to reduce noise
- Group related alerts
- Include runbook links
- Test alert routing regularly

## Troubleshooting

### Common Issues

#### 1. Grafana Not Loading Data
```bash
# Check Prometheus connectivity
curl http://prometheus:9090/api/v1/query?query=up

# Check data source configuration
docker exec teknofest-grafana cat /etc/grafana/provisioning/datasources/prometheus.yml
```

#### 2. Missing Metrics
```bash
# Check exporter status
curl http://localhost:9100/metrics  # Node exporter
curl http://localhost:9187/metrics  # PostgreSQL exporter
curl http://localhost:9121/metrics  # Redis exporter

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

#### 3. Alerts Not Firing
```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Check AlertManager config
docker exec teknofest-alertmanager amtool config show

# Test alert routing
docker exec teknofest-alertmanager amtool config routes test
```

#### 4. High Memory Usage
```bash
# Check Prometheus storage
du -sh /var/lib/docker/volumes/*prometheus*

# Reduce retention if needed
docker exec teknofest-prometheus promtool tsdb analyze /prometheus

# Clean up old data
docker exec teknofest-prometheus promtool tsdb clean /prometheus
```

## Performance Optimization

### 1. Prometheus
- Adjust scrape intervals based on needs
- Use recording rules for expensive queries
- Implement federation for large deployments
- Use remote storage for long-term retention

### 2. Grafana
- Enable query caching
- Use dashboard refresh intervals wisely
- Optimize queries with time ranges
- Use variables to reduce query count

### 3. Loki
- Configure appropriate retention periods
- Use index optimization
- Implement log sampling for high-volume logs
- Use structured logging

## Security Considerations

### 1. Authentication
- Enable Grafana authentication
- Use strong passwords
- Implement LDAP/OAuth integration
- Enable 2FA for admin accounts

### 2. Network Security
- Use TLS for all connections
- Restrict access with firewalls
- Implement network segmentation
- Use VPN for remote access

### 3. Data Protection
- Encrypt sensitive metrics
- Implement data retention policies
- Regular backup of configurations
- Audit log access

## Maintenance Tasks

### Daily
- Check alert notifications
- Review error rates
- Monitor disk usage
- Verify backup completion

### Weekly
- Review dashboard usage
- Update alert thresholds
- Check for software updates
- Test disaster recovery

### Monthly
- Clean up unused dashboards
- Review and optimize queries
- Update documentation
- Performance analysis

## Integration with CI/CD

### GitHub Actions
```yaml
- name: Send metrics to Prometheus
  run: |
    curl -X POST http://prometheus:9090/api/v1/write \
      -H "Content-Type: application/x-prometheus-remote-write-1.0" \
      -d "deployment_success{env=\"production\",version=\"${{ github.sha }}\"} 1"
```

### Deployment Notifications
```bash
# Send deployment marker to Grafana
curl -X POST http://grafana:3000/api/annotations \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "dashboardId": 1,
    "time": '$(date +%s000)',
    "tags": ["deployment"],
    "text": "Deployed version '$VERSION'"
  }'
```

## Support

For issues or questions:
- Check logs: `docker-compose logs grafana`
- Review metrics: http://localhost:9090/graph
- Consult documentation: [Grafana Docs](https://grafana.com/docs/)
- Contact: devops@teknofest2025.com