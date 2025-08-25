#!/bin/bash

# TEKNOFEST 2025 - Monitoring Stack Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    echo -e "${2}${1}${NC}"
}

# Function to check if service is healthy
check_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    print_message "Checking $service on port $port..." "$YELLOW"
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            print_message "✓ $service is healthy" "$GREEN"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_message "✗ $service failed to start" "$RED"
    return 1
}

# Function to create required directories
create_directories() {
    print_message "Creating monitoring directories..." "$YELLOW"
    
    directories=(
        "monitoring/grafana/provisioning/datasources"
        "monitoring/grafana/provisioning/dashboards"
        "monitoring/grafana/dashboards"
        "monitoring/prometheus"
        "monitoring/loki"
        "monitoring/promtail"
        "monitoring/alertmanager"
        "monitoring/blackbox-exporter"
        "monitoring/postgres-exporter"
        "monitoring/tempo"
        "monitoring/grafana-agent"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        echo "  Created: $dir"
    done
    
    print_message "✓ Directories created" "$GREEN"
}

# Function to check environment variables
check_environment() {
    print_message "Checking environment variables..." "$YELLOW"
    
    required_vars=(
        "DB_USER"
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "GRAFANA_USER"
        "GRAFANA_PASSWORD"
    )
    
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_message "✗ Missing environment variables: ${missing_vars[*]}" "$RED"
        print_message "Please set these variables in your .env file" "$YELLOW"
        exit 1
    fi
    
    print_message "✓ Environment variables OK" "$GREEN"
}

# Function to start monitoring stack
start_monitoring() {
    print_message "\n🚀 Starting TEKNOFEST 2025 Monitoring Stack..." "$GREEN"
    
    # Check if docker-compose files exist
    if [ ! -f "docker-compose.production.yml" ]; then
        print_message "✗ docker-compose.production.yml not found" "$RED"
        exit 1
    fi
    
    # Start main production stack if not running
    print_message "\nStarting main production services..." "$YELLOW"
    docker-compose -f docker-compose.production.yml up -d \
        postgres redis prometheus grafana loki promtail alertmanager jaeger
    
    # Wait for core services
    print_message "\nWaiting for core services to be healthy..." "$YELLOW"
    check_service "PostgreSQL" 5432
    check_service "Redis" 6379
    check_service "Prometheus" 9090
    check_service "Grafana" 3000
    check_service "Loki" 3100
    check_service "AlertManager" 9093
    check_service "Jaeger" 16686
    
    # Start additional monitoring services if available
    if [ -f "docker-compose.monitoring.yml" ]; then
        print_message "\nStarting additional monitoring services..." "$YELLOW"
        docker-compose -f docker-compose.monitoring.yml up -d
        
        # Wait for exporters
        check_service "Node Exporter" 9100
        check_service "PostgreSQL Exporter" 9187
        check_service "Redis Exporter" 9121
        check_service "Blackbox Exporter" 9115
    fi
    
    print_message "\n✓ Monitoring stack started successfully!" "$GREEN"
}

# Function to show access URLs
show_urls() {
    print_message "\n📊 Access URLs:" "$GREEN"
    echo "  Grafana:       http://localhost:3000"
    echo "    Username:    ${GRAFANA_USER}"
    echo "    Password:    ${GRAFANA_PASSWORD}"
    echo ""
    echo "  Prometheus:    http://localhost:9090"
    echo "  AlertManager:  http://localhost:9093"
    echo "  Jaeger:        http://localhost:16686"
    echo "  Loki:          http://localhost:3100"
    
    if [ -f "docker-compose.monitoring.yml" ]; then
        echo ""
        echo "  Kibana:        http://localhost:5601 (if Elasticsearch is enabled)"
    fi
}

# Function to test monitoring
test_monitoring() {
    print_message "\n🧪 Testing monitoring endpoints..." "$YELLOW"
    
    # Test Prometheus metrics endpoint
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/metrics | grep -q "200"; then
        print_message "✓ Prometheus metrics endpoint OK" "$GREEN"
    else
        print_message "✗ Prometheus metrics endpoint failed" "$RED"
    fi
    
    # Test Grafana API
    if curl -s -o /dev/null -w "%{http_code}" \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        http://localhost:3000/api/health | grep -q "200"; then
        print_message "✓ Grafana API OK" "$GREEN"
    else
        print_message "✗ Grafana API failed" "$RED"
    fi
    
    # Check Prometheus targets
    targets=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | length')
    print_message "✓ Active Prometheus targets: $targets" "$GREEN"
    
    # Check for alerts
    alerts=$(curl -s http://localhost:9090/api/v1/alerts | jq -r '.data.alerts | length')
    if [ "$alerts" -gt 0 ]; then
        print_message "⚠ Active alerts: $alerts" "$YELLOW"
    else
        print_message "✓ No active alerts" "$GREEN"
    fi
}

# Function to import dashboards
import_dashboards() {
    print_message "\n📈 Importing Grafana dashboards..." "$YELLOW"
    
    # Wait for Grafana to be fully ready
    sleep 5
    
    # Get Grafana API key
    api_key=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"monitoring-import\",\"role\":\"Admin\"}" \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        http://localhost:3000/api/auth/keys | jq -r '.key')
    
    if [ -n "$api_key" ] && [ "$api_key" != "null" ]; then
        print_message "✓ Grafana API key created" "$GREEN"
        
        # Import dashboards
        for dashboard in monitoring/grafana/dashboards/*.json; do
            if [ -f "$dashboard" ]; then
                dashboard_name=$(basename "$dashboard" .json)
                response=$(curl -s -X POST \
                    -H "Authorization: Bearer $api_key" \
                    -H "Content-Type: application/json" \
                    -d "@$dashboard" \
                    http://localhost:3000/api/dashboards/db)
                
                if echo "$response" | grep -q "success"; then
                    print_message "  ✓ Imported: $dashboard_name" "$GREEN"
                else
                    print_message "  ✗ Failed to import: $dashboard_name" "$RED"
                fi
            fi
        done
    else
        print_message "⚠ Could not create Grafana API key (dashboards may be auto-provisioned)" "$YELLOW"
    fi
}

# Main execution
main() {
    print_message "═══════════════════════════════════════════════════════" "$GREEN"
    print_message "   TEKNOFEST 2025 - Monitoring Stack Manager" "$GREEN"
    print_message "═══════════════════════════════════════════════════════" "$GREEN"
    
    # Load environment variables
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
        print_message "✓ Environment variables loaded from .env" "$GREEN"
    else
        print_message "⚠ No .env file found, using system environment" "$YELLOW"
    fi
    
    # Execute steps
    check_environment
    create_directories
    start_monitoring
    test_monitoring
    import_dashboards
    show_urls
    
    print_message "\n═══════════════════════════════════════════════════════" "$GREEN"
    print_message "   Monitoring stack is ready! 🎉" "$GREEN"
    print_message "═══════════════════════════════════════════════════════" "$GREEN"
    
    # Show logs command
    print_message "\nTo view logs, run:" "$YELLOW"
    echo "  docker-compose -f docker-compose.production.yml logs -f grafana"
    echo "  docker-compose -f docker-compose.production.yml logs -f prometheus"
    
    print_message "\nTo stop monitoring, run:" "$YELLOW"
    echo "  docker-compose -f docker-compose.production.yml down"
}

# Run main function
main "$@"