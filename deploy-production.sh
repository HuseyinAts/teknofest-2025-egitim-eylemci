#!/bin/bash
# TEKNOFEST 2025 - Production Deployment Script
# Usage: ./deploy-production.sh [version]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VERSION=${1:-$(git describe --tags --always)}
ENVIRONMENT="production"
DOCKER_REGISTRY="ghcr.io/teknofest2025"
NAMESPACE="teknofest-prod"
SLACK_WEBHOOK=${SLACK_WEBHOOK:-""}

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

send_slack_notification() {
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST $SLACK_WEBHOOK \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"$1\"}" \
            --silent > /dev/null
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if we're on main branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" != "main" ]; then
        log_warning "Not on main branch. Current branch: $CURRENT_BRANCH"
        read -p "Continue deployment? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Deployment cancelled"
            exit 1
        fi
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_error "Uncommitted changes detected. Please commit or stash changes."
        exit 1
    fi
    
    # Run tests
    log_info "Running tests..."
    pytest tests/ --cov=src --cov-report=term-missing || {
        log_error "Tests failed. Deployment cancelled."
        exit 1
    }
    
    # Security scan
    log_info "Running security scan..."
    python scripts/security_check.py || {
        log_error "Security scan failed. Deployment cancelled."
        exit 1
    }
    
    log_info "Pre-deployment checks passed ✅"
}

# Create backup
create_backup() {
    log_info "Creating backup before deployment..."
    ./scripts/backup-system.sh || {
        log_warning "Backup failed. Continue anyway?"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    }
}

# Build and push Docker image
build_and_push() {
    log_info "Building Docker image..."
    docker build -f Dockerfile.production -t $DOCKER_REGISTRY/api:$VERSION . || {
        log_error "Docker build failed"
        exit 1
    }
    
    # Tag as latest
    docker tag $DOCKER_REGISTRY/api:$VERSION $DOCKER_REGISTRY/api:latest
    
    log_info "Pushing Docker image to registry..."
    docker push $DOCKER_REGISTRY/api:$VERSION || {
        log_error "Docker push failed"
        exit 1
    }
    docker push $DOCKER_REGISTRY/api:latest
    
    log_info "Docker image pushed successfully ✅"
}

# Database migrations
run_migrations() {
    log_info "Running database migrations..."
    python scripts/migrate_production.py || {
        log_error "Database migration failed"
        exit 1
    }
    log_info "Database migrations completed ✅"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Update deployment
    kubectl set image deployment/teknofest-api \
        api=$DOCKER_REGISTRY/api:$VERSION \
        -n $NAMESPACE || {
        log_error "Kubernetes deployment failed"
        exit 1
    }
    
    # Wait for rollout
    log_info "Waiting for rollout to complete..."
    kubectl rollout status deployment/teknofest-api -n $NAMESPACE || {
        log_error "Rollout failed"
        rollback
        exit 1
    }
    
    log_info "Kubernetes deployment completed ✅"
}

# Health check
health_check() {
    log_info "Running health checks..."
    
    # Wait for service to be ready
    sleep 10
    
    # Check health endpoint
    for i in {1..5}; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://api.teknofest2025.com/health)
        if [ "$HTTP_CODE" = "200" ]; then
            log_info "Health check passed ✅"
            return 0
        fi
        log_warning "Health check attempt $i failed. Retrying..."
        sleep 5
    done
    
    log_error "Health checks failed"
    return 1
}

# Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    python scripts/production_tests.py || {
        log_warning "Smoke tests failed"
        return 1
    }
    log_info "Smoke tests passed ✅"
}

# Rollback function
rollback() {
    log_error "Rolling back deployment..."
    kubectl rollout undo deployment/teknofest-api -n $NAMESPACE
    kubectl rollout status deployment/teknofest-api -n $NAMESPACE
    send_slack_notification "⚠️ Deployment rolled back for version $VERSION"
    log_info "Rollback completed"
}

# Clear cache
clear_cache() {
    log_info "Clearing application cache..."
    kubectl exec -it deployment/teknofest-api -n $NAMESPACE -- python -c "
from src.core.cache import cache
cache.delete_pattern('*')
print('Cache cleared')
    " || log_warning "Cache clearing failed"
}

# Main deployment flow
main() {
    echo "========================================="
    echo "   TEKNOFEST 2025 Production Deployment"
    echo "   Version: $VERSION"
    echo "   Environment: $ENVIRONMENT"
    echo "========================================="
    echo
    
    # Confirmation
    log_warning "You are about to deploy to PRODUCTION!"
    read -p "Are you sure? Type 'DEPLOY' to continue: " confirmation
    if [ "$confirmation" != "DEPLOY" ]; then
        log_error "Deployment cancelled"
        exit 1
    fi
    
    # Start deployment
    START_TIME=$(date +%s)
    send_slack_notification "🚀 Starting production deployment for version $VERSION"
    
    # Run deployment steps
    pre_deployment_checks
    create_backup
    build_and_push
    run_migrations
    deploy_to_kubernetes
    
    # Verify deployment
    if health_check; then
        clear_cache
        run_smoke_tests || log_warning "Smoke tests failed but deployment continues"
        
        # Calculate deployment time
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        log_info "========================================="
        log_info "   DEPLOYMENT SUCCESSFUL! 🎉"
        log_info "   Version: $VERSION"
        log_info "   Duration: ${DURATION}s"
        log_info "========================================="
        
        send_slack_notification "✅ Production deployment successful!
Version: $VERSION
Duration: ${DURATION}s
URL: https://api.teknofest2025.com"
        
        # Tag the release
        git tag -a "deployed-$VERSION" -m "Deployed to production on $(date)"
        git push origin "deployed-$VERSION"
        
    else
        log_error "Deployment verification failed"
        rollback
        exit 1
    fi
}

# Run main function
main

# Post-deployment tasks
log_info "Running post-deployment tasks..."

# Update monitoring dashboard
curl -X POST https://grafana.teknofest2025.com/api/annotations \
    -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"dashboardId\": 1,
        \"text\": \"Deployment: $VERSION\",
        \"tags\": [\"deployment\", \"production\"]
    }" || log_warning "Failed to update Grafana"

# Warm up cache
log_info "Warming up cache..."
curl -s https://api.teknofest2025.com/api/v1/warmup > /dev/null

log_info "Deployment complete! 🚀"
