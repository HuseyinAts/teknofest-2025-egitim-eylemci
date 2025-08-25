#!/bin/bash

# Production Deployment Script
# Usage: ./deploy.sh [environment] [version] [options]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_TIMEOUT=600
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

# Default values
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"
DRY_RUN="${DRY_RUN:-false}"
ROLLBACK="${ROLLBACK:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
SKIP_BACKUP="${SKIP_BACKUP:-false}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local tools=("docker" "kubectl" "helm" "git")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
    done
    
    # Check environment configuration
    if [[ ! -f "$PROJECT_ROOT/configs/${ENVIRONMENT}.env" ]]; then
        log_error "Configuration file for ${ENVIRONMENT} not found"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

load_environment() {
    log_info "Loading environment configuration for ${ENVIRONMENT}..."
    
    # Load environment variables
    source "$PROJECT_ROOT/configs/${ENVIRONMENT}.env"
    
    # Set deployment variables
    export NAMESPACE="${KUBE_NAMESPACE:-teknofest-${ENVIRONMENT}}"
    export RELEASE_NAME="${RELEASE_NAME:-teknofest-${ENVIRONMENT}}"
    export IMAGE_TAG="${VERSION}"
    
    log_success "Environment configuration loaded"
}

run_pre_deployment_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping pre-deployment tests"
        return 0
    fi
    
    log_info "Running pre-deployment tests..."
    
    # Run unit tests
    python -m pytest tests/ -m "not integration" --timeout=300 || {
        log_error "Unit tests failed"
        exit 1
    }
    
    # Run security scan
    trivy fs . --severity HIGH,CRITICAL || {
        log_error "Security scan failed"
        exit 1
    }
    
    log_success "Pre-deployment tests passed"
}

backup_current_deployment() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log_warning "Skipping deployment backup"
        return 0
    fi
    
    log_info "Backing up current deployment..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup Kubernetes resources
    kubectl get all -n "$NAMESPACE" -o yaml > "$backup_dir/resources.yaml"
    kubectl get configmap -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml"
    kubectl get secret -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml"
    
    # Backup database
    if [[ -n "${DATABASE_URL:-}" ]]; then
        log_info "Creating database backup..."
        pg_dump "$DATABASE_URL" | gzip > "$backup_dir/database.sql.gz"
    fi
    
    log_success "Backup created at $backup_dir"
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Build image
    docker build \
        --tag "ghcr.io/huseyinats/teknofest-egitim:${VERSION}" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="${VERSION}" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        "$PROJECT_ROOT"
    
    # Push to registry
    docker push "ghcr.io/huseyinats/teknofest-egitim:${VERSION}"
    
    log_success "Docker image built and pushed"
}

deploy_with_helm() {
    log_info "Deploying with Helm..."
    
    local helm_args=(
        "--namespace" "$NAMESPACE"
        "--create-namespace"
        "--set" "image.tag=${VERSION}"
        "--set" "environment=${ENVIRONMENT}"
        "--timeout" "${DEPLOYMENT_TIMEOUT}s"
        "--wait"
    )
    
    # Add environment-specific values
    if [[ -f "$PROJECT_ROOT/k8s/helm/values-${ENVIRONMENT}.yaml" ]]; then
        helm_args+=("-f" "$PROJECT_ROOT/k8s/helm/values-${ENVIRONMENT}.yaml")
    fi
    
    # Dry run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_args+=("--dry-run" "--debug")
        log_warning "Running in dry-run mode"
    fi
    
    # Deploy or upgrade
    helm upgrade --install \
        "$RELEASE_NAME" \
        "$PROJECT_ROOT/k8s/helm" \
        "${helm_args[@]}"
    
    log_success "Helm deployment completed"
}

perform_blue_green_deployment() {
    log_info "Performing blue-green deployment..."
    
    # Get current active color
    local current_color=$(kubectl get service "${RELEASE_NAME}" -n "$NAMESPACE" \
        -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue")
    
    local new_color="green"
    if [[ "$current_color" == "green" ]]; then
        new_color="blue"
    fi
    
    log_info "Current: ${current_color}, New: ${new_color}"
    
    # Deploy new version
    helm upgrade --install \
        "${RELEASE_NAME}-${new_color}" \
        "$PROJECT_ROOT/k8s/helm" \
        --namespace "$NAMESPACE" \
        --set "image.tag=${VERSION}" \
        --set "deployment.color=${new_color}" \
        --wait \
        --timeout "${DEPLOYMENT_TIMEOUT}s"
    
    # Run health checks on new deployment
    wait_for_deployment "${RELEASE_NAME}-${new_color}"
    
    # Switch traffic
    kubectl patch service "${RELEASE_NAME}" -n "$NAMESPACE" \
        -p "{\"spec\":{\"selector\":{\"color\":\"${new_color}\"}}}"
    
    log_success "Traffic switched to ${new_color} deployment"
    
    # Wait before removing old deployment
    sleep 60
    
    # Scale down old deployment
    kubectl scale deployment "${RELEASE_NAME}-${current_color}" \
        --replicas=0 -n "$NAMESPACE"
    
    log_success "Blue-green deployment completed"
}

wait_for_deployment() {
    local deployment_name="$1"
    log_info "Waiting for deployment ${deployment_name} to be ready..."
    
    kubectl rollout status deployment/"${deployment_name}" \
        -n "$NAMESPACE" \
        --timeout="${DEPLOYMENT_TIMEOUT}s"
}

run_health_checks() {
    log_info "Running health checks..."
    
    local service_url
    if [[ "$ENVIRONMENT" == "production" ]]; then
        service_url="https://teknofest-egitim.example.com"
    else
        service_url="https://${ENVIRONMENT}.teknofest-egitim.example.com"
    fi
    
    local retry_count=0
    while [[ $retry_count -lt $HEALTH_CHECK_RETRIES ]]; do
        if curl -fsS "${service_url}/health" > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        log_warning "Health check failed (attempt ${retry_count}/${HEALTH_CHECK_RETRIES})"
        sleep "$HEALTH_CHECK_INTERVAL"
    done
    
    log_error "Health checks failed after ${HEALTH_CHECK_RETRIES} attempts"
    return 1
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Run basic API tests
    python -m pytest tests/test_smoke.py \
        --env="${ENVIRONMENT}" \
        --timeout=60 || {
        log_error "Smoke tests failed"
        return 1
    }
    
    log_success "Smoke tests passed"
}

perform_rollback() {
    log_error "Performing rollback..."
    
    # Rollback Helm release
    helm rollback "$RELEASE_NAME" -n "$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/"${RELEASE_NAME}" \
        -n "$NAMESPACE" \
        --timeout="${DEPLOYMENT_TIMEOUT}s"
    
    log_warning "Rollback completed"
}

update_monitoring() {
    log_info "Updating monitoring and alerts..."
    
    # Update Prometheus alerts
    kubectl apply -f "$PROJECT_ROOT/monitoring/alerts-${ENVIRONMENT}.yaml" \
        -n monitoring
    
    # Update Grafana dashboards
    kubectl apply -f "$PROJECT_ROOT/monitoring/dashboards/" \
        -n monitoring
    
    log_success "Monitoring updated"
}

send_notification() {
    local status="$1"
    local message="$2"
    
    log_info "Sending deployment notification..."
    
    # Send Slack notification
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{
                \"text\": \"Deployment ${status}\",
                \"attachments\": [{
                    \"color\": \"${status,,}\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"${ENVIRONMENT}\", \"short\": true},
                        {\"title\": \"Version\", \"value\": \"${VERSION}\", \"short\": true},
                        {\"title\": \"Message\", \"value\": \"${message}\"}
                    ]
                }]
            }"
    fi
}

cleanup() {
    log_info "Performing cleanup..."
    
    # Clean old images
    docker image prune -af --filter "until=24h"
    
    # Clean old Kubernetes resources
    kubectl delete pods --field-selector status.phase=Failed -n "$NAMESPACE"
    kubectl delete pods --field-selector status.phase=Succeeded -n "$NAMESPACE"
    
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    log_info "Starting deployment to ${ENVIRONMENT} with version ${VERSION}"
    
    # Pre-deployment
    check_prerequisites
    load_environment
    
    if [[ "$ROLLBACK" == "true" ]]; then
        perform_rollback
        send_notification "ROLLBACK" "Rolled back to previous version"
        exit 0
    fi
    
    run_pre_deployment_tests
    backup_current_deployment
    
    # Build and deploy
    build_and_push_image
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        perform_blue_green_deployment
    else
        deploy_with_helm
        wait_for_deployment "$RELEASE_NAME"
    fi
    
    # Post-deployment
    if run_health_checks && run_smoke_tests; then
        update_monitoring
        cleanup
        send_notification "SUCCESS" "Deployment completed successfully"
        log_success "Deployment completed successfully!"
    else
        perform_rollback
        send_notification "FAILED" "Deployment failed and rolled back"
        log_error "Deployment failed!"
        exit 1
    fi
}

# Handle interrupts
trap 'log_error "Deployment interrupted"; perform_rollback; exit 1' INT TERM

# Run main function
main "$@"