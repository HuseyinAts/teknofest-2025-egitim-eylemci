# TEKNOFEST 2025 - Eğitim Teknolojileri Makefile
.PHONY: help install dev-install test lint format type-check clean run docker-build docker-up docker-down all check

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose -f docker-compose.optimized.yml
PROJECT_NAME := teknofest-2025-egitim-eylemci
VERSION := 1.0.0
SRC_DIR := src
TEST_DIR := tests

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)TEKNOFEST 2025 - Eğitim Eylemleri Ajanı$(NC)"
	@echo "$(YELLOW)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install base dependencies
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Base dependencies installed$(NC)"

install-dev: install ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

install-prod: ## Install production dependencies
	$(PIP) install -e ".[production]"
	@echo "$(GREEN)✓ Production dependencies installed$(NC)"

install-model: ## Install model dependencies
	$(PIP) install -r requirements-model.txt
	@echo "$(GREEN)✓ Model dependencies installed$(NC)"

test: ## Run unit tests
	$(PYTHON) run_automated_tests.py --type unit --verbose

test-cov: ## Run tests with coverage report
	$(PYTHON) scripts/coverage_report.py --markdown --badges
	@echo "$(GREEN)✓ Coverage report generated in test_results/coverage/$(NC)"

coverage: ## Generate detailed coverage report
	$(PYTHON) scripts/coverage_report.py --markdown --badges
	@echo "$(GREEN)✓ Coverage analysis complete$(NC)"

coverage-html: ## Generate HTML coverage report
	$(PYTHON) scripts/coverage_report.py
	@echo "$(GREEN)✓ HTML coverage report: test_results/coverage/html/index.html$(NC)"

coverage-badges: ## Generate coverage badges
	$(PYTHON) scripts/coverage_report.py --badges
	@echo "$(GREEN)✓ Coverage badges generated$(NC)"

coverage-sonar: ## Export coverage for SonarQube
	$(PYTHON) scripts/coverage_report.py --sonar
	@echo "$(GREEN)✓ SonarQube coverage exported$(NC)"

coverage-check: ## Check coverage thresholds
	@$(PYTHON) -c "import json; d=json.load(open('test_results/coverage/coverage.json')); c=d['totals']['percent_covered']; print(f'Coverage: {c:.2f}%'); exit(0 if c>=80 else 1)"

test-integration: ## Run integration tests
	$(PYTHON) run_automated_tests.py --type integration --verbose

test-smoke: ## Run smoke tests
	$(PYTHON) run_automated_tests.py --type smoke --fail-fast

test-e2e: ## Run end-to-end tests
	$(PYTHON) run_automated_tests.py --type e2e

test-performance: ## Run performance tests
	$(PYTHON) run_automated_tests.py --type performance

test-security: ## Run security tests
	$(PYTHON) run_automated_tests.py --type security

test-all: ## Run all test suites
	$(PYTHON) run_automated_tests.py --type all --parallel

test-quick: ## Run quick smoke tests
	$(PYTHON) -m pytest tests/ -m "smoke and not slow" -v -x --maxfail=5

test-watch: ## Run tests in watch mode
	$(PYTHON) -m pytest tests/ -v --watch

lint: ## Run code linters
	$(PYTHON) -m flake8 src/ backend/app/ tests/
	$(PYTHON) -m mypy src/ backend/app/
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	$(PYTHON) -m black src/ backend/app/ tests/
	$(PYTHON) -m isort src/ backend/app/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean: ## Clean cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	rm -f nul 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned cache and temporary files$(NC)"

# Database commands
db-init: ## Initialize database tables
	$(PYTHON) scripts/seed_database.py --init-db
	@echo "$(GREEN)✓ Database initialized$(NC)"

db-migrate: ## Run database migrations
	cd backend && alembic upgrade head
	@echo "$(GREEN)✓ Database migrations complete$(NC)"

db-rollback: ## Rollback last migration
	cd backend && alembic downgrade -1
	@echo "$(YELLOW)⚠ Database rolled back one migration$(NC)"

db-seed-dev: ## Seed database for development
	$(PYTHON) scripts/seed_database.py --env development
	@echo "$(GREEN)✓ Development data seeded$(NC)"

db-seed-staging: ## Seed database for staging
	$(PYTHON) scripts/seed_database.py --env staging
	@echo "$(GREEN)✓ Staging data seeded$(NC)"

db-seed-prod: ## Seed database for production (use with caution!)
	$(PYTHON) scripts/seed_database.py --env production
	@echo "$(GREEN)✓ Production data seeded$(NC)"

db-clear: ## Clear all database data (development only)
	$(PYTHON) scripts/seed_database.py --clear
	@echo "$(YELLOW)⚠ Database cleared$(NC)"

db-stats: ## Show database seed statistics
	$(PYTHON) scripts/seed_database.py --stats

db-reset: db-clear db-init db-seed-dev ## Reset database with fresh development data
	@echo "$(GREEN)✓ Database reset complete$(NC)"

# Docker commands
docker-build: ## Build Docker images
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-up: ## Start Docker containers
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Docker containers started$(NC)"
	@echo "API: http://localhost:8000"
	@echo "MCP: http://localhost:8080"

docker-down: ## Stop Docker containers
	$(DOCKER_COMPOSE) down
	@echo "$(YELLOW)✓ Docker containers stopped$(NC)"

docker-logs: ## Show Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-clean: docker-down ## Clean Docker volumes
	$(DOCKER_COMPOSE) down -v
	@echo "$(YELLOW)✓ Docker volumes cleaned$(NC)"

# Development commands
dev: ## Start development server
	$(PYTHON) run_server_lightweight.py

dev-full: ## Start full development server with model
	$(PYTHON) src/api_server_integrated.py

dev-docker: ## Start development with Docker
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml up

# Production commands
prod-build: ## Build for production
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)✓ Production build complete$(NC)"

prod-deploy: ## Deploy to production (requires configuration)
	@echo "$(YELLOW)Deploying to production...$(NC)"
	# Add your deployment commands here
	@echo "$(GREEN)✓ Deployment complete$(NC)"

# API testing commands
api-test: ## Test API endpoints
	$(PYTHON) test_api_endpoints.py

api-docs: ## Open API documentation
	@echo "Starting server and opening API docs..."
	@$(PYTHON) -c "import webbrowser; webbrowser.open('http://localhost:8000/docs')" &
	$(PYTHON) run_server_lightweight.py

# Model commands
model-download: ## Download model from Hugging Face
	$(PYTHON) download_and_test_model.py

model-train: ## Train the model
	$(PYTHON) -m src.fine_tuning.qwen_trainer

model-evaluate: ## Evaluate model performance
	$(PYTHON) -m src.evaluation.leaderboard

# MCP Server commands
mcp-start: ## Start MCP server
	$(PYTHON) -m src.mcp_server.production_server

mcp-test: ## Test MCP server
	$(PYTHON) test_mcp_server.py

# Utility commands
setup-env: ## Setup environment file
	cp .env.example .env
	@echo "$(GREEN)✓ Environment file created. Please edit .env with your values$(NC)"

check-deps: ## Check for outdated dependencies
	$(PIP) list --outdated

update-deps: ## Update all dependencies
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)✓ Dependencies updated$(NC)"

version: ## Show project version
	@echo "$(PROJECT_NAME) v$(VERSION)"

# Security commands
security-check: ## Run security checks
	$(PIP) install safety
	safety check
	@echo "$(GREEN)✓ Security check complete$(NC)"

# Documentation commands
docs-build: ## Build documentation
	cd docs && $(MAKE) html
	@echo "$(GREEN)✓ Documentation built in docs/_build/html$(NC)"

docs-serve: ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server

# Quick commands for common workflows
quick-start: clean install-dev db-init dev ## Quick start for development

full-test: lint test-cov security-check ## Run all tests and checks

all: clean install-prod docker-build ## Full production setup