#!/bin/bash

# SSL Certificate Setup Script with Let's Encrypt
# TEKNOFEST 2025 - Production SSL Configuration

set -e

# Configuration
DOMAIN="teknofest2025.com"
EMAIL="admin@teknofest2025.com"
NGINX_SSL_DIR="./nginx/ssl"
CERTBOT_DIR="./certbot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TEKNOFEST 2025 - SSL Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to check if running in production
check_environment() {
    if [ "$APP_ENV" != "production" ]; then
        echo -e "${YELLOW}Warning: Not in production environment${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to create directories
create_directories() {
    echo -e "${GREEN}Creating SSL directories...${NC}"
    mkdir -p $NGINX_SSL_DIR
    mkdir -p $CERTBOT_DIR/www
    mkdir -p $CERTBOT_DIR/conf
}

# Function to generate self-signed certificate (for testing)
generate_self_signed() {
    echo -e "${YELLOW}Generating self-signed certificate for testing...${NC}"
    
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout $NGINX_SSL_DIR/privkey.pem \
        -out $NGINX_SSL_DIR/fullchain.pem \
        -subj "/C=TR/ST=Istanbul/L=Istanbul/O=TEKNOFEST/OU=IT/CN=$DOMAIN"
    
    echo -e "${GREEN}Self-signed certificate generated${NC}"
}

# Function to generate DH parameters
generate_dhparam() {
    echo -e "${GREEN}Generating DH parameters (this may take a while)...${NC}"
    
    if [ ! -f "$NGINX_SSL_DIR/dhparam.pem" ]; then
        openssl dhparam -out $NGINX_SSL_DIR/dhparam.pem 2048
        echo -e "${GREEN}DH parameters generated${NC}"
    else
        echo -e "${YELLOW}DH parameters already exist${NC}"
    fi
}

# Function to obtain Let's Encrypt certificate
get_letsencrypt_cert() {
    echo -e "${GREEN}Obtaining Let's Encrypt certificate...${NC}"
    
    # Check if domain is provided
    if [ -z "$DOMAIN" ]; then
        echo -e "${RED}Error: DOMAIN not set${NC}"
        exit 1
    fi
    
    # Check if email is provided
    if [ -z "$EMAIL" ]; then
        echo -e "${RED}Error: EMAIL not set${NC}"
        exit 1
    fi
    
    # Create docker-compose for certbot
    cat > docker-compose.certbot.yml <<EOF
version: '3.8'

services:
  nginx-certbot:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./certbot/www:/var/www/certbot:ro
      - ./nginx/nginx.certbot.conf:/etc/nginx/nginx.conf:ro
    networks:
      - certbot-network

  certbot:
    image: certbot/certbot
    depends_on:
      - nginx-certbot
    volumes:
      - ./certbot/www:/var/www/certbot
      - ./certbot/conf:/etc/letsencrypt
    networks:
      - certbot-network
    command: certonly --webroot -w /var/www/certbot --email ${EMAIL} --agree-tos --no-eff-email -d ${DOMAIN} -d www.${DOMAIN}

networks:
  certbot-network:
    driver: bridge
EOF

    # Create temporary nginx config for certbot
    cat > ./nginx/nginx.certbot.conf <<EOF
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name ${DOMAIN} www.${DOMAIN};

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 301 https://\$server_name\$request_uri;
        }
    }
}
EOF

    # Run certbot
    docker-compose -f docker-compose.certbot.yml up
    
    # Copy certificates to nginx directory
    cp ./certbot/conf/live/${DOMAIN}/fullchain.pem $NGINX_SSL_DIR/
    cp ./certbot/conf/live/${DOMAIN}/privkey.pem $NGINX_SSL_DIR/
    
    echo -e "${GREEN}Let's Encrypt certificate obtained${NC}"
    
    # Clean up
    docker-compose -f docker-compose.certbot.yml down
    rm docker-compose.certbot.yml
    rm ./nginx/nginx.certbot.conf
}

# Function to setup certificate renewal
setup_renewal() {
    echo -e "${GREEN}Setting up automatic certificate renewal...${NC}"
    
    # Create renewal script
    cat > ./scripts/renew_ssl.sh <<'EOF'
#!/bin/bash

# SSL Certificate Renewal Script
DOMAIN="teknofest2025.com"
CERTBOT_DIR="./certbot"
NGINX_SSL_DIR="./nginx/ssl"

# Renew certificate
docker run --rm \
    -v $(pwd)/certbot/conf:/etc/letsencrypt \
    -v $(pwd)/certbot/www:/var/www/certbot \
    certbot/certbot renew --quiet

# Copy renewed certificates
if [ $? -eq 0 ]; then
    cp $CERTBOT_DIR/conf/live/$DOMAIN/fullchain.pem $NGINX_SSL_DIR/
    cp $CERTBOT_DIR/conf/live/$DOMAIN/privkey.pem $NGINX_SSL_DIR/
    
    # Reload nginx
    docker-compose exec nginx nginx -s reload
    
    echo "Certificate renewed successfully"
else
    echo "Certificate renewal failed"
    exit 1
fi
EOF

    chmod +x ./scripts/renew_ssl.sh
    
    # Add to crontab (runs twice daily)
    echo -e "${YELLOW}Add this line to your crontab (crontab -e):${NC}"
    echo "0 0,12 * * * cd $(pwd) && ./scripts/renew_ssl.sh >> ./logs/ssl_renewal.log 2>&1"
    
    echo -e "${GREEN}Renewal script created${NC}"
}

# Function to verify SSL setup
verify_ssl() {
    echo -e "${GREEN}Verifying SSL setup...${NC}"
    
    if [ -f "$NGINX_SSL_DIR/fullchain.pem" ] && [ -f "$NGINX_SSL_DIR/privkey.pem" ]; then
        echo -e "${GREEN}✓ SSL certificates found${NC}"
        
        # Check certificate expiry
        openssl x509 -in $NGINX_SSL_DIR/fullchain.pem -noout -dates
    else
        echo -e "${RED}✗ SSL certificates not found${NC}"
        return 1
    fi
    
    if [ -f "$NGINX_SSL_DIR/dhparam.pem" ]; then
        echo -e "${GREEN}✓ DH parameters found${NC}"
    else
        echo -e "${YELLOW}⚠ DH parameters not found (optional)${NC}"
    fi
}

# Main execution
main() {
    echo -e "${GREEN}SSL Setup Options:${NC}"
    echo "1. Production (Let's Encrypt)"
    echo "2. Testing (Self-signed)"
    echo "3. Verify existing setup"
    read -p "Choose option (1-3): " option
    
    case $option in
        1)
            check_environment
            create_directories
            get_letsencrypt_cert
            generate_dhparam
            setup_renewal
            verify_ssl
            ;;
        2)
            create_directories
            generate_self_signed
            generate_dhparam
            verify_ssl
            ;;
        3)
            verify_ssl
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}SSL setup complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo "1. Update docker-compose.yml to use nginx.ssl.conf"
    echo "2. Update .env file with SSL_ENABLED=true"
    echo "3. Restart services with: docker-compose up -d"
    echo "4. Test SSL at: https://www.ssllabs.com/ssltest/"
}

# Run main function
main