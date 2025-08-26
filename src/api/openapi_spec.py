"""
OpenAPI/Swagger Specification Generator
TEKNOFEST 2025 - API Documentation

This module configures automatic API documentation for FastAPI.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse

def custom_openapi(app: FastAPI):
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title="TEKNOFEST 2025 Educational Platform API",
        version="1.0.0",
        description="""
        ## 🚀 TEKNOFEST 2025 Educational Technology Platform
        
        A comprehensive API for managing educational content, student assessments, 
        and AI-powered learning experiences.
        
        ### 🔑 Authentication
        This API uses JWT Bearer tokens. Include the token in the Authorization header:
        ```
        Authorization: Bearer <your-token>
        ```
        
        ### 📚 Main Features
        - **User Management**: Registration, authentication, and profile management
        - **Learning Paths**: Personalized curriculum and progress tracking
        - **Assessments**: Quizzes, exams, and performance analytics
        - **AI Assistant**: Intelligent tutoring and question generation
        - **Gamification**: Points, badges, and leaderboards
        - **Resources**: Educational content and materials
        
        ### 🔒 Rate Limiting
        - Authentication endpoints: 5 req/min
        - AI endpoints: 30 req/min  
        - General endpoints: 100 req/min
        
        ### 📊 Response Format
        All responses follow a consistent JSON structure with appropriate HTTP status codes.
        
        ### 🆘 Support
        - Documentation: https://docs.teknofest2025.com
        - Email: api-support@teknofest2025.com
        """,
        routes=app.routes,
        tags=[
            {
                "name": "Authentication",
                "description": "User registration, login, and token management"
            },
            {
                "name": "Users",
                "description": "User profile and settings management"
            },
            {
                "name": "Learning Paths",
                "description": "Curriculum management and progress tracking"
            },
            {
                "name": "Assessments",
                "description": "Quizzes, exams, and evaluations"
            },
            {
                "name": "AI Assistant",
                "description": "AI-powered tutoring and content generation"
            },
            {
                "name": "Progress",
                "description": "Learning analytics and performance tracking"
            },
            {
                "name": "Gamification",
                "description": "Points, achievements, and leaderboards"
            },
            {
                "name": "Resources",
                "description": "Educational materials and content"
            },
            {
                "name": "Admin",
                "description": "Administrative functions (requires admin role)"
            },
            {
                "name": "Health",
                "description": "Service health and status checks"
            }
        ],
        servers=[
            {
                "url": "https://api.teknofest2025.com",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.teknofest2025.com",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT authorization token"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service communication"
        }
    }
    
    # Add common responses
    openapi_schema["components"]["responses"] = {
        "BadRequest": {
            "description": "Bad request",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string"},
                                    "message": {"type": "string"},
                                    "details": {"type": "array"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "Unauthorized": {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string", "example": "UNAUTHORIZED"},
                                    "message": {"type": "string", "example": "Invalid or expired token"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "Forbidden": {
            "description": "Insufficient permissions",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string", "example": "FORBIDDEN"},
                                    "message": {"type": "string", "example": "You don't have permission to access this resource"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "NotFound": {
            "description": "Resource not found",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string", "example": "NOT_FOUND"},
                                    "message": {"type": "string", "example": "The requested resource was not found"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "RateLimited": {
            "description": "Too many requests",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string", "example": "RATE_LIMITED"},
                                    "message": {"type": "string", "example": "Too many requests. Please try again later."},
                                    "retry_after": {"type": "integer", "example": 60}
                                }
                            }
                        }
                    }
                }
            }
        },
        "ServerError": {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string", "example": "INTERNAL_ERROR"},
                                    "message": {"type": "string", "example": "An unexpected error occurred"},
                                    "request_id": {"type": "string", "example": "req-123456"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    # Add example schemas
    openapi_schema["components"]["schemas"].update({
        "User": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "format": "uuid"},
                "username": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "full_name": {"type": "string"},
                "grade": {"type": "integer", "minimum": 1, "maximum": 12},
                "points": {"type": "integer"},
                "level": {"type": "integer"},
                "created_at": {"type": "string", "format": "date-time"}
            }
        },
        "LoginRequest": {
            "type": "object",
            "required": ["username", "password"],
            "properties": {
                "username": {"type": "string"},
                "password": {"type": "string", "format": "password"}
            }
        },
        "TokenResponse": {
            "type": "object",
            "properties": {
                "access_token": {"type": "string"},
                "refresh_token": {"type": "string"},
                "token_type": {"type": "string", "default": "bearer"},
                "expires_in": {"type": "integer"}
            }
        },
        "Quiz": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "subject": {"type": "string"},
                "topic": {"type": "string"},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}},
                            "points": {"type": "integer"}
                        }
                    }
                }
            }
        },
        "PaginatedResponse": {
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {}},
                "total": {"type": "integer"},
                "page": {"type": "integer"},
                "per_page": {"type": "integer"},
                "pages": {"type": "integer"}
            }
        }
    })
    
    # Apply security globally (except for specific endpoints)
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method in ["get", "post", "put", "delete", "patch"]:
                # Skip auth for health checks and auth endpoints
                if not any(skip in path for skip in ["/health", "/auth/login", "/auth/register", "/docs"]):
                    openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def configure_swagger_ui(app: FastAPI):
    """Configure Swagger UI with custom settings."""
    app.openapi = lambda: custom_openapi(app)
    
    # Custom Swagger UI configuration
    @app.get("/", include_in_schema=False)
    async def custom_swagger_ui_html():
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>TEKNOFEST 2025 API</title>
                <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
                <style>
                    body { margin: 0; padding: 0; }
                    .swagger-ui .topbar { display: none; }
                    .swagger-ui .info { margin: 20px 0; }
                    .swagger-ui .info .title { color: #2c3e50; }
                    .swagger-ui .btn.authorize { background-color: #3498db; }
                    .swagger-ui .btn.authorize:hover { background-color: #2980b9; }
                </style>
            </head>
            <body>
                <div id="swagger-ui"></div>
                <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
                <script>
                    window.onload = function() {
                        window.ui = SwaggerUIBundle({
                            url: "/openapi.json",
                            dom_id: '#swagger-ui',
                            deepLinking: true,
                            presets: [
                                SwaggerUIBundle.presets.apis,
                                SwaggerUIBundle.SwaggerUIStandalonePreset
                            ],
                            layout: "BaseLayout",
                            persistAuthorization: true,
                            tryItOutEnabled: true,
                            requestSnippetsEnabled: true,
                            docExpansion: "none",
                            filter: true,
                            showExtensions: true,
                            showCommonExtensions: true
                        });
                    };
                </script>
            </body>
            </html>
            """,
            status_code=200
        )