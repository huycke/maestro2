# GEMINI.md

## 1. High-Level Overview & Project Charter

*   **Project Goal & Domain:** MAESTRO is a self-hosted, AI-powered research platform for managing complex research tasks in a collaborative, multi-user environment, intended for users like academics, analysts, writers, and developers.
*   **High-Level Architecture:** The project is a monorepo containing a decoupled frontend and backend. It uses a unified reverse proxy architecture with nginx to serve a FastAPI backend and a React single-page application from a single port, eliminating CORS issues.

## 2. Technology Stack & Core Libraries

*   **Languages, Runtimes & Versions:**
    *   Python 3.10
    *   TypeScript ~5.8.3
    *   Node.js (version managed by environment)
*   **Key Frameworks:**
    *   **Backend:** FastAPI
    *   **Frontend:** React, Vite
*   **State Management:**
    *   Zustand is the exclusive state management solution for the frontend. All global client-side state should be managed through Zustand stores.
*   **Styling:**
    *   **Frontend:** Tailwind CSS is the required styling methodology. All styling should be implemented using utility classes. The configuration is in `tailwind.config.js`.
*   **Testing:**
    *   **Backend:** Pytest is the designated testing framework.
    *   **Coverage:** Coverage expectations are not formally defined, but all new features should be accompanied by relevant tests.

## 3. Directory Structure & Key File Locations

*   **Project Map:**
    ```
    /
    ├── maestro_backend/         # FastAPI backend application
    │   ├── api/                 # API route definitions
    │   ├── database/            # Database models, migrations, and CRUD operations
    │   ├── services/            # Business logic and services
    │   ├── main.py              # FastAPI application entry point
    │   └── requirements.txt     # Python dependencies
    ├── maestro_frontend/        # React frontend application
    │   ├── src/                 # Frontend source code
    │   │   ├── components/      # Reusable React components
    │   │   ├── hooks/           # Custom React hooks
    │   │   └── stores/          # Zustand state management stores
    │   ├── package.json         # Node.js dependencies
    │   └── vite.config.ts       # Vite configuration
    └── docker-compose.yml       # Docker container orchestration
    ```
*   **Sources of Truth:**
    *   **Database Schemas:** `maestro_backend/database/models.py` (SQLAlchemy ORM models)
    *   **API Routes:** Files within `maestro_backend/api/` define the RESTful API endpoints using FastAPI routers.
    *   **Global Constants & Configuration:** Primarily managed via environment variables loaded from `.env` files. There is no single file for global constants.

## 4. Coding Conventions & Style Guides

*   **Formatting:**
    *   **Frontend:** Enforced by ESLint. Configuration is in `maestro_frontend/eslint.config.js`.
    *   **Backend:** No explicit formatter (like Black or Ruff) is configured. Adherence to PEP 8 is expected but not automatically enforced.
*   **Naming Conventions:**
    *   **Backend (Python):** `snake_case` for variables, functions, and modules. `PascalCase` for classes.
    *   **Frontend (TypeScript/React):** `camelCase` for variables and functions. `PascalCase` for React components and type definitions.
*   **API Design:**
    *   The API follows RESTful principles.
    *   Error responses are structured in a consistent JSON format: `{ "error": "ErrorType", "message": "Descriptive message", "type": "error_category", "technical_details": "Optional technical info" }`.
*   **Commenting:**
    *   **Backend:** Docstrings should be used for all public modules, classes, and functions, explaining their purpose, arguments, and return values.
    *   **Frontend:** Commenting is used sparingly. Code should be self-documenting, with clear naming and strong typing.

## 5. Constraints & Anti-Patterns

*   **Dependency Management:**
    *   Third-party packages must not be installed directly in the environment. They must be added to the appropriate dependency file (`maestro_backend/requirements.txt` or `maestro_frontend/package.json`) and the Docker containers must be rebuilt.
*   **Forbidden Code Practices:**
    *   **No Direct DB Access in API Routes:** API route handlers in `maestro_backend/api/` must not directly execute database queries. All database interactions should be delegated to functions in the `maestro_backend/database/crud.py` or `maestro_backend/services/` modules.
*   **Security Mandates:**
    *   **Secret Management:** All secrets (API keys, database credentials, etc.) must be managed through environment variables via the `.env` file. No secrets should be hardcoded in the source code.
    *   **Input Sanitization:** All user-provided input, especially in API endpoints, must be treated as untrusted and properly validated and sanitized before use.
*   **Architectural Boundaries:**
    *   **Frontend-Backend Separation:** The frontend application must not, under any circumstances, access the database directly. All data must be fetched and modified through the backend API.
