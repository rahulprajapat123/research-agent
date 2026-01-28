"""
Database connection and session management
"""
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2.extras import RealDictCursor
from config import get_settings
from pgvector.psycopg2 import register_vector
from urllib.parse import quote

settings = get_settings()

# Normalize DB URL so passwords with special chars work without manual encoding.
def _normalize_db_url(db_url: str) -> str:
    scheme_sep = "://"
    if scheme_sep not in db_url or "@" not in db_url:
        return db_url

    scheme, rest = db_url.split(scheme_sep, 1)
    if ":" not in rest or "@" not in rest:
        return db_url

    userinfo, hostpart = rest.rsplit("@", 1)
    if ":" not in userinfo:
        return db_url

    user, password = userinfo.split(":", 1)
    if "%" in password:
        return db_url

    safe_password = quote(password, safe="")
    if safe_password == password:
        return db_url

    return f"{scheme}{scheme_sep}{user}:{safe_password}@{hostpart}"


def _get_azure_db_url() -> str:
    """Construct Azure database connection string"""
    # Use full connection string if provided
    if settings.azure_db_connection_string:
        return _normalize_db_url(settings.azure_db_connection_string)
    
    # Otherwise construct from individual components
    if not all([settings.azure_db_server, settings.azure_db_name, 
                settings.azure_db_username, settings.azure_db_password]):
        raise ValueError("Azure database configuration incomplete. Provide either "
                        "azure_db_connection_string or all of: azure_db_server, "
                        "azure_db_name, azure_db_username, azure_db_password")
    
    password = quote(settings.azure_db_password, safe="")
    
    return (
        f"postgresql://{settings.azure_db_username}:{password}@"
        f"{settings.azure_db_server}:{settings.azure_db_port}/{settings.azure_db_name}"
        f"?sslmode={settings.azure_db_ssl_mode}"
    )


_db_url = _get_azure_db_url()

# SQLAlchemy setup
engine = create_engine(
    _db_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.debug
)


@event.listens_for(engine, "connect")
def _register_vector(dbapi_connection, _):
    register_vector(dbapi_connection)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_raw_connection():
    """Get raw psycopg2 connection for pgvector operations"""
    return psycopg2.connect(_db_url)


def execute_schema_file(schema_file_path: str) -> None:
    """Execute SQL schema file to set up database"""
    with open(schema_file_path, 'r') as f:
        schema_sql = f.read()
    
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
        print(f"OK: Successfully executed schema: {schema_file_path}")
    except Exception as e:
        conn.rollback()
        print(f"ERROR: Error executing schema: {e}")
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """Test database connection and pgvector extension"""
    try:
        conn = get_raw_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            result = cur.fetchone()
            
            if result and result[0] == 'vector':
                print("OK: Database connection successful, pgvector extension enabled")
                conn.close()
                return True
            else:
                print("WARN: Database connected but pgvector extension not found")
                conn.close()
                return False
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test connection when run directly
    test_connection()
