"""Background job / queue helpers (platform scaffold).

chrono_origin ships an in-memory JobManager and spokenverse ships a durable
Postgres-leased worker. A shared abstraction over both belongs here.
"""
