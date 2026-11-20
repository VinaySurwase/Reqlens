import os

from sqlalchemy import create_engine

from alembic import context
from reqlens.config import get_settings

url = os.environ.get("ALEMBIC_URL") or get_settings().database_url


def run() -> None:
    if context.is_offline_mode():
        context.configure(url=url, literal_binds=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()
        conn.commit()


run()
