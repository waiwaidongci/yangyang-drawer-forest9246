from src.config.database import Base, engine
from src.models import account, dataset, review, run, sampling  # noqa: F401


def bootstrap():
    Base.metadata.create_all(engine)
