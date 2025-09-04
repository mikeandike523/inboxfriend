from flask import jsonify
from sqlalchemy import select

from preamble import Session, engine
from models import Classification


def get_categories():
    """
    Return available categories.

    By default, return all categories recorded in the database (distinct).
    To return only pretrained model categories, pass ?model=true in the query string.
    """
    with Session(engine) as s:
        cats = (
            s.execute(select(Classification.category).distinct()).scalars().all()
        )
    return jsonify({"categories": cats})
