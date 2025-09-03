from flask import request, jsonify
from sqlalchemy import select

from app import app, Session, engine, _MODEL_CATEGORIES, Classification


@app.get("/categories")
def get_categories():
    """
    Return available categories.

    By default, return all categories recorded in the database (distinct).
    To return only pretrained model categories, pass ?model=true in the query string.
    """
    use_model = request.args.get("model", "false").lower() == "true"
    if use_model and _MODEL_CATEGORIES:
        cats = _MODEL_CATEGORIES
    else:
        with Session(engine) as s:
            cats = (
                s.execute(select(Classification.category).distinct()).scalars().all()
            )
    return jsonify({"categories": cats})
