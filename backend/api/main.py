from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.api.database import get_db_connection
from backend.api.model_service import (
    MODEL_NAME,
    MODEL_VERSION,
    predict_image,
)

app = FastAPI(
    title="Medical Image Diagnostic Assistant API",
    description=(
        "Research API for experimental chest X-ray "
        "normal/abnormal classification."
    ),
    version="1.0.0",
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/health")
def health_check():
    """Check whether the API and database are accessible."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@app.get("/model-info")
def model_info():
    """Return model information and intended use."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "classes": ["normal", "abnormal"],
        "input_format": ["JPEG", "PNG"],
        "maximum_upload_size_mb": 10,
        "device": "cpu",
        "intended_use": "Research and educational purposes only",
        "disclaimer": (
            "This model does not provide a medical diagnosis "
            "and must not be used to make treatment decisions."
        ),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify an uploaded image and save its result to MySQL."""
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or PNG image.",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(BytesIO(contents)) as source_image:
            source_image.verify()

        with Image.open(BytesIO(contents)) as source_image:
            image = source_image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        result = predict_image(image)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs.",
        )

    connection = None
    try:
        predicted_class = str(result["predicted_class"]).lower()
        scores = result["class_scores"]
        normal_score = float(scores["normal"])
        abnormal_score = float(scores["abnormal"])

        if predicted_class not in ("normal", "abnormal"):
            raise ValueError("Unexpected predicted class.")

        if not all(
            0.0 <= score <= 1.0
            for score in (normal_score, abnormal_score)
        ):
            raise ValueError("Class scores must be between 0 and 1.")

        connection = get_db_connection()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO prediction_history (
                        image_filename,
                        predicted_class,
                        normal_score,
                        abnormal_score,
                        model_name,
                        model_version
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        (file.filename or "uploaded_image")[:255],
                        predicted_class,
                        normal_score,
                        abnormal_score,
                        MODEL_NAME,
                        MODEL_VERSION,
                    ),
                )
                connection.commit()
                result["history_id"] = cursor.lastrowid
            finally:
                cursor.close()
        finally:
            connection.close()

    except Exception:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction completed, but saving its history failed. "
                "Check the database connection and backend logs."
            ),
        )

    return result

@app.get("/history")
def prediction_history(limit: int = 20):
    """Return recent prediction records without storing image files."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100.",
        )

    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT id, image_filename, predicted_class,
                       normal_score, abnormal_score, model_name,
                       model_version, created_at
                FROM prediction_history
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            records = cursor.fetchall()
            for record in records:
                if record["created_at"] is not None:
                    record["created_at"] = record["created_at"].isoformat()
            return {"count": len(records), "records": records}
        finally:
            cursor.close()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve prediction history.",
        )
    finally:
        if connection is not None and connection.is_connected():
            connection.close()
