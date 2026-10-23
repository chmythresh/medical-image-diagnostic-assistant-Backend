# Medical Image Diagnostic Assistant

Experimental chest X-ray classification backend using Python, PyTorch, FastAPI, and MySQL.

## Setup

1. Use Python 3.11 and create a virtual environment.
2. Install dependencies: `pip install -r backend\requirements.txt`
3. Configure MySQL and a root `.env` file with the database settings expected by `backend/api/database.py`.
4. Ensure `backend/artifacts/best_resnet18_finetuned.pt` exists.
5. Start the API: `python -m uvicorn backend.api.main:app --reload`
6. Open http://127.0.0.1:8000/docs
7. Dataset Used: Shenzhen chest X-ray dataset.

## Endpoints

- `GET /health` — health check
- `GET /model-info` — model information
- `POST /predict` — classify an uploaded JPEG or PNG image
- `GET /history?limit=20` — prediction history

## Evaluation

The fine-tuned ResNet18 achieved approximately 0.85 accuracy and 0.849 macro F1 on a 100-image test split used during model comparison. These are exploratory results, not independent clinical validation.

## Disclaimer

For educational and research use only. This project is not a medical device and must not be used to diagnose or treat disease. Predictions may be incorrect and must not replace professional medical assessment.

## Security

Never commit `.env`, datasets, virtual environments, or private medical images.

No license has been specified.
