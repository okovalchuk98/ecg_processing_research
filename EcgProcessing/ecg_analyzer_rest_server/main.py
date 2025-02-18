import os
import sys

import_directories = [
    "Notebooks",
    "Notebooks/SubModeles",
]

for import_dir in import_directories:
    import_dir = os.path.abspath(import_dir)
    parent_dir = os.path.dirname(import_dir)
    sys.path.append(parent_dir)

from fastapi import FastAPI
from RouterModules import classify_dataset_ecg_signal 
from RouterModules import explain_ecg_classification 

app = FastAPI()

app.include_router(classify_dataset_ecg_signal.router)
app.include_router(explain_ecg_classification.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=3721)