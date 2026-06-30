# MLOps Assignment Plan

## Goal
Build an end-to-end heart disease prediction MLOps project that covers data analysis, model training, experiment tracking, API serving, containerization, CI/CD, deployment, monitoring, and reporting.

## Project Scope
- Use the Heart Disease UCI dataset
- Perform EDA and preprocessing
- Train at least two classification models
- Track experiments with MLflow
- Package a reusable inference pipeline
- Serve predictions through a FastAPI app
- Containerize the app with Docker
- Add automated tests and GitHub Actions CI
- Deploy locally using Kubernetes or Docker Desktop
- Add basic monitoring and prepare a final report

## Work Plan

### Phase 1 - Setup and structure
- Create the repository structure
- Add Python environment and dependency file
- Create baseline package modules and tests
- Verify the project runs locally

### Phase 2 - Data and EDA
- Download or reference the Heart Disease dataset
- Create a data loader script
- Perform EDA with plots and summaries
- Clean and preprocess the dataset
- Save processed data for reuse

### Phase 3 - Model development
- Build preprocessing pipeline
- Train at least two models: Logistic Regression and Random Forest
- Compare metrics: accuracy, precision, recall, F1-score, ROC-AUC
- Tune hyperparameters using GridSearchCV or RandomizedSearchCV

### Phase 4 - Experiment tracking and reproducibility
- Integrate MLflow
- Log parameters, metrics, artifacts, and plots
- Save model and preprocessing pipeline artifacts
- Ensure code and environment can be reproduced

### Phase 5 - API and testing
- Build FastAPI prediction endpoint
- Add input/output schemas
- Add unit tests for preprocessing and prediction logic
- Validate API locally

### Phase 6 - Containerization and CI/CD
- Write Dockerfile and docker-compose.yml
- Add GitHub Actions workflow for linting/tests
- Keep the pipeline failing on test failures

### Phase 7 - Deployment and monitoring
- Create Kubernetes deployment/service YAML
- Deploy locally using Docker Desktop Kubernetes or Minikube
- Add logging and basic monitoring setup

### Phase 8 - Documentation and submission
- Prepare README with setup instructions
- Create architecture diagram and screenshots
- Write the final report in PDF/Markdown format
- Prepare GitHub repository and submission materials

## Deliverables
- GitHub repository with code, tests, Docker, CI workflow, and deployment files
- Jupyter notebook or scripts for EDA and training
- MLflow experiment tracking logs and artifacts
- Dockerized API service
- Local Kubernetes deployment manifests
- Final report and screenshots

## Suggested Milestones
- Milestone 1: Project scaffold and baseline tests
- Milestone 2: EDA and data preprocessing complete
- Milestone 3: Two trained models with evaluation metrics
- Milestone 4: MLflow integrated and model artifact saved
- Milestone 5: API containerized and tested locally
- Milestone 6: CI/CD and deployment ready
- Milestone 7: Final report and submission package

## Notes
- Keep the code modular and reusable
- Prefer simple, understandable implementations over overly complex ones
- Focus on correctness, documentation, and reproducibility
- Use local/free tools wherever possible to avoid cost
