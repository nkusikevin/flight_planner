#!/bin/bash

# deploy.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display error messages
error() {
    echo "ERROR: $1" >&2
    exit 1
}

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    error "OPENAI_API_KEY is not set. Please set it in your environment."
fi

# Set default values
PROJECT_ID="ai-project-431615"
REGION="us-central1"
SERVICE_NAME="flying"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
echo "OPenAI_API_KEY: $OPENAI_API_KEY"
gcloud run deploy $SERVICE_NAME \
    --source . \
    --port 8080 \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --region $REGION \
    --set-env-vars=OPENAI_API_KEY=$OPENAI_API_KEY \
    || error "Deployment failed"

echo "Deployment completed successfully!"

# Get the URL of the deployed service
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
echo "Your service is available at: $SERVICE_URL"