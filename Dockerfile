
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


# Copy the main flask app file
COPY FlaskApp_SemanticSearch.py /app/

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "FlaskApp_SemanticSearch.py"]
