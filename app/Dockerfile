FROM python:3.11

# Set working directory
WORKDIR /riskCalculator

# Install dependencies
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8080

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]