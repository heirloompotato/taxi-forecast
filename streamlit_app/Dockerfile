FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all application files
COPY . .

# Set environment variables
ENV PORT=8080

# Command to run the application
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0