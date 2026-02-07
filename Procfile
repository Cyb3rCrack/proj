web: gunicorn --bind 0.0.0.0:$PORT --workers 4 --worker-class sync --timeout 120 --access-logfile - --error-logfile - wsgi:app
