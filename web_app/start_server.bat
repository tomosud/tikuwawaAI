@echo off
echo Starting Chihuahua vs Chikuwa AI Classifier Web Server...
echo.

REM Change to the web_app directory
cd /d "%~dp0"

REM Start Python HTTP server on port 8000
echo Starting Python web server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

REM Start the server and open the browser
start "" "http://localhost:8000"
python -m http.server 8000

pause