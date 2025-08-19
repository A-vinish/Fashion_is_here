@echo off
start cmd /k "cd /d A:\Ml Project\Fashion_Is_Here && venv\Scripts\activate && python app.py"
timeout /t 5
start cmd /k "cd /d A:\Ml Project\Fashion_Is_Here && python -m http.server 3000"
//This opens one terminal to run your backend, waits 5 seconds, then starts your frontend server in a second terminal.

//Adjust/remove the venv\Scripts\activate part if not using a virtual environment.

//Double-click run_project.bat to start both servers at once!