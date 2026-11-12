@echo off
rem The fake sd-server (tests/fakes/sd_server.py), started like the real one: config/dev-swap.yaml.
"%~dp0..\..\.venv\Scripts\python.exe" -m tests.fakes.sd_server %*
