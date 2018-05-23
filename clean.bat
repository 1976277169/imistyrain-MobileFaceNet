@echo off
del bin\*.pdb
del windows\*.pdb
if exist windows\Release rd windows\Release /s /q
pause