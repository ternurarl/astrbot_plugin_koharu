@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "KEEP_COUNT=5"
set "INTERVAL_SECONDS=600"

echo [koharu-cleanup] watching %CD%
echo [koharu-cleanup] keep newest %KEEP_COUNT%, check every %INTERVAL_SECONDS%s

:loop
set /a COUNT=0

for /f "delims=" %%D in ('dir /ad /b /o-d "astrbot-koharu-*" 2^>nul') do (
    set /a COUNT+=1
    if !COUNT! GTR %KEEP_COUNT% (
        echo [koharu-cleanup] deleting %%D
        rmdir /s /q "%%D"
    )
)

if !COUNT! LEQ %KEEP_COUNT% (
    echo [koharu-cleanup] found !COUNT! projects; nothing to delete
) else (
    echo [koharu-cleanup] found !COUNT! projects; cleanup done
)

timeout /t %INTERVAL_SECONDS% /nobreak >nul
goto loop
