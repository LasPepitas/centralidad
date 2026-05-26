# Script de configuracion del entorno virtual de Python para Windows (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Configurando Entorno Virtual Python - 01-app" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Verificar si existe la carpeta del entorno virtual (.venv)
if (-not (Test-Path ".venv")) {
    Write-Host "[+] Creando entorno virtual (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "[OK] Entorno virtual creado." -ForegroundColor Green
} else {
    Write-Host "[*] El entorno virtual (.venv) ya existe." -ForegroundColor Gray
}

# 2. Rutas
$PipPath = ".venv\Scripts\pip.exe"
$PythonVenvPath = ".venv\Scripts\python.exe"

# 3. Actualizar pip
Write-Host "[+] Actualizando pip..." -ForegroundColor Yellow
Start-Process -FilePath $PythonVenvPath -ArgumentList "-m pip install --upgrade pip" -Wait -NoNewWindow
Write-Host "[OK] Pip actualizado." -ForegroundColor Green

# 4. Instalar dependencias
if (Test-Path "requirements.txt") {
    Write-Host "[+] Instalando dependencias desde requirements.txt..." -ForegroundColor Yellow
    Start-Process -FilePath $PipPath -ArgumentList "install -r requirements.txt" -Wait -NoNewWindow
    Write-Host "[OK] Dependencias instaladas con exito." -ForegroundColor Green
} else {
    Write-Warning "requirements.txt no encontrado. No se instalaron dependencias."
}

Write-Host "=============================================" -ForegroundColor Green
Write-Host "Todo listo, loco. El entorno virtual esta configurado." -ForegroundColor Green
Write-Host "Para correr la app, recorda usar:" -ForegroundColor Green
Write-Host "  .venv\Scripts\streamlit run app/app.py" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Green
