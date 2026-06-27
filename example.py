#!/usr/bin/env python3
"""
Launcher script for the AI Core Infra Hub.
"""
import sys
import os
import subprocess

def check_dependencies():
    print("[*] Checking dependencies for AI Core Infra Hub...")
    try:
        import fastapi
        import uvicorn
        import sse_starlette
        import numpy
        import pydantic
        print("[+] All essential dependencies are available.")
    except ImportError as e:
        print(f"[-] Missing dependency: {e.name}")
        print("[*] Installing requirements from requirements.txt...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("[+] Successfully installed dependencies.")
        except Exception as err:
            print(f"[-] Failed to install requirements: {err}")
            sys.exit(1)

def main():
    check_dependencies()
    import uvicorn
    print("\n" + "="*60)
    print("      AI CORE INFRA SHOWCASE HUB LAUNCHER")
    print("="*60)
    print("[-] Server starting at: http://127.0.0.1:8000")
    print("[-] Open this link in your browser to interact with all 21 projects.")
    print("="*60 + "\n")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
