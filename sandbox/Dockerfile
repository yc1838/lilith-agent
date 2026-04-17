FROM python:3.11-slim

# Dependencies the agent expects to be available inside run_python.
# Keep this list in sync with the tool's docstring in src/lilith_agent/tools/__init__.py.
RUN pip install --no-cache-dir --no-compile \
        requests==2.32.3 \
        beautifulsoup4==4.12.3 \
        pandas==2.2.3 \
        trafilatura==1.12.2 \
        openpyxl==3.1.5 \
        pypdf==5.1.0 \
    && rm -rf /root/.cache/pip

# sitecustomize is auto-imported by CPython at startup, before any user code.
# It installs the metadata-IP connect guard.
COPY sitecustomize.py /usr/local/lib/python3.11/site-packages/sitecustomize.py
COPY runner.py /app/runner.py

# Run as an unprivileged user. UID 1000 avoids mapping to host root under
# userns-remap and makes write attempts to the read-only rootfs fail with EACCES.
RUN useradd --uid 1000 --create-home --shell /bin/false sandbox
USER sandbox

WORKDIR /scratch

# stdin → runner.py → stdout. No args means no secret surface in `ps`.
ENTRYPOINT ["python", "-I", "/app/runner.py"]
