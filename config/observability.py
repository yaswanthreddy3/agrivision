import logfire
from config.settings import settings

_logfire_configured = False

def setup_logfire():
    global _logfire_configured
    if _logfire_configured:
        return
    _logfire_configured = True
    if settings.LOGFIRE_TOKEN:
        logfire.configure(token=settings.LOGFIRE_TOKEN, service_name="agrivision")
        logfire.instrument_requests()  # auto-traces HTTP calls (Groq API calls included)
    else:
        print("LOGFIRE_TOKEN not set — skipping observability setup")