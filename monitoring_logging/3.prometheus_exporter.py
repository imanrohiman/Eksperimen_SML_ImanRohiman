from prometheus_client import start_http_server, Counter
import time

REQUESTS = Counter("http_requests_total", "Total HTTP Requests")
SUCCESS = Counter("success_requests_total", "Successful Requests")
FAILED = Counter("failed_requests_total", "Failed Requests")

start_http_server(8001)

print("Exporter running on port 8001")

while True:
    REQUESTS.inc()
    SUCCESS.inc()
    FAILED.inc()
    print("Metric updated")
    time.sleep(1)
