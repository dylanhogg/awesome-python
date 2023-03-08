import http.server
import socketserver
import time
from loguru import logger

PORT = 8002
DIRECTORY = "app"
DEFAULT_PAGE = "app.html"


class CustomHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        logger.trace(f"GET: {self.path=}")
        if self.path == "/":
            self.path = DEFAULT_PAGE
        if "/?" in self.path:
            self.path = self.path.replace("/?", f"/{DEFAULT_PAGE}?")
        return http.server.SimpleHTTPRequestHandler.do_GET(self)


with socketserver.TCPServer(("", PORT), CustomHttpRequestHandler) as httpd:
    logger.info(f"Custom http.server on port {PORT} from dir {DIRECTORY} with default page {DEFAULT_PAGE}. "
                f"Press CTRL+C to close the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        httpd.server_close()
        time.sleep(1)
        logger.info(f"Server on port {PORT} closed.")
