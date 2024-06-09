import os
import webbrowser

from ansi2html import Ansi2HTMLConverter
from invoke.tasks import task

from treatment_checker.build_db.build_db import build_db


@task
def build_database(c):
    build_db()


@task
def display_log(c, file):
    html_log_path = f"data/logs/{file}.html"
    raw_log_path = f"data/logs/{file}.log"

    # Convert raw log file to HTML file if it does not exist already.
    if not os.path.exists(html_log_path):
        print("No HTML version of log file found. Converting.")
        converter = Ansi2HTMLConverter()
        with open(raw_log_path, "r") as raw_log_file:
            log_html = converter.convert(raw_log_file.read(), full=True)
        with open(html_log_path, "w") as html_log_file:
            html_log_file.write(log_html)

    # Display HTML log file in browser using absolute path.
    html_log_abspath = os.path.abspath(html_log_path)
    webbrowser.open(f"file://{html_log_abspath}")
