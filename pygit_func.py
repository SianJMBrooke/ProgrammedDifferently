from io import StringIO
import re
import time
from datetime import datetime
from pylint.lint import Run
from pylint.reporters import text
from astroid import MANAGER
import pandas as pd


def pylint_test(testfile):
    """
    Runs the pylint assessment on a python file.
    :param testfile: .py filename as string
    :return: Dictionary of the report
    """

    # Clear cache
    MANAGER.astroid_cache.clear()

    # Get Assessment of Code
    pylint_output = StringIO()
    reporter = text.TextReporter(pylint_output)
    Run([testfile, "--reports=y", "--enable=all"], reporter=reporter, do_exit=False)
    pylint_output.seek(0)
    pylint_report = pylint_output.read()

    # Regex to extract data from the pylint report
    labels = [i.strip() for i in re.findall(r"(?<=\|)[a-z- ]+(?=\s+\|[0-9+])", pylint_report)]
    label_nums = [i.strip() for i in re.findall(r"(?<=\s\|)([0-9.= |]+)(?=\|\n)", pylint_report)]
    res = [char.replace(" ", "").split("|") for char in label_nums]

    # Dictionary of Vars
    report_dict = {labels[i]: res[i] for i in range(len(labels))}
    report_dict["statements analysed"] = re.findall(r"([0-9]+)(?=\s[statements])", pylint_report)
    report_dict["pylint rating"] = re.findall(r"([0-9].+)(?=/10\s)", pylint_report)
    report_dict["pylint verbose"] = re.findall(r"(python-test\.py.*)", pylint_report)

    # Output as dictionary
    return report_dict


def github_ratelimit(g):
    print("\n***** Rate Limit Reached *****")
    print(time.strftime("\t%l:%M%p %Z on %b %d, %Y"))

    core_rate_limit = g.get_rate_limit().core
    now = datetime.now()
    pause = core_rate_limit.reset - now
    sleep_time = pause.seconds + 5
    print("\tPausing for:", round(sleep_time / 60), "mins\n")

    time.sleep(sleep_time)
