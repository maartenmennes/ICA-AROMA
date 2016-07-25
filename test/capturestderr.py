"""
This Nose plugin captures stderr during test execution.

It's configured together with the built-in stdout capturing plugin.
You may disable it by passing ``--nocapture`` or ``-s`` (it will be disabled
together with the built-in stdout capture plugin) or ``--nocapturestderr`` to
disable only this plugin.

:Options:
  ``--nocapturestderr``
    Don't capture stderr (any stderr output will be printed immediately)

"""

# Heavily based on
# http://somethingaboutorange.com/mrl/projects/nose/0.11.1/plugins/capture.html

import logging
import os
import sys
from nose.plugins.base import Plugin
from nose.pyversion import exc_to_unicode, force_unicode
from nose.util import ln
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

log = logging.getLogger(__name__)


class CaptureStderr(Plugin):
    """
    Output capture plugin. Enabled by default. Disable with ``-s`` or
    ``--nocapture`` or ``--nocapturestderr``. This plugin captures stderr
    during test execution, appending any output captured to the error or
    failure output, should the test fail or raise an error.
    """
    enabled = True
    env_opt = 'NOSE_NOCAPTURESTDERR'
    name = 'capturestderr'
    score = 500

    def __init__(self):
        self.stderr = []
        self._buf = None

    def options(self, parser, env):
        """Register commandline options
        """
        parser.add_option(
            "--nocapturestderr", action="store_false",
            default=not env.get(self.env_opt), dest="capturestderr",
            help="Don't capture stderr (any stderr output "
            "will be printed immediately) [NOSE_NOCAPTURESTDERR]")

    def configure(self, options, conf):
        """Configure plugin. Plugin is enabled by default.
        """
        self.conf = conf
        if not options.capture or not options.capturestderr:
            self.enabled = False

    def afterTest(self, test):
        """Clear capture buffer.
        """
        self.end()
        self._buf = None

    def begin(self):
        """Replace sys.stderr with capture buffer.
        """
        self.start() # get an early handle on sys.stderr

    def beforeTest(self, test):
        """Flush capture buffer.
        """
        self.start()

    def formatError(self, test, err):
        """Add captured output to error report.
        """
        test.capturedOutput = output = self.buffer
        self._buf = None
        if not output:
            # Don't return None as that will prevent other
            # formatters from formatting and remove earlier formatters
            # formats, instead return the err we got
            return err
        ec, ev, tb = err
        return ec, self.addCaptureToErr(ev, output), tb

    def formatFailure(self, test, err):
        """Add captured output to failure report.
        """
        return self.formatError(test, err)

    def addCaptureToErr(self, ev, output):
        ev = exc_to_unicode(ev)
        output = force_unicode(output)      
        return u'\n'.join([
            ev,
            ln(u'>> begin captured stderr <<'),
            output,
            ln(u'>> end captured stderr <<')
        ])

    def start(self):
        self.stderr.append(sys.stderr)
        self._buf = StringIO()
        if (not hasattr(self._buf, 'encoding') and
                hasattr(sys.stderr, 'encoding')):
            self._buf.encoding = sys.stderr.encoding
        if (not hasattr(self._buf, 'errors') and
                hasattr(sys.stderr, 'errors')):
            self._buf.errors = sys.stderr.errors
        sys.stderr = self._buf


    def end(self):
        if self.stderr:
            sys.stderr = self.stderr.pop()

    def finalize(self, result):
        """Restore stderr.
        """
        while self.stderr:
            self.end()

    def _get_buffer(self):
        if self._buf is not None:
            return self._buf.getvalue()

    buffer = property(_get_buffer, None, None, "Captured stderr output.")

