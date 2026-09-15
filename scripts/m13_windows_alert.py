#!/usr/bin/env python3
"""Display a watchdog transition through Windows notifications from WSL."""
import base64
import json
import subprocess
import sys
from xml.sax.saxutils import escape


def main():
    event = json.load(sys.stdin)
    title = escape("M13: " + event["name"] + " — " + event["state"])
    body = escape(event["detail"])
    xml = '<toast><visual><binding template="ToastGeneric"><text>' + title
    xml += '</text><text>' + body + '</text></binding></visual></toast>'
    payload = base64.b64encode(xml.encode("utf-8")).decode("ascii")
    script = """
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$doc = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD')))
$toast = [Windows.UI.Notifications.ToastNotification]::new($doc)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Windows PowerShell').Show($toast)
""".replace("PAYLOAD", payload)
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    subprocess.run(["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                    "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                   check=True, timeout=10, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
