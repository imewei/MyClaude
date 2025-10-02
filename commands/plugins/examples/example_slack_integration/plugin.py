#!/usr/bin/env python3
"""
Slack Integration Plugin
=========================

Send notifications to Slack after command execution.

Configuration:
    webhook_url: Slack webhook URL (required)
    channel: Slack channel (optional)
    username: Bot username (optional)
    notify_success: Notify on success (default: true)
    notify_failure: Notify on failure (default: true)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.plugin_base import IntegrationPlugin, PluginContext, PluginResult, HookType
from api.command_api import CommandAPI


class SlackIntegrationPlugin(IntegrationPlugin):
    """Slack integration plugin"""

    def __init__(self, metadata, config=None):
        super().__init__(metadata, config)
        self.webhook_url = None
        self.connected = False

    def load(self) -> bool:
        """Load plugin"""
        self.logger.info(f"Loading {self.metadata.name} plugin")

        # Get webhook URL from config
        self.webhook_url = self.get_config('webhook_url')

        if not self.webhook_url:
            self.logger.error("Slack webhook URL not configured")
            return False

        # Register post-execution hook
        self.register_hook(HookType.POST_EXECUTION, self._post_execution_hook)
        self.register_hook(HookType.ON_ERROR, self._on_error_hook)

        return True

    def connect(self) -> bool:
        """Connect to Slack (validate webhook)"""
        if not self.webhook_url:
            return False

        self.connected = True
        self.logger.info("Slack integration ready")
        return True

    def send(self, data: Dict[str, Any]) -> bool:
        """
        Send message to Slack.

        Args:
            data: Message data

        Returns:
            True if successful
        """
        if not self.webhook_url:
            return False

        try:
            import requests

            payload = self._format_message(data)

            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                self.logger.info("Sent notification to Slack")
                return True
            else:
                self.logger.error(f"Slack API error: {response.status_code}")
                return False

        except ImportError:
            self.logger.error("requests library not installed")
            return False
        except Exception as e:
            self.logger.error(f"Error sending to Slack: {e}")
            return False

    def execute(self, context: PluginContext) -> PluginResult:
        """Execute plugin (send test message)"""
        message_data = {
            "command": context.command_name,
            "status": "test",
            "message": "Test message from Slack integration"
        }

        success = self.send(message_data)

        if success:
            return CommandAPI.success_result(
                self.metadata.name,
                {"message": "Test notification sent to Slack"}
            )
        else:
            return CommandAPI.error_result(
                self.metadata.name,
                "Failed to send notification to Slack"
            )

    def _post_execution_hook(self, context: PluginContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-execution hook to send notifications"""
        notify_success = self.get_config('notify_success', True)

        success = data.get('success', True)

        if success and notify_success:
            message_data = {
                "command": context.command_name,
                "status": "success",
                "message": f"Command {context.command_name} completed successfully"
            }
            self.send(message_data)

        return data

    def _on_error_hook(self, context: PluginContext, data: Dict[str, Any]) -> Dict[str, Any]:
        """Error hook to send failure notifications"""
        notify_failure = self.get_config('notify_failure', True)

        if notify_failure:
            message_data = {
                "command": context.command_name,
                "status": "error",
                "message": f"Command {context.command_name} failed",
                "error": data.get('error', 'Unknown error')
            }
            self.send(message_data)

        return data

    def _format_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format message for Slack"""
        username = self.get_config('username', 'Claude Code Bot')
        channel = self.get_config('channel')

        status = data.get('status', 'info')
        command = data.get('command', 'unknown')
        message = data.get('message', '')

        # Choose emoji based on status
        emoji_map = {
            'success': ':white_check_mark:',
            'error': ':x:',
            'warning': ':warning:',
            'info': ':information_source:',
            'test': ':test_tube:'
        }
        emoji = emoji_map.get(status, ':robot_face:')

        payload = {
            "username": username,
            "text": f"{emoji} *{command}*\n{message}"
        }

        if channel:
            payload["channel"] = channel

        # Add error details if present
        if 'error' in data:
            payload["attachments"] = [{
                "color": "danger",
                "fields": [{
                    "title": "Error",
                    "value": data['error'],
                    "short": False
                }]
            }]

        return payload

    def disconnect(self):
        """Disconnect from Slack"""
        self.connected = False
        self.logger.info("Slack integration disconnected")


def main():
    """Test plugin"""
    from core.plugin_base import PluginMetadata, PluginType

    metadata = PluginMetadata(
        name="slack-integration",
        version="1.0.0",
        plugin_type=PluginType.INTEGRATION,
        description="Slack integration",
        author="Test"
    )

    # Note: Requires webhook URL configuration
    config = {
        "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        "username": "Test Bot"
    }

    plugin = SlackIntegrationPlugin(metadata, config)

    if plugin.load():
        print("Slack integration loaded successfully")
        print("Note: Configure webhook_url to send actual messages")
    else:
        print("Failed to load Slack integration")

    return 0


if __name__ == "__main__":
    sys.exit(main())