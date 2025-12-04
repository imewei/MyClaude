"""
Custom Sphinx directives for Claude Code plugin documentation.

This module provides three custom Sphinx directives for documenting plugin components:

- ``.. agent::``: Documents AI agents with their descriptions and status
- ``.. command::``: Documents CLI commands with usage and priority levels
- ``.. skill::``: Documents skills with their capabilities and status

Usage Example
-------------

In your RST file::

    .. agent:: python-pro

       Master Python 3.12+ with modern features and async programming.

       **Status:** active

    .. command:: /python-scaffold
       :priority: high

       Scaffold production-ready Python projects.

       **Status:** active

    .. skill:: async-python-patterns

       Master Python asyncio and concurrent programming patterns.

       **Status:** active

Configuration
-------------

Add 'plugin_directives' to your Sphinx extensions in conf.py::

    extensions = [
        ...
        'plugin_directives',
    ]

Ensure the _ext directory is in your Python path::

    sys.path.insert(0, os.path.abspath('_ext'))

Attributes
----------
VERSION : str
    The version of this extension (1.0).
"""

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util.docutils import SphinxDirective


class AgentDirective(SphinxDirective):
    """
    Directive for documenting AI agents in Claude Code plugins.

    This directive creates a styled container for agent documentation,
    including the agent name, status badge, and description content.

    Arguments
    ---------
    name : str
        The agent name (e.g., 'python-pro', 'jax-scientist').

    Options
    -------
    status : str, optional
        The agent status (default: 'active'). Common values: 'active', 'beta', 'deprecated'.

    Content
    -------
    The directive body should contain the agent description and any additional
    documentation in reStructuredText format.

    Example
    -------
    ::

        .. agent:: python-pro
           :status: active

           Master Python 3.12+ with modern features, async programming,
           and production-ready practices.

           **Status:** active
    """

    has_content = True
    required_arguments = 1  # Agent name
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'status': directives.unchanged,
    }

    def run(self):
        agent_name = self.arguments[0]
        status = self.options.get('status', 'active')

        # Create container
        container = nodes.container()
        container['classes'].append('agent-directive')

        # Add title
        title = nodes.strong(text=f"Agent: {agent_name}")
        title_para = nodes.paragraph()
        title_para += title
        container += title_para

        # Add status badge
        status_node = nodes.inline(text=f"Status: {status}")
        status_node['classes'].append('agent-status')
        status_para = nodes.paragraph()
        status_para += status_node
        container += status_para

        # Add description from content
        self.state.nested_parse(self.content, self.content_offset, container)

        return [container]


class CommandDirective(SphinxDirective):
    """Directive for documenting CLI commands."""

    has_content = True
    required_arguments = 1  # Command name
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'status': directives.unchanged,
        'priority': directives.unchanged,
    }

    def run(self):
        command_name = self.arguments[0]
        status = self.options.get('status', 'active')
        priority = self.options.get('priority', None)

        # Create container
        container = nodes.container()
        container['classes'].append('command-directive')

        # Add title
        title = nodes.literal(text=command_name)
        title_para = nodes.paragraph()
        title_para += nodes.strong(text="Command: ")
        title_para += title
        container += title_para

        # Add status and priority
        meta_para = nodes.paragraph()
        status_node = nodes.inline(text=f"Status: {status}")
        status_node['classes'].append('command-status')
        meta_para += status_node

        if priority:
            meta_para += nodes.Text(" | ")
            priority_node = nodes.inline(text=f"Priority: {priority}")
            priority_node['classes'].append('command-priority')
            meta_para += priority_node

        container += meta_para

        # Add description from content
        self.state.nested_parse(self.content, self.content_offset, container)

        return [container]


class SkillDirective(SphinxDirective):
    """Directive for documenting skills."""

    has_content = True
    required_arguments = 1  # Skill name
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'status': directives.unchanged,
    }

    def run(self):
        skill_name = self.arguments[0]
        status = self.options.get('status', 'active')

        # Create container
        container = nodes.container()
        container['classes'].append('skill-directive')

        # Add title
        title = nodes.strong(text=f"Skill: {skill_name}")
        title_para = nodes.paragraph()
        title_para += title
        container += title_para

        # Add status
        status_node = nodes.inline(text=f"Status: {status}")
        status_node['classes'].append('skill-status')
        status_para = nodes.paragraph()
        status_para += status_node
        container += status_para

        # Add description from content
        self.state.nested_parse(self.content, self.content_offset, container)

        return [container]


def setup(app):
    """Setup function for Sphinx extension."""
    app.add_directive('agent', AgentDirective)
    app.add_directive('command', CommandDirective)
    app.add_directive('skill', SkillDirective)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
