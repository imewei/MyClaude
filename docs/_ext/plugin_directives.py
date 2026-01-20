from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

class BasePluginDirective(SphinxDirective):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'description': directives.unchanged,
        'model': directives.unchanged,
        'version': directives.unchanged,
    }

    def run(self):
        name = self.arguments[0]
        description = self.options.get('description', '')

        # Create a section or admonition-like structure
        container = nodes.admonition()
        container['classes'].append(self.node_class)

        title = nodes.title(text=f"{self.label}: {name}")
        container += title

        if description:
            desc_paragraph = nodes.paragraph(text=description)
            container += desc_paragraph

        if 'model' in self.options:
            model_node = nodes.paragraph()
            model_node += nodes.strong(text="Model: ")
            model_node += nodes.Text(self.options['model'])
            container += model_node

        if 'version' in self.options:
            version_node = nodes.paragraph()
            version_node += nodes.strong(text="Version: ")
            version_node += nodes.Text(self.options['version'])
            container += version_node

        return [container]

class AgentDirective(BasePluginDirective):
    node_class = 'agent-directive'
    label = 'Agent'

class CommandDirective(BasePluginDirective):
    node_class = 'command-directive'
    label = 'Command'

class SkillDirective(BasePluginDirective):
    node_class = 'skill-directive'
    label = 'Skill'

def setup(app):
    app.add_directive('agent', AgentDirective)
    app.add_directive('command', CommandDirective)
    app.add_directive('skill', SkillDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
