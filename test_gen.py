#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, 'tools')
from sphinx_doc_generator import SphinxDocGenerator

plugins_dir = Path('plugins')
output_dir = Path('docs/plugins')

print("Creating generator...")
generator = SphinxDocGenerator(plugins_dir, verbose=True)

print("Discovering plugins...")
plugins = generator.discover_plugins()
print(f"Found {len(plugins)} plugins")

if len(plugins) > 0:
    print(f"\nTesting with first plugin: {plugins[0].name}")
    try:
        generator.detect_integrations()
        generator.build_reverse_dependencies()
        generator.identify_integration_patterns()
        generator.generate_plugin_rst(plugins[0], output_dir)
        print("SUCCESS!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
