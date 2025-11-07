# AST Parsing Implementation

**Version**: 1.0.3
**Category**: code-documentation
**Purpose**: Python, JavaScript, TypeScript, Go, and Rust AST extraction implementations

## Overview

Comprehensive AST parsing implementations for extracting code structure, docstrings, type hints, and documentation from multiple programming languages.

## Python AST Extraction

### Complete Module Analysis

```python
import ast
import inspect
from typing import Dict, List

class PythonASTExtractor:
    def extract_module_structure(self, file_path: str) -> Dict:
        """
        Extract complete module structure including classes, functions, and docstrings

        Returns:
            Dictionary with modules, classes, functions, and their documentation
        """
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read(), filename=file_path)

        structure = {
            'file': file_path,
            'module_docstring': ast.get_docstring(tree),
            'imports': self._extract_imports(tree),
            'classes': self._extract_classes(tree),
            'functions': self._extract_functions(tree),
            'constants': self._extract_constants(tree)
        }

        return structure

    def _extract_classes(self, tree) -> List[Dict]:
        """Extract all class definitions with methods and docstrings"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'bases': [ast.unparse(base) for base in node.bases],
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'methods': [],
                    'attributes': [],
                    'lineno': node.lineno
                }

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'docstring': ast.get_docstring(item),
                            'parameters': self._extract_function_params(item),
                            'returns': ast.unparse(item.returns) if item.returns else None,
                            'decorators': [ast.unparse(dec) for dec in item.decorator_list],
                            'is_async': isinstance(item, ast.AsyncFunctionDef),
                            'lineno': item.lineno
                        }
                        class_info['methods'].append(method_info)

                    # Extract class attributes
                    elif isinstance(item, ast.AnnAssign):
                        attr_info = {
                            'name': item.target.id if hasattr(item.target, 'id') else None,
                            'type': ast.unparse(item.annotation),
                            'value': ast.unparse(item.value) if item.value else None
                        }
                        class_info['attributes'].append(attr_info)

                classes.append(class_info)

        return classes

    def _extract_functions(self, tree) -> List[Dict]:
        """Extract all module-level functions"""
        functions = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node),
                    'parameters': self._extract_function_params(node),
                    'returns': ast.unparse(node.returns) if node.returns else None,
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'lineno': node.lineno
                }
                functions.append(func_info)

        return functions

    def _extract_function_params(self, func_node) -> List[Dict]:
        """Extract function parameters with type hints and defaults"""
        params = []

        args = func_node.args
        defaults = [None] * (len(args.args) - len(args.defaults)) + args.defaults

        for arg, default in zip(args.args, defaults):
            param_info = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else None,
                'default': ast.unparse(default) if default else None,
                'kind': 'positional'
            }
            params.append(param_info)

        # Handle *args
        if args.vararg:
            params.append({
                'name': args.vararg.arg,
                'type': ast.unparse(args.vararg.annotation) if args.vararg.annotation else None,
                'kind': 'var_positional'
            })

        # Handle **kwargs
        if args.kwarg:
            params.append({
                'name': args.kwarg.arg,
                'type': ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else None,
                'kind': 'var_keyword'
            })

        return params

    def _extract_imports(self, tree) -> List[Dict]:
        """Extract all import statements"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        'module': node.module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'type': 'from_import'
                    })

        return imports

    def _extract_constants(self, tree) -> List[Dict]:
        """Extract module-level constants"""
        constants = []

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            'name': target.id,
                            'value': ast.unparse(node.value)
                        })

        return constants
```

## JavaScript/TypeScript AST Extraction

### TypeScript Structure Extraction

```javascript
// Using @babel/parser or typescript compiler API
const ts = require('typescript');
const fs = require('fs');

class TypeScriptExtractor {
    extractStructure(filePath) {
        const sourceCode = fs.readFileSync(filePath, 'utf-8');
        const sourceFile = ts.createSourceFile(
            filePath,
            sourceCode,
            ts.ScriptTarget.Latest,
            true
        );

        const structure = {
            file: filePath,
            exports: [],
            functions: [],
            classes: [],
            interfaces: [],
            types: []
        };

        this.visit(sourceFile, structure);
        return structure;
    }

    visit(node, structure) {
        switch (node.kind) {
            case ts.SyntaxKind.FunctionDeclaration:
                structure.functions.push(this.extractFunction(node));
                break;

            case ts.SyntaxKind.ClassDeclaration:
                structure.classes.push(this.extractClass(node));
                break;

            case ts.SyntaxKind.InterfaceDeclaration:
                structure.interfaces.push(this.extractInterface(node));
                break;

            case ts.SyntaxKind.TypeAliasDeclaration:
                structure.types.push(this.extractTypeAlias(node));
                break;
        }

        ts.forEachChild(node, child => this.visit(child, structure));
    }

    extractFunction(node) {
        return {
            name: node.name ? node.name.text : 'anonymous',
            parameters: node.parameters.map(p => ({
                name: p.name.text,
                type: p.type ? this.getTypeText(p.type) : 'any',
                optional: !!p.questionToken
            })),
            returnType: node.type ? this.getTypeText(node.type) : 'any',
            isAsync: !!node.modifiers?.find(m => m.kind === ts.SyntaxKind.AsyncKeyword),
            isExported: !!node.modifiers?.find(m => m.kind === ts.SyntaxKind.ExportKeyword),
            jsDoc: this.extractJSDoc(node)
        };
    }

    extractClass(node) {
        return {
            name: node.name.text,
            members: node.members.map(m => this.extractMember(m)),
            heritage: node.heritageClauses?.map(h => this.getTypeText(h)) || [],
            isExported: !!node.modifiers?.find(m => m.kind === ts.SyntaxKind.ExportKeyword),
            jsDoc: this.extractJSDoc(node)
        };
    }

    extractInterface(node) {
        return {
            name: node.name.text,
            properties: node.members.map(m => ({
                name: m.name.text,
                type: this.getTypeText(m.type),
                optional: !!m.questionToken
            })),
            jsDoc: this.extractJSDoc(node)
        };
    }

    extractJSDoc(node) {
        const jsDoc = node.jsDoc?.[0];
        if (!jsDoc) return null;

        return {
            comment: jsDoc.comment || '',
            tags: jsDoc.tags?.map(tag => ({
                tagName: tag.tagName.text,
                comment: tag.comment || ''
            })) || []
        };
    }

    getTypeText(typeNode) {
        if (!typeNode) return 'any';
        return typeNode.getText();
    }
}
```

## Go AST Extraction

### Go Package Structure

```go
package main

import (
    "go/ast"
    "go/parser"
    "go/token"
)

type GoExtractor struct {
    fset *token.FileSet
}

func (e *GoExtractor) ExtractPackage(path string) (*PackageStructure, error) {
    e.fset = token.NewFileSet()

    pkgs, err := parser.ParseDir(e.fset, path, nil, parser.ParseComments)
    if err != nil {
        return nil, err
    }

    structure := &PackageStructure{
        Functions: []FunctionInfo{},
        Types:     []TypeInfo{},
        Constants: []ConstantInfo{},
    }

    for _, pkg := range pkgs {
        for _, file := range pkg.Files {
            ast.Inspect(file, func(n ast.Node) bool {
                switch x := n.(type) {
                case *ast.FuncDecl:
                    structure.Functions = append(structure.Functions, e.extractFunction(x))
                case *ast.GenDecl:
                    if x.Tok == token.TYPE {
                        for _, spec := range x.Specs {
                            if ts, ok := spec.(*ast.TypeSpec); ok {
                                structure.Types = append(structure.Types, e.extractType(ts, x.Doc))
                            }
                        }
                    }
                }
                return true
            })
        }
    }

    return structure, nil
}

func (e *GoExtractor) extractFunction(fn *ast.FuncDecl) FunctionInfo {
    info := FunctionInfo{
        Name:    fn.Name.Name,
        Doc:     fn.Doc.Text(),
        Params:  e.extractParams(fn.Type.Params),
        Results: e.extractParams(fn.Type.Results),
    }

    if fn.Recv != nil {
        info.Receiver = e.extractParams(fn.Recv)[0]
    }

    return info
}
```

## Rust AST Extraction

### Using syn Crate

```rust
use syn::{File, Item, ItemFn, ItemStruct, ItemEnum};
use std::fs;

pub struct RustExtractor;

impl RustExtractor {
    pub fn extract_structure(file_path: &str) -> Result<ModuleStructure, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(file_path)?;
        let syntax_tree = syn::parse_file(&content)?;

        let mut structure = ModuleStructure {
            functions: Vec::new(),
            structs: Vec::new(),
            enums: Vec::new(),
            traits: Vec::new(),
        };

        for item in syntax_tree.items {
            match item {
                Item::Fn(func) => structure.functions.push(extract_function(func)),
                Item::Struct(strukt) => structure.structs.push(extract_struct(strukt)),
                Item::Enum(enm) => structure.enums.push(extract_enum(enm)),
                _ => {}
            }
        }

        Ok(structure)
    }
}

fn extract_function(func: ItemFn) -> FunctionInfo {
    FunctionInfo {
        name: func.sig.ident.to_string(),
        visibility: format!("{:?}", func.vis),
        parameters: func.sig.inputs.iter().map(|arg| {
            // Extract parameter info
            format!("{:?}", arg)
        }).collect(),
        return_type: match func.sig.output {
            syn::ReturnType::Type(_, ty) => format!("{:?}", ty),
            _ => String::from("()"),
        },
        doc_comments: extract_doc_comments(&func.attrs),
    }
}
```

## Usage Examples

### Python Complete Analysis

```python
extractor = PythonASTExtractor()
structure = extractor.extract_module_structure('mymodule.py')

print(f"Module: {structure['file']}")
print(f"Classes: {len(structure['classes'])}")
print(f"Functions: {len(structure['functions'])}")

for cls in structure['classes']:
    print(f"\nClass: {cls['name']}")
    print(f"Docstring: {cls['docstring'][:100] if cls['docstring'] else 'None'}")
    print(f"Methods: {[m['name'] for m in cls['methods']]}")
```

### TypeScript Analysis

```javascript
const extractor = new TypeScriptExtractor();
const structure = extractor.extractStructure('app.ts');

console.log(`Exports: ${structure.exports.length}`);
console.log(`Functions: ${structure.functions.length}`);
console.log(`Interfaces: ${structure.interfaces.length}`);
```
