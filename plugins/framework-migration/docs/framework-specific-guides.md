# Framework-Specific Migration Guides

**Version:** 1.0.3 | **Category:** framework-migration | **Type:** Reference

Quick reference for common framework migration scenarios with links to official documentation.

---

## React Migrations

### React 16 → 18
- **Official Guide**: https://react.dev/blog/2022/03/08/react-18-upgrade-guide
- **Key Changes**: Concurrent features, automatic batching, new root API
- **Codemod**: `npx react-codemod rename-unsafe-lifecycles`

### Class Components → Hooks
- **Pattern**: See [migration-patterns-library.md](./migration-patterns-library.md#react-class-to-functional-components)
- **Codemod**: `npx react-codemod class-to-hooks`

---

## Angular Migrations

### AngularJS → Angular
- **Official Guide**: https://angular.io/guide/upgrade
- **Tool**: `@angular/upgrade` (hybrid mode)
- **Strategy**: Incremental with ngUpgrade

### Angular Major Versions
- **Tool**: `ng update @angular/core @angular/cli`
- **Auto-Migration**: Angular schematics handle most breaking changes

---

## Vue Migrations

### Vue 2 → 3
- **Official Guide**: https://v3-migration.vuejs.org/
- **Tool**: `@vue/compat` (compatibility build)
- **Breaking Changes**: Composition API, multiple root elements

---

## Python Upgrades

### Python 2 → 3
- **Tool**: `2to3` or `python-modernize`
- **Key Changes**: Print function, unicode handling, division operator

### Python 3.6 → 3.11
- **Features**: Structural pattern matching, better error messages, faster
- **Tool**: `pyupgrade` for syntax modernization

---

## Node.js/JavaScript

### Node 12 → 18 LTS
- **Guide**: https://nodejs.org/en/docs/guides/updating-to-node-18
- **Key Changes**: Fetch API, test runner, V8 updates

### Webpack 4 → 5
- **Guide**: https://webpack.js.org/migrate/5/
- **Tool**: `npx webpack-cli migrate`

---

**For complete migration workflows**, see `/code-migrate` command.
